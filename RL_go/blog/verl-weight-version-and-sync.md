# RL 系统基础第 1 课：权重版本与训推同步

> 状态：讲解与 CPU 实验完成；三道课后题已回答并反馈，同步完成条件待追问。更新时间：2026-10-14。
> 源码基线：`verl@c4b389adadc58ce51cb2b63e70df497ca166d77f`，对应本地 `RL_go/verl`，本课沿 `RayPPOTrainer` 路径。
> 学习形式：机制讲解 → 图像 → 固定版本真实代码 → 可运行教学实验 → 课后题。后续回答继续追加本文。

## 学习目标与前置条件

- 区分 Actor 当前参数、Rollout 实际加载的参数与样本来源版本。
- 沿源码找到更新后向生成侧交接权重的调用链。
- 识别“改了版本号”“只同步一部分 replica”“改写历史经验标签”的错误。
- CPU 实验仅需 Python 3.10+；真实 verl/GPU 执行还需适配的训练、推理与通信环境，本课未运行。

## 1. 核心机制：参数更新不会自动改变另一个引擎

Actor 负责更新参数，Rollout 引擎负责采样。即使二者最初加载相同模型，它们也不必引用同一份运行时参数。Actor 从 θ17 更新到 θ18 后，生成侧仍可能驻留 θ17。

![Actor 从 v17 更新到 v18、生成侧等待权重传输和应用完成，以及历史经验保留采样来源版本](assets/rl-weight-version-sync.png)

图 1：同步教学时间线，蓝为 v17，青为 v18。三条参数带仅示意权重集合，不表示实际模型分片数量。样本 τ1 的来源不会因 Actor 更新而改变。新请求使用完成同步的版本；已有请求、partial rollout 与异步部署的处理需按后端另行分析。

这里有三种不同的信息：

| 信息 | 含义 | 不应误认为 |
| --- | --- | --- |
| Actor 版本 | 当前训练参数所对应的修订 | 所有 rollout replica 已加载它 |
| Rollout 版本 | 生成引擎实际采样所用的修订 | 写入一个新编号就完成权重更新 |
| 样本来源版本 | 这次采样经历对应的策略来源 | 训练时可以改写为当前 Actor 的版本 |

版本标签是追溯依据，权重同步是实际传输与应用操作。只修改标签会产生虚假的一致性；还需核验生成后端是否真正应用预期参数。标签相同也不保证 token、精度、采样变换等训推契约一致。

## 2. 真实 verl 代码：先看调用顺序

以下摘自固定版本的 `RayPPOTrainer.fit()`；为阅读省略周围日志与条件，保留关键语句。初始化先恢复训练状态，再将权重交给 Rollout：

```python
self.global_steps = 0
self._load_checkpoint()
self.checkpoint_manager.update_weights(self.global_steps)
```

入口：[ray_trainer.py，第 1401 行附近](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/trainer/ppo/ray_trainer.py#L1401-L1405)。这保证首次生成前发生一次权重交接；不能据此将权重同步与完整恢复等同，后者还涉及 optimizer、scheduler、数据进度等状态。

正常 Actor 更新分支中，顺序为：

```python
actor_output = self._update_actor(batch)
# 中间还有按条件保存 checkpoint 等步骤。
self.checkpoint_manager.update_weights(self.global_steps)
```

入口：[Actor 更新与权重交接](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/trainer/ppo/ray_trainer.py#L1663-L1691)。Critic warmup 是例外分支，可能只唤醒/同步 replicas，不更新 Actor。

`global_steps` 是训练步骤标签，不应无条件解释为 optimizer 更新次数或参数内容哈希：一个外层 step 可以包含多次 mini-batch 更新，也可能因 warmup 不更新 Actor。

## 3. 同步管理器：发起调用不等于全部完成

`CheckpointEngineManager.update_weights()` 根据 backend 选择同步路径。`naive` 分支直接调用 Actor worker group；下面是非 naive 分支的关键调用：

```python
await self.abort_replicas()
await self.release_kv_cache_replicas()
self.build_process_group(rollout)

results = ray.get(
    actor_wg.update_weights(global_steps=global_steps, mode=self.backend)
    + rollout.update_weights(global_steps=global_steps)
)

# 源码在恢复前还执行各 worker 的 checkpoint-engine finalize。
await self.resume_kv_cache_replicas()
await self.resume_generation_replicas()
```

入口：[base.py 的 update_weights](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/checkpoint_engine/base.py#L506-L558)。源码顺序体现请求协调、cache 生命周期、发送/接收、完成等待和恢复。`ray.get` 等待相应任务完成；它本身不证明任意后端在任何故障下都能实现原子切换或内容校验。

Rollout worker 再将接收的权重交给后端 adapter：

```python
weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
await self.server_adapter.update_weights(
    weights,
    global_steps=global_steps,
    wire_format=getattr(self.checkpoint_engine, "wire_format", "named_tensors"),
)
```

入口：[接收与应用权重](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/checkpoint_engine/base.py#L355-L361)。继续往下读应核对具体 adapter 的权重布局、dtype、等待与 cache 处理，不能把这个接口当成只有“改编号”的操作。

本版本 vLLM server 的输出 extra fields 携带 `global_steps`，`llm_server.py` 还处理 `min_global_steps` / `max_global_steps`。这些是追溯线索，不是每条历史轨迹可随意改成当前版本的理由，也不自动保证跨版本继续生成的轨迹只有一个策略来源。[vLLM 输出元数据](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L626)、[多段生成的版本范围](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/workers/rollout/llm_server.py#L435-L460)

## 4. 代码实验：四种状态，哪些允许新一轮同步采样？

[可运行 demo](../code/weight_version_demo/demo.py) 使用不可变 `Policy` 和两份 RolloutEngine 模拟上述契约，不导入 verl，也不训练 LLM。

```powershell
python RL_go/code/weight_version_demo/demo.py
```

教学伪代码可以先记这几行：

```python
history = rollout.generate()          # 来源 v17
actor = actor_update(actor)           # Actor 到 v18
pause_and_drain(rollout)              # 实际实现须协调在途请求
transfer_and_apply(actor, rollout)
wait_for_all_replicas()
verify_expected_version_and_weights()
resume_generation(rollout)
new_batch = rollout.generate()        # 完成同步后，来源 v18
# history 仍属于 v17；不得重贴为 v18。
```

| 实验状态 | 预期 |
| --- | --- |
| Actor 已到 v18，Rollout 仍是 v17 | 拒绝同步一致性检查 |
| 只将 Rollout 标签写为 18，参数未更新 | 拒绝：same label, different weights |
| 两个 replicas 仅更新一个 | 拒绝：其余 replica 参数仍旧 |
| 全部 replicas 应用 v18 | 检查通过，新经验来源为 18，历史经验保持 17 |

实际结果：2026-10-14 本地 Python 3.13 运行通过。真实同步是分布式机制；CPU tuple 的精确比较只验证教学状态，不替代真实 GPU 参数布局、数值与请求并发检查。完整输出与限制见[实验 README](../code/weight_version_demo/README.md)。

## 5. 官方 RL 仓库对照

2026-10-14 检索了以下官方项目；本课不按 star 排名，也不把其他仓库的当前 API 混入固定 verl 快照：

- [verl](https://github.com/verl-project/verl)：本课源码主线，沿 trainer → checkpoint manager → rollout adapter 追踪交接。
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)：可在后续对照其 Ray/vLLM 与同步/异步训练组织。[官方异步训练说明](https://openrlhf.readthedocs.io/en/latest/async_training.html)
- [slime](https://github.com/THUDM/slime)：可在后续对照训练与生成侧的权重同步。[官方权重同步专题](https://github.com/THUDM/slime/blob/main/docs/en/advanced/delta-weight-sync.md)

本课只确认其公开入口并选择 verl 深读，没有运行另外两套框架，也不引入其性能宣称。

## 6. 课后题与反馈

1. Actor 已到 v18，两份 Rollout 分别是 v18 和 v17。若你只检查 trainer 的 `global_steps=18`，漏掉了什么？同步模式下什么时候才允许下一批请求？
2. 一条经验由 v17 生成，训练处理它时 Actor 已到 v20。是否可以将经验来源标签改为 v20？请说明原因。
3. 把伪代码里的 `wait_for_all_replicas()` 移到 `resume_generation()` 之后，可能出现什么问题？这与“给所有 replicas 发出了更新请求”有什么区别？

### 交互记录

- 2026-10-14：用户要求每课包含讲解、图像、真实源码/伪代码和课后题，并实时记录 RL 博客。
- 本课提供权重版本与同步的机制图、固定 verl 源码及 CPU 实验。
- 2026-10-14：三道题已作答，反馈如下；不将反馈给出等同于所有补充点已掌握。

### 第 1 题：漏掉了生成侧的实际权重状态

用户回答：“漏掉了检查 rollout 是否也是最新的 v18 的权重，保证训推权重一致。”

判断：核心正确。同步模式下，检查范围应覆盖参与新一轮采样的所有 rollout replicas/workers，而不只 trainer 或其中一份生成实例；应确认预期权重传输、应用及必要的资源状态恢复完成后，再放行新请求。这里只指本课同步契约；异步部署可能允许受控版本差，需要另外定义策略。

### 第 2 题：新权重生成新经验，历史来源不改写

用户回答：“我们使用的 response 来源就是 rollout v17，如果要改为 v20，就需要同步，然后生成 response。”

判断：方向正确，需要明确“生成的是新经验”。完成 v20 同步后重新生成，可得到来源 v20 的新样本；这不会让原 v17 回答变成 v20 样本。即使新旧回答文本恰好相同，采样事件、来源与对应 log-prob 仍需各自记录。

也不意味着旧经验一律丢弃或必须重采：是否继续使用，取决于训练方案、版本滞后容忍及采样修正。关键是保留真实来源，不能通过重贴标签消除 off-policy 差异。

### 第 3 题：未等待完成的风险不止旧版本

用户回答：“还是使用旧权重生成的样本和经验。”

判断：这是可能后果，但还不完整。若恢复请求时同步仍在进行，可能出现某些 replicas 已是新版本、另一些仍是旧版本；若后端缺少必要的切换隔离，还可能访问未完成应用的参数或发生请求/缓存状态冲突。具体表现取决于后端，不能断言每次都会使用旧权重或每个后端都允许半更新访问。

“所有更新请求已发出”只说明操作启动；“所有相关 worker 已完成应用并可服务”才符合本课放行条件。请求暂停、完成等待和恢复顺序正是为了协调这两个状态。

### 追问：怎样定义同步完成？

假设三个 worker 的状态分别是：发送完成、接收完成、参数应用完成。下一批生成能否只依据“发送端完成”放行？请用自己的话说明需要等待什么。

状态：追问尚未作答；下一课数据/mask 正确性尚未开始。

## 下一课

数据与 mask 正确性：从 prompt、assistant token、工具 observation 和 padding 的位置出发，理解哪些 token 提供上下文，哪些参与策略 loss。
