# verl 训练生命周期 Q&A：样本对齐、训练循环与策略概率

> 记录日期：2026-10-13。来源：本轮交互式源码复习；保留经讨论校正后的结论，不将初始误解当作最终答案。
> 源码基线：`verl@c4b389adadc58ce51cb2b63e70df497ca166d77f`，本地路径 `RL_go/verl`。
> 对应文章：[Verl 训练生命周期](verl-training-lifecycle.md)。这是该文章的 `RayPPOTrainer` 路径，不与另一快照的 V1 教程混用。
> 验证状态：已静态核对下列源码入口；未运行 GPU 训练。数字例子为教学示例，不是真实实验结果。

## Q1：DataProto 的第 0 维存什么？

第 0 维是样本排列的方向，不存储 prompt 身份。采样前的 batch 通常一行一条 prompt；生成后的训练 batch 通常一行对应一条 prompt 的一次 rollout。Agentic RL 中，一次 rollout 可以包含多轮生成、工具调用与环境反馈。

设两个 prompt 各采样两次：

| 第 0 维索引 | uid | 回答 | 回答级 reward |
| --- | --- | --- | --- |
| 0 | A | A1 | 1 |
| 1 | A | A2 | 0 |
| 2 | B | B1 | 0 |
| 3 | B | B2 | 1 |

`responses[2]` 取出 B1 这一次 rollout；查找 `uid=B` 对应索引 `[2, 3]`，才得到问题 B 的多次 rollout。`uid` 标识 prompt 分组，不自动唯一标识组内每条回答。

## Q2：为什么各字段第 0 维必须一致？

同一索引必须指向同一次 rollout。回答、mask、log-prob、reward 与 advantage 如果分别来自不同样本，loss 可以正常计算，但训练信号已经错误。

例如 `old_log_probs=[4, 2]` 表示四条回答，每条两个 token 位置。`ref_log_prob` 和 token 级 advantage 在这个示例中同为 `[4, 2]`；回答级 reward 可为 `[4]`，prompt `input_ids` 可为 `[4, P]`。第 0 维相同不要求完整 shape 相同，参与同一 token 运算的字段还需对齐 token 位置。批次级 `meta_info` 不按样本索引重排。

`from_dict` / `check_consistency` 检查结构与长度，无法识别回答 A1 是否误配了 B2 的优势。[构造与一致性检查](../verl/verl/protocol.py)

## Q3：repeat、select、select_idxs、union、reorder 分别做什么？

| 操作 | 对象 | 关键行为 |
| --- | --- | --- |
| `repeat(2, interleave=True)` | 样本行 | `[A,B] → [A,A,B,B]`，tensor 与非 tensor 样本字段共同复制 |
| `select(...)` | 字段 | 保留所需 key，样本数量通常不变；默认不保证独立深拷贝 |
| `select_idxs([2,3])` | 样本行 | 所有样本字段按共同索引取出子批次 |
| `union(other)` | 字段 | 按现有行位置合并、检查冲突；原地修改并返回 `self` |
| `reorder(indices)` | 样本行 | 按共同索引重排 tensor 与非 tensor 字段；原地修改，不返回新 batch |

所以 `batch.reorder(indices)` 可以，`batch = batch.reorder(indices)` 会把变量赋为 `None`。本版本 `reorder` 的非 tensor 索引转换使用 `indices.detach().numpy()`，教学调用应使用 CPU 上的索引 tensor。[操作实现](../verl/verl/protocol.py)

## Q4：union 检查双方都是四行，为什么仍不能保证配对正确？

讨论中的判断是：“行数相同，但 prompt 和 response 除了 IDs 外没有绑定的属性。”更精确的结论是：配对可由调用方保证的返回顺序或 rollout 唯一标识建立；`union` 本身不会按 ID 自动关联，也不会验证语义身份。

假设输入任务与输出顺序分别是：

```text
输入：A1  A2  B1  B2
输出：B1  A1  B2  A2
```

直接按行合并就会错配。先合并再整体重排，只是移动已经错误的组合。若只有输出被打乱，应先依据每次 rollout 的标识或位置映射恢复输出，再 `union`。这个例子的恢复索引为 `[1, 3, 0, 2]`；问题级 `uid` 本身不足以区分 A1/A2。

只有输入输出已经逐行对应，而你希望改变整个批次的共同顺序时，合并后再整体重排才成立。博客教学伪代码已按这个前提修订；这不表示固定版本真实 rollout manager 必然乱序返回。

## Q5：训练编排器是 rollout 后的多次 epoch 内循环吗？

这里的编排器是 `RayPPOTrainer`，它覆盖完整外层 RL 流程。Actor 使用本轮经验做多次 PPO epoch，只是编排器调用的一个阶段。

```text
prompt 数据集 epoch
  └─ 多个外层 RL step
      ├─ 取 prompt，生成 rollout
      ├─ reward、log-prob、可选 value、advantage
      ├─ 可选 critic 更新
      ├─ actor 更新
      │   └─ 本轮固定经验上的 PPO epochs
      │       └─ mini-batch loss、反向与参数更新
      └─ 同步新权重给 rollout，进入下一步
```

数据 epoch 遍历 prompt 数据集，期间产生多批新经验；PPO epoch 重复使用某一轮生成的经验，通常不重新采样。Actor worker 管理实际 mini-batch/epoch 执行，engine 承担计算与更新。[外层编排](../verl/verl/trainer/ppo/ray_trainer.py)、[worker 执行](../verl/verl/workers/engine_workers.py)

## Q6：为什么内层多个 PPO epochs 要固定 old_log_probs？

`old_log_probs` 是本轮更新的旧策略锚点，当前 Actor 概率随更新变化：

\[
\rho_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right).
\]

固定分母，可以衡量当前策略相对本轮旧策略的累计变化。PPO clip 影响目标函数和梯度激励，不是强制 ratio 永远不越界的硬约束。

“必须计算 old”应修正为：必须取得与所选训练方案一致的固定旧策略概率，不一定要重算。在这个版本中，解耦路径调用 `_compute_old_log_prob`；bypass 路径把 `rollout_log_probs` 赋给 `old_log_probs`。[路径选择](../verl/verl/trainer/ppo/ray_trainer.py)、[apply_bypass_mode](../verl/verl/trainer/ppo/rollout_corr_helper.py)

## Q7：每次更新后把 old 概率刷新成上一步，会怎样？

会失去原先固定的锚点，改变 PPO 更新目标。质量可能退化，但直接问题是累计偏移被滚动分母掩盖。

| 阶段 | 当前 token 概率 | 固定 old=0.10 的 ratio | 分母换为上一步概率 |
| --- | ---: | ---: | ---: |
| 更新 1 | 0.12 | 1.20 | 1.20 |
| 更新 2 | 0.144 | 1.44 | 1.20 |
| 更新 3 | 0.1728 | 1.728 | 1.20 |

每次局部变化都为 20%，累计却已经为 72.8%。因此，同一轮 PPO epochs 内旧概率固定；新的外层 batch/采样阶段再建立新的锚点。

## Q8：old、reference 与 current policy 有何区别？

| 策略 | 时间尺度与职责 |
| --- | --- |
| current Actor | 内层更新中参数改变，用于当前 log-prob 与梯度 |
| old policy | 本轮更新开始前固定的锚点，内层不刷新，下一轮重新建立 |
| reference policy | 通常是冻结 SFT/初始策略，跨轮提供长期 KL 参照；具体配方可有不同更新规则 |

讨论中的初始理解是“ref 约束会导致学了半天还在原始分布附近”。校正：reference 的 KL 正则可以与 old 锚点同时使用，允许偏离但增加代价；KL 系数过强才可能压制奖励驱动的学习。reference 也不必是预训练模型。

## Q9：为什么 reference 不能直接替代 PPO ratio 的 old 分母？

两者测量的是不同变化。设一个 token 的概率：

```text
reference：0.02
old：      0.10
current：  0.10（本轮尚未更新）
```

current/old 为 1，表示尚未变化；current/reference 为 5，却把历史多轮积累的变化当作本轮偏移。若优势为正且 PPO ratio clip 上界为 1.2，换成 reference 后目标已经在裁剪平台，可能失去继续提高该 token 概率的直接梯度贡献。整体参数仍可能受其他 token 或其他 loss 项影响。

还有采样分布的原因。在采样来自 old、current 的支持集被 old 覆盖等条件下，**未裁剪**的重要性加权有：

\[
\mathbb E_{a\sim\pi_{\mathrm{old}}}
\left[\frac{\pi_{\mathrm{current}}(a)}{\pi_{\mathrm{old}}(a)}f(a)\right]
=\mathbb E_{a\sim\pi_{\mathrm{current}}}[f(a)].
\]

数据仍来自 old，却将分母改为 ref，这个关系不再成立。这个等式用于说明分母与数据分布的关系，不表示 clipped PPO 目标仍是无偏的重要性估计。

old 与 ref 恰好相同时可以数值等价；一般不能直接替换。若实际数据来自不同 rollout 分布，还需另外分析采样修正，见 Q12。

## Q10：同一份权重，rollout_log_probs 与重算的 old_log_probs 为什么仍可能不同？

相同权重不保证输入契约、计算路径和概率定义相同。

1. **输入对齐**：实际 token IDs、前缀、position IDs、mask、模板与截断等可能不一致。
2. **计算差异**：不同精度、量化、attention/fused kernel 与并行规约路径可能产生数值差异。
3. **分布定义**：temperature、top-k/top-p 等采样变换可能改变用于采样的分布；后端返回 log-prob 的口径也需要核对。

例如模型原始概率为 `A=0.6, B=0.3, C=0.1`。top-2 丢弃 C 后重新归一化，采样概率为 `A=2/3, B=1/3`。若采到 A，原始模型概率是 `log(0.6)`，处理后的采样概率是 `log(2/3)`。

这是定义差异的教学例子，**不宣称当前 vLLM/SGLang 一定返回 top-k 处理后的概率**。是否重算 temperature、是否返回过滤前概率，必须核对固定后端实现与调用配置；本轮没有进一步核验各后端的具体口径。

## Q11：只保证 prompt 相同，是否足够比较概率？

不够。应固定同一条回答、同一 token 位置与相同前缀。回答为 `[A,B,C]` 时，比较 B 对应的是：

\[
\log\pi(B\mid\mathrm{prompt},A).
\]

训练侧通常对已有回答做 teacher-forced 前向，逐位置计算实际回答 token 的条件概率，不需要重新生成回答。应先对齐输入与 token 位置，再对齐概率口径，最后分析计算误差。

## Q12：rollout 与 old 不同，这批样本还能直接当作来自 old 的 on-policy 数据吗？

本轮初答是“算”。校正后的结论：一般不能。这里将 on-policy 的参照明确为本轮 PPO 的 old 策略；只有实际 rollout 分布与 old 分布一致时，才能严格视作来自 old 的数据。这不意味着 PPO 内层每次更新后都重新采样；内层复用旧经验正是其更新方案的一部分。

经验的来源由采样过程决定，重算 log-prob 不会改变来源。教学例子：

```text
rollout：P(A)=0.8，P(B)=0.2
old：    P(A)=0.5，P(B)=0.5

从 rollout 抽 100 次，平均约有 80 个 A、20 个 B。
训练侧重算 old：A 与 B 都得到 log(0.5)。
原有样本的频率没有变成从 old 抽样时预期的 50:50。
```

因此，“重算旧概率，不会自动让经验变成 on-policy”的意思是：**重评分不等于改变采样分布**。它不是说 old 与 rollout 没有关系；old 正是在对 rollout 产生的相同回答计算概率。

需区分两个比率：

\[
\underbrace{\frac{\pi_{\mathrm{old}}}{\pi_{\mathrm{rollout}}}}_{\text{数据来源修正}}
\qquad
\underbrace{\frac{\pi_{\mathrm{current}}}{\pi_{\mathrm{old}}}}_{\text{PPO 更新比率}}.
\]

在这个单步、支持集满足条件的教学例子中，A 的来源修正权重是 `0.5/0.8=0.625`，B 是 `0.5/0.2=2.5`：降低过度采样的 A 的贡献，提高欠采样的 B 的贡献。这是未裁剪重要性加权的直觉；真实序列/token 级修正还要核对条件概率、归约粒度、截断、拒绝采样和方差。

如果 top-k 等操作把某些目标动作的采样概率截为零，目标分布仍给它正概率，单靠已有样本的权重不能恢复这些从未可采的动作。若差异仅为很小的数值误差，工程上可能近似按 on-policy 处理，但需要指标支持，不能只因权重相同就认定一致。

## Q13：rollout、old、current 具体指什么？谁最旧？

在当前讨论的解耦路径中，可记为：

| 策略 | 角色与时点 | 对应数据 |
| --- | --- | --- |
| rollout policy | 生成侧实际采样回答时的策略 | `rollout_log_probs`，具体返回口径需核对后端 |
| old policy | 训练侧开始本轮更新前固定的锚点 | `old_log_probs`，本轮更新期间不刷新 |
| current policy | 训练侧此刻正在优化的 Actor | 当前前向产生的 log-prob，随参数更新变化 |

“old”不是年龄排名。同步且训推一致时，rollout 与 old 可以相同；异步时，生成侧 rollout 权重可能比训练侧 old 更旧。内层 PPO 循环中变化的是 current，而不是把 rollout 当作更新中的分布。

```text
生成侧使用 v17 → 采样回答，记录 rollout 概率
训练侧尚未更新 → 对这条回答重算 old 概率，并固定这些值
Actor 多次更新 → current 概率不断变化
交接新权重     → 下一轮 rollout
```

old 不一定是一份单独驻留的模型：可以只在更新前计算并保存本轮旧概率。bypass 则直接将 rollout 的记录用作旧概率，角色仍应区分，不再额外重算训练侧锚点。

## Q14：GRPO 为什么通常能去掉 Critic，却不一定能去掉 Reference？

讨论中的回答是：GRPO 采用组内相对优势，无需 Critic；Reference 用于约束分布漂移。更精确地说，GRPO 替代的是 Critic 在优势估计中的职责，没有替代 Reference 的 KL 参照职责。

- Critic 学习状态价值，支持 GAE 等优势估计。GRPO 从同题多个 rollout 的奖励构造比较基准，通常无需学习价值模型。
- Reference 通常冻结，提供 KL 参照，为偏离增加代价；这是软正则，不是保证分布永不越界。

若配方启用 reference-KL，通常还需要 Reference；相关 KL 项均关闭时，它可能也不创建。因此 Reference 不是 GRPO 的必选组件，要看具体算法配方和条件接线。

此前提出“冻结 Reference 为什么仍有成本”的问题尚未互动展开。它保留为后续资源账本话题，不阻挡当前转向系统特性的主线。

## Q15：这里要学习的“RL 特性”具体是什么？

用户明确希望关注异步训推等热门、好用和基础的**训练系统能力**，不再只反复讨论算法名词。下面是系统能力地图；支持形态与配置需按实际源码版本核对，不能由算法公式推断当前后端全支持。

| 能力 | 解决的问题 | 主要代价或正确性要求 |
| --- | --- | --- |
| 异步训推 | rollout 与 actor 更新重叠，减少互相等待 | 策略滞后、队列积压、经验来源修正 |
| 异步多轮 rollout / Agent Loop | 并发处理生成、工具调用与环境交互 | 长短轨迹、工具失败、状态隔离和动作 mask |
| 共置、sleep/wake、offload | 有限显存下复用训练与生成资源 | 搬运、切换和峰值；是否可重叠取决于部署 |
| 权重同步与版本追踪 | 生成侧使用预期策略，经验可追溯 | 传输成本、缓存生命周期、版本错配 |
| rollout correction | 处理实际生成策略与训练锚点的差异 | 支持集、概率定义、方差与截断偏差 |
| 动态 batching、packing、长度均衡 | 提高变长请求与训练样本的执行效率 | 样本/attention/mask 对齐与 loss 归约 |
| 完整 checkpoint / 恢复 | 长训练失败后续跑 | optimizer、scheduler、随机状态与数据进度 |

优先掌握权重版本、样本/mask 正确性与恢复能力。异步、共置、packing 等优化按硬件和瓶颈选择，并非全部开启就更好。

## Q16：异步 rollout 与异步训推是一回事吗？

不是。

**异步 rollout**：生成侧并发处理请求，一个 Agent 等工具时，其他 Agent 继续生成。训练侧仍可能等整批经验完成后才更新，因此它不自动意味着训练与生成重叠。

**异步训推**：Actor 消费完成的经验进行更新时，rollout 仍在生成后续经验。下面只表达概念，不代表固定快照所有部署模式的真实调度：

```text
同步：生成一批 → 更新这批 → 同步权重 → 再生成

异步训推：
Rollout：生成 ── 生成 ── 生成 ──
           ↓       ↓       ↓
队列：  已完成经验，携带采样策略版本
           ↓       ↓
Actor： 更新 ── 更新 ── 更新 ──
           └── 按同步策略交接新权重
```

Agent 轨迹和工具延迟差异大时，重叠执行可能减少对慢轨迹的等待；收益还取决于资源划分、生成/训练吞吐和调度策略。Actor 已到 v20，队列仍有 v17 经验时，就需要分析 staleness，不能只看设备利用率。

衡量异步方案至少看四项：设备空闲是否减少、队列/背压是否可控、样本年龄与策略版本差是否合理、单位时间有效学习信号是否增加。生成 tokens/s 提升不等同模型质量改善。

## 下一次学习入口：异步执行、队列与策略滞后

接下来按以下顺序展开，不将概念图冒充已验证的后端能力：

1. 同步/异步时间线：哪些阶段重叠，哪些必须等待。
2. 经验队列与背压：生成过快积压、训练过快断粮如何处理。
3. 权重交接：更新时机、策略版本、缓存与资源状态。
4. staleness：样本年龄、版本差与学习稳定性的关系。
5. rollout correction：不同概率分布、修正粒度与偏差/方差。

这条线始终保持三种策略的区别：

```text
实际数据来源：rollout policy
更新固定锚点：old policy
正在优化对象：current policy
```

相关源码入口是 `compute_rollout_correction_and_add_to_batch` 与 `apply_bypass_mode`。[rollout correction helper](../verl/verl/trainer/ppo/rollout_corr_helper.py)

本轮已完成 on-policy 概念澄清；异步系统的具体配置与实现留待下一次源码核验，不记录为已实验验证或已掌握。
