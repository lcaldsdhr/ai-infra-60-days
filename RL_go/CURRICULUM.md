# AI 推理、强化学习与 verl 完整学习地图

这份课程按“概念 → 图解 → 最小代码 → 框架源码 → 系统实验”组织。目标不是背算法名，而是能回答三个问题：数据怎样流动、瓶颈/梯度从哪里来、怎样用指标和小实验验证判断。

## 路线 A：LLM 推理系统

| 阶段 | 主题 | 主要材料 | 实验/验收 |
| --- | --- | --- | --- |
| A1 | 请求生命周期、Prefill/Decode、指标 | [推理系统全景](blog/inference/serving-system-guide.md) | 能分解 queue、TTFT、TPOT 与 E2E |
| A2 | KV Cache 原理与容量 | [PD/KV 专题](blog/inference/pd-disaggregation.md) | [KV 容量计算器](code/inference_kv_capacity/README.md) |
| A3 | Continuous batching、Paged KV、Prefix Cache | [Batch/Cache/加速](blog/inference/batching-cache-and-acceleration.md) | 画出 static/continuous 时间线与 block table |
| A4 | 量化、投机解码 | [Batch/Cache/加速](blog/inference/batching-cache-and-acceleration.md) | 同时报性能、质量、显存与接受率 |
| A5 | TP/PP/EP、PD 分离 | [PD 分离](blog/inference/pd-disaggregation.md) | 记录拓扑、KV 传输与 P/D 队列 |
| A6 | 容量、压测、SLO | [容量与可观测性](blog/inference/observability-and-capacity.md) | 获得并发—goodput—P99 安全工作区 |

## 路线 B：经典强化学习

| 阶段 | 主题 | 主要材料 | 实验/验收 |
| --- | --- | --- | --- |
| B1 | MDP、return、Bellman、Value/Q | [从 MDP 到 PPO](blog/reinforcement-learning/foundations-to-ppo.md) | 手算短轨迹 return 与 Value |
| B2 | Policy Gradient、REINFORCE、baseline | [从 MDP 到 PPO](blog/reinforcement-learning/foundations-to-ppo.md) | 能解释 baseline 为何降方差不改期望 |
| B3 | Actor-Critic、GAE | [从 MDP 到 PPO](blog/reinforcement-learning/foundations-to-ppo.md) | 手算 δ、GAE、terminal mask |
| B4 | PPO ratio、Clip、KL | [PPO Clip 实验](code/ppo_clip_demo/README.md) | 正负优势四种 ratio 分支与实现一致 |

## 路线 C：LLM RL 与 Agentic RL

| 阶段 | 主题 | 主要材料 | 实验/验收 |
| --- | --- | --- | --- |
| C1 | RLHF、reward、token mask | [LLM RL 训练栈](blog/reinforcement-learning/llm-rl-training-stack.md) | reward 有版本、分解和异常测试 |
| C2 | GRPO | [GRPO 专题](docs/grpo.md) | [组内优势实验](code/grpo_advantage_demo/README.md) |
| C3 | DAPO | [DAPO 专题](docs/dapo.md) | 逐项消融 Clip-Higher、过滤、token loss、长度 shaping |
| C4 | SPIN | [SPIN 专题](docs/spin.md) | 固定每轮 generator checkpoint 与偏好数据版本 |
| C5 | Agentic RL | [Agentic RL 专题](docs/agentic-rl.md) | 轨迹可重放、工具可隔离、reward 可重算 |
| C6 | 训推一致性 | [训推一致性](blog/reinforcement-learning/training-inference-consistency.md) | 对齐 token IDs、log-prob、policy version、ratio/KL |

## 路线 D：verl 源码与分布式系统

固定版本：`verl@c4b389adadc58ce51cb2b63e70df497ca166d77f`。

1. [Verl 训练生命周期](blog/verl-training-lifecycle.md)：DataProto、初始化、rollout、reward/advantage、训练与权重同步。
2. `verl/trainer/main_ppo.py`：Hydra 配置、数据与 trainer 入口。
3. `verl/trainer/ppo/ray_trainer.py`：完整分布式编排。
4. `verl/trainer/ppo/core_algos.py`：GAE、GRPO、policy/value loss 与 KL。
5. `verl/workers/rollout/`：vLLM/SGLang rollout、采样和权重同步。
6. `verl/experimental/agent_loop/`：多轮工具与环境轨迹。

## 路线 E：训练效率、稳定性与恢复

1. [显存与吞吐](blog/training/memory-and-throughput.md)：梯度累积、混合精度、激活重计算、状态分片与 Sequence Packing。
2. [训练稳定性与故障恢复](blog/training/stability-and-recovery.md)：梯度裁剪、LR Warmup 与完整 Checkpoint。
3. 验收时同时记录显存、有效 tokens/s、通信占比、grad norm、NaN/Inf、LR、验证指标和恢复演练结果。

## 每章统一验收模板

- **能讲清**：用自己的话解释输入、输出、状态和瓶颈；
- **能手算**：至少一个小张量/短轨迹例子；
- **能运行**：命令、预期输出、版本和已知限制齐全；
- **能定位源码**：论文符号 → 配置键 → 函数 → 指标；
- **能证伪**：写出失败模式和出现什么日志时应停止扩大实验。

权威阅读入口统一收录在[一手资料索引](docs/research/learning-curriculum-primary-sources.md)。
