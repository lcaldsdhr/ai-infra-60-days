# SPIN：用旧策略生成「负例」的自博弈微调

> **本文采用的 SPIN 定义**：Chen 等人的 *Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models*（ICML 2024，[arXiv:2401.01335](https://arxiv.org/abs/2401.01335)；[官方实现](https://github.com/uclaml/SPIN)）。这不是 SPPO，也不是面向扩散模型的同名 SPIN-Diffusion。

## 1. 要解决什么问题

SFT 模型已经见过标注示范，再对同一批数据直接 SFT 可能没有收益甚至退化；而 RLHF / DPO 通常还需要额外的偏好标注或更强模型的评判。SPIN 的前提是：已有一批高质量 `(prompt, human response)` 与一个 SFT 初始化模型；它把**上一轮模型对相同 prompt 的生成**当作合成对手回答，用“人工回答优于旧策略回答”的比较来继续训练，而不新增人工或强教师反馈。[论文 §3–4](https://arxiv.org/abs/2401.01335)

因此它的目标不是做环境交互式 RL，而是让当前策略逐轮逼近已有 SFT 数据的条件分布。论文在其函数类与损失假设下证明：当策略分布与目标数据分布一致时达到全局最优；该结论不是对有限模型、有限采样训练必然收敛的保证。[论文 Theorem 5.2](https://arxiv.org/abs/2401.01335)

## 2. 核心算法

记第 `t` 轮冻结的对手为 `π_t`。对每个 SFT 样本 `(x, y⁺)`：

1. 用 `π_t` 对 `x` 采样 `y⁻ ~ π_t(·|x)`；
2. 将 `(x, y⁺, y⁻)` 组成偏好对：`y⁺` 是数据中的真实回答，`y⁻` 是旧策略生成；
3. 训练新策略 `π_{t+1}`，使真实回答相对于合成回答的、以旧策略为基准的对数似然比增大；
4. 冻结 `π_{t+1}`，回到第 1 步，得到下一轮对手。

论文给出的 SPIN 目标与 DPO 形式相近：对 `log πθ(y⁺|x) - log πθ(y⁻|x)` 和对应的旧策略差作 logistic 偏好优化；`β` 控制该差值的尺度。关键区别是：DPO 的 rejected response 来自外部偏好数据，而 SPIN 的 rejected response 是上一轮自身生成。[论文 Eq. (4.7) 及 §4.1](https://arxiv.org/abs/2401.01335)

```text
SFT 数据 (x, y+) ───────────────────────────┐
                                            ├─> 偏好对 (x, y+, y-) ─> SPIN 更新 πt+1
冻结旧策略 πt ── generate(x) ──> y- ─────────┘                              │
                 （下一轮以 πt+1 作为对手） ◄────────────────────────────────┘
```

## 3. 一轮训练的工程流程

官方实现的流程是“先整轮生成、再整轮训练”，不是在每个 minibatch 内在线生成：

1. 准备 SFT 格式数据；生成脚本只用 prompt，输出 `real` 与 `generated` 两个对话字段。
2. 用上一轮 checkpoint 批量生成 `y⁻`；官方建议普通生成按较小分片（示例 `frac_len=800`）降低长生成崩溃风险，并提供 vLLM 生成脚本。[官方 README：generation](https://github.com/uclaml/SPIN#-faster-generation-with-vllm)
3. 聚合 JSON、转换为 `train_prefs-*.parquet` / `test_prefs-*.parquet`。
4. 运行 `spin/run_spin.py` 训练一个新 checkpoint；下一轮用它重新生成对手回答。[官方 README：convert 与 fine-tuning](https://github.com/uclaml/SPIN#step-15-gather-generations-and-convert-data-type)

## 4. 关键超参数与机制

以下是**论文官方仓库的复现实验默认值**，不是跨任务通用配方：

| 项目 | 官方默认 / 机制 | 为什么要记录 |
| --- | --- | --- |
| `beta` | `0.1` | 决定偏好 logit 的尺度；过大可能使更新过激，过小则信号弱。[配置说明](https://github.com/uclaml/SPIN#step-2-fine-tuning) |
| 每卡 batch | `16` | 真正 global batch 是 `per_device_train_batch_size × processes × gradient_accumulation_steps`。[配置说明](https://github.com/uclaml/SPIN#step-2-fine-tuning) |
| 每轮 epoch | `3` | 与数据量、学习率、模型大小联动，不能孤立搬用。[配置说明](https://github.com/uclaml/SPIN#step-2-fine-tuning) |
| 迭代数据混合 | 官方 iter 1/2 同时混入当前及前一轮数据 | 这是实现中的稳定性选择，应在实验日志记录 mix 与权重。[配置说明](https://github.com/uclaml/SPIN#step-2-fine-tuning) |
| 对手版本 | 由**用于 generation 的 checkpoint**决定 | 不要把“训练步数”误当作 SPIN iteration；官方明确以生成模型决定迭代编号。[数据约定](https://github.com/uclaml/SPIN#step-1-generation) |

另外，官方 README 标明其早期上传的数据集曾错误、后已更换；复现时应锁定数据 revision、SFT checkpoint revision 和生成参数，而不是只记录论文名。[官方更正公告](https://github.com/uclaml/SPIN#-news)

## 5. 优点、局限与常见失败模式

**优点**

- 只依赖既有 SFT 示范与自身生成，减少额外人工偏好或强模型裁判成本。[论文摘要](https://arxiv.org/abs/2401.01335)
- 把“模型答案与示范答案的质量差”变成可训练的 pairwise 信号；官方代码把生成、格式转换、训练三步拆开，便于审计每轮数据。[官方实现](https://github.com/uclaml/SPIN)

**局限 / 失败模式**

- 若 SFT 示范质量、覆盖或格式有问题，SPIN 的目标仍是逼近它；它不会凭空得到任务外能力。这是由其目标分布定义直接推得的限制。[论文 §3–5](https://arxiv.org/abs/2401.01335)
- 对手回答太弱时，偏好对很容易；太接近当前策略时，比较信号变弱。迭代生成还会放大采样、模板、tokenizer 或数据去重错误。
- 生成成本和存储会随轮数增长；务必保存 `prompt_id`、`generator_commit/checkpoint`、采样参数及每轮胜率/长度分布。
- 不应把论文的理论最优性误读为实际训练单调提升；应逐轮在固定评测集和长度控制指标上验收。

## 6. 与本仓库 verl 学习路径的关系

SPIN 可视为理解“**离线偏好式自改进**”的支线：它不需要 reward model、critic 或逐步环境 reward；而本仓库的 [GRPO](grpo.md) / [DAPO](dapo.md) 主线以 rollout reward 和策略梯度为核心。先用 SPIN 学清 `chosen/rejected`、reference / opponent checkpoint、数据版本，能帮助辨析为何 agentic RL 又额外需要轨迹、环境状态和 credit assignment。

本固定版本的 `verl` 主要提供 PPO/GRPO 等 RL 训练基础设施；将 SPIN 接入时，应把上游生成/转换产物作为可版本化的偏好数据，并在外层训练脚本实现其迭代编排，而不要假设它等同于 GRPO。阅读入口：

- [SPIN 论文](https://arxiv.org/abs/2401.01335) 的 §4（目标）与 Algorithm 1；
- [SPIN 官方 `spin/generate.py`](https://github.com/uclaml/SPIN/tree/main/spin) 与 [`spin/run_spin.py`](https://github.com/uclaml/SPIN/tree/main/spin)；
- 本地 verl 的 [GRPO 算法说明](../verl/docs/algo/grpo.md)（用于对照）和 [PPO trainer 入口](../verl/verl/trainer/main_ppo.py)。

## 7. 最小实验验收清单

记录 `base/SFT checkpoint`、每轮 generator checkpoint、数据哈希、chat template、`temperature/top_p/max_new_tokens`、`β`、global batch、epoch；抽查同一 prompt 的 `y⁺/y⁻`；并在固定 held-out 集上同时报告质量、长度和拒答率。若下一轮的合成回答为空、模板错位、与 `y⁺` 重复或长度分布突变，应先停止训练排查数据链路。
