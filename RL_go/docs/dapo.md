# DAPO：面向长 CoT 的 GRPO 训练配方

> 学习目标：把 DAPO 看成一组可单独验证的工程/算法改动，而不是一个可脱离数据、reward、系统和评测直接复刻的“万能 RL 算法”。

## 1. 问题背景与目标

**Decoupled Clip and Dynamic Sampling Policy Optimization（DAPO）** 是 ByteDance Seed 与清华 AIR 提出的开源 LLM RL 系统/训练方法。论文从 Qwen2.5-32B 的朴素 GRPO 起步，报告其出现 entropy collapse、reward noise 与训练不稳定，并提出四项改动来训练长 Chain-of-Thought 推理模型。[DAPO 论文 §1](https://arxiv.org/html/2503.14476v1#S1)

论文报告在 Qwen2.5-32B Base 上 AIME 2024 达到 50 分；这是**该论文的特定模型、数据、评测和训练预算下的结果**，不是对其它模型的保证。[论文摘要](https://arxiv.org/abs/2503.14476) DAPO 的官方仓库同时公开了数据、代码与训练记录，并明确其实现构建在 verl 之上。[官方仓库的可复现说明](https://github.com/BytedTsinghua-SIA/DAPO#reproducibility)

## 2. 核心目标与算法

DAPO 仍使用 GRPO 风格的组内相对优势：对每题的 \(G\) 个回答按 reward 标准化，并以 token 的新旧策略概率比做 clipped policy-gradient。区别是它将单一 clip 区间拆为 \(\varepsilon_{low}\)、\(\varepsilon_{high}\)，并在目标中按**所有有效 token**聚合 loss。完整目标与约束见[论文 Eq. (10)–(11)](https://arxiv.org/html/2503.14476v1#S3)。

四个组成部分及对应问题如下：

| 机制 | 解决的问题 | 做法 |
| --- | --- | --- |
| Clip-Higher | 朴素上界限制低概率 token 被强化，探索/熵易下降 | 令 \(\varepsilon_{high}>\varepsilon_{low}\)，给正向概率增量更大空间。 |
| Dynamic Sampling | 一题所有回答全对或全错时，组优势为零 | 过采样并丢弃 accuracy 0 或 1 的组，直到填满训练 batch。 |
| Token-level PG loss | sequence-level 聚合会使长回答中不良 token 的惩罚变弱 | 按有效 token 汇总 loss，而不是先对每个序列平均。 |
| Overlong Reward Shaping | 截断的超长回答带来 noisy reward | 在最大长度前留惩罚缓冲区，长度越超界罚得越多。 |

表中结论均来自论文对四项技术的定义和消融动机。[DAPO §3](https://arxiv.org/html/2503.14476v1#S3)

## 3. 训练流程

1. 从题目/答案对取 prompt，以当前 rollout policy 每题生成 \(G\) 条候选。
2. 用可验证 reward 判定每条候选；动态采样缓存只保留组内既非全对也非全错的 prompt，凑满训练 batch。
3. 在保留样本上计算组相对优势；对超长但未硬截断的回答加入长度相关负奖励。
4. actor 以解耦 clip 的 token-level loss 更新若干轮；记录 reward、长度、entropy、生成概率和验证准确率。
5. 同步 actor 到 rollout，引入新策略继续采样。

论文伪代码明确包含“生成—过滤—缓冲至批量—计算优势—更新”的顺序；过滤改变了训练分布，因此日志必须同时记录生成量、丢弃量和有效组比例。[Algorithm 1](https://arxiv.org/html/2503.14476v1#S3)

## 4. 关键配置与工程落点

以下是 verl 中与论文概念的直接对应；数值仅作为文档/配方的示例，不是建议默认值。

| DAPO 概念 | verl 配置/代码入口 | 含义 |
| --- | --- | --- |
| 组相对优势 | `algorithm.adv_estimator=grpo` | DAPO 沿用 GRPO advantage estimator。[官方 best practices](https://verl.readthedocs.io/en/latest/perf/best_practices.html) |
| 解耦 clip | `actor.clip_ratio_low`、`actor.clip_ratio_high` | 例：`0.2/0.28`；实际代码对 ratio 分别下/上截断。[固定版 DAPO 文档](../verl/docs/algo/dapo.md) |
| 动态采样 | `algorithm.filter_groups` | `enable`、评分 `metric`、`max_num_gen_batches` 控制过滤和重采样上限。[固定版 DAPO 文档](../verl/docs/algo/dapo.md) |
| token-level loss | `actor.loss_agg_mode=token-mean` | 跨 mini-batch 有效 token 做均值；要和长度上限一同解释。[固定版 DAPO 文档](../verl/docs/algo/dapo.md) |
| 超长 shaping | `reward_model.overlong_buffer` | 在 hard max response length 前的 buffer 中线性增加惩罚。[固定版 DAPO 文档](../verl/docs/algo/dapo.md) |

工程上还要明确两个 batch：`gen_batch_size` 是为过滤而产生的候选规模，`train_batch_size` 是最终用于更新的 prompt 数；二者不等。若只报告 train batch，无法估计 dynamic sampling 的真实 rollout 成本。该关系由官方 DAPO recipe 的配置说明给出。[verL DAPO recipe](https://verl.readthedocs.io/en/latest/algo/dapo.html)

## 5. 优点、限制与失败模式

**收益**

- Clip-Higher 旨在缓解熵塌缩、保留探索；论文在其设置中观察到更高 entropy 和更多样输出。[DAPO §3.1](https://arxiv.org/html/2503.14476v1#S3.SS1)
- Dynamic Sampling 不让无梯度组稀释训练 batch；论文报告虽要生成更多样本，但在其同步系统中总体收敛时间未显著增加。[DAPO §3.2、§4.1](https://arxiv.org/html/2503.14476v1#S3.SS2)
- token-level loss 与 overlong shaping 直接针对长 CoT 的长度偏置和截断 reward 噪声。[DAPO §3.3–§3.4](https://arxiv.org/html/2503.14476v1#S3.SS3)

**需要主动防范的失败模式**

- **过滤后分布偏移**：只训练“半会半不会”的题会改变样本分布；必须监控每个难度段和 reward 桶的保留率。
- **重采样无上限**：若 reward 几乎总是 0 或 1，过滤无法填满 batch。为 `max_num_gen_batches` 设上限并把失败视为数据/reward 告警，而不是无限等待。[官方配置说明](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- **长度惩罚替代了正确性**：overlong penalty 是 shaping，不是质量判定；惩罚过强会压制必要的长推理，过弱则不能抑制跑飞长度。
- **把熵当作唯一目标**：论文指出低熵意味着探索不足、过高熵则可能对应乱码/重复；需与验证准确率、reward、生成长度联看。[DAPO §4.3](https://arxiv.org/html/2503.14476v1#S4.SS3)
- **不可复现的“配方搬运”**：官方 verl 文档也提醒 RL 基础设施仍不够稳健，建议一次只改变一项；CUDA graph 等运行时选择也可能影响结果。[固定版 DAPO FAQ](../verl/docs/algo/dapo.md)

## 6. 与本仓库 verl 学习路径的关系

本仓库固定的是 [`verl@c4b389ad`](https://github.com/volcengine/verl/tree/c4b389adadc58ce51cb2b63e70df497ca166d77f)，其内置 DAPO 文档和实现入口；它不是论文当年的唯一复现实验环境。因此学习顺序应是：

1. 先读 [GRPO 笔记](grpo.md)，能解释 group advantage、PPO ratio 和 KL 后再看 DAPO。
2. 阅读 [本地 DAPO recipe](../verl/docs/algo/dapo.md)，把四项技术逐项映射到配置与核心代码。
3. 阅读 [group filter 配置](../verl/verl/trainer/config/algorithm.py) 与 reward manager 中的 DAPO 实现，验证“过滤”和“长度 shaping”究竟在何处发生。
4. 再参考[官方 DAPO repo](https://github.com/BytedTsinghua-SIA/DAPO)与[verl-recipe DAPO 配方](https://github.com/verl-project/verl-recipe/tree/main/dapo)，比较论文复现环境和当前框架的差异。

最小实验应只启用基础 GRPO 后逐一加一项：先 Clip-Higher，再 group filtering、token-mean、overlong shaping。每轮固定种子、数据、评测与总 rollout token，记录有效组率、discard/retry、长度分布、entropy、KL、reward 和验证集成绩；否则消融结论无法归因。

## 7. 一手阅读入口

- [DAPO 原论文](https://arxiv.org/abs/2503.14476)
- [论文 HTML（算法、消融与训练动态）](https://arxiv.org/html/2503.14476v1)
- [DAPO 官方代码与训练记录](https://github.com/BytedTsinghua-SIA/DAPO)
- [verL 官方 DAPO 文档](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- [本仓库固定 commit 的 DAPO 文档](../verl/docs/algo/dapo.md)
