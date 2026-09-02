# 特性对比图谱：Why / Without / With / Trade-off

本页按统一问题组织所有特性图：它因为什么瓶颈产生；没有它时系统怎样失败；采用后改变了哪一段数据流；又引入什么新成本。

## 推理特性

| 特性 | 产生原因 | Without | With | 新代价 | 深入阅读 |
| --- | --- | --- | --- | --- | --- |
| KV Cache | 自回归每步会重复计算历史 K/V | 每个新 token 重算全部历史 | 缓存历史 K/V，只追加新 token | 显存线性增长、仍需读取历史 | [PD/KV Cache](inference/pd-disaggregation.md) |
| 连续批处理 | 请求输出长度不同造成 batch 空洞 | 短请求完成后等待最长序列 | 每轮重组 batch、完成即换入新请求 | 调度、公平性与 P99 | [Batch/Cache](inference/batching-cache-and-acceleration.md#1-从静态批处理到连续批处理) |
| Paged KV | KV 动态增长导致预留浪费和碎片 | 连续大块、并发受限 | 逻辑 block 映射到离散物理页 | block table、页大小与调度 | [Batch/Cache](inference/batching-cache-and-acceleration.md#2-paged-kv-cache逻辑连续物理可离散) |
| Prefix Cache | 系统提示/RAG 前缀重复 | 重复执行同一 Prefill | 共享公共前缀 KV，从分叉点继续 | 命中率、驱逐和版本一致性 | [Batch/Cache](inference/batching-cache-and-acceleration.md#3-prefix-cache相同前缀只做一次-prefill) |
| 量化 | 权重容量和 Decode 带宽成为瓶颈 | FP16/BF16 占用与传输字节高 | INT8/INT4 以 scale 近似表示 | 量化误差、校准和 kernel | [Batch/Cache](inference/batching-cache-and-acceleration.md#4-量化减少容量与带宽但要有匹配-kernel) |
| 投机解码 | 大模型 Decode 串行、一次只前进一步 | 目标模型逐 token forward | Draft 多步提议、Target 批量验证 | Draft/验证成本、接受率依赖 | [Batch/Cache](inference/batching-cache-and-acceleration.md#5-投机解码草稿提议目标模型批量验证) |
| PD 分离 | Prefill/Decode 资源特征不同且互相干扰 | 同一 Worker 抢资源 | P/D 独立伸缩，以 KV 交接 | KV 传输、背压与池间失衡 | [PD 分离](inference/pd-disaggregation.md) |

## 强化学习特性

| 特性 | 产生原因 | Without | With | 新代价 | 深入阅读 |
| --- | --- | --- | --- | --- | --- |
| PPO Clip | rollout batch 多轮复用，策略易移动过远 | ratio 极端、旧数据迅速失效 | 限制过度更新带来的目标收益 | ε/LR/Epoch 联动，不是硬 KL | [从 MDP 到 PPO](reinforcement-learning/foundations-to-ppo.md#5-ppo-为什么需要-ratio-与-clip) |
| GRPO | 大型 Critic 带来显存、计算和同步成本 | Actor + Critic 共同训练 | 同题多样本的组内相对优势 | 更多 rollout、同分组无信号 | [GRPO](../docs/grpo.md) |
| DAPO | 长 CoT 出现熵塌缩、无效组、长度偏置和截断噪声 | 朴素 GRPO 的四类不稳定 | Clip-Higher、动态采样、token loss、长度 shaping | 过滤成本、分布偏移与额外校准 | [DAPO](../docs/dapo.md) |
| 训推一致性 | rollout 与训练由不同引擎执行 | token/log-prob 不可比，ratio/KL 失真 | 对齐权重、Tokenizer、位置、精度和概率语义 | 同步、版本与对齐测试成本 | [训推一致性](reinforcement-learning/training-inference-consistency.md) |

这些图是判断框架，不是性能承诺。任何 With 方案都要用固定负载和外部质量指标验证；当产生原因不存在时，新增复杂度可能大于收益。
