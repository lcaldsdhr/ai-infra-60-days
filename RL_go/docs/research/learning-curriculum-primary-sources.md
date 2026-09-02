# LLM 推理与强化学习课程：一手资料索引

> 调研日期：2026-09-02
> 范围：Transformer/Attention、LLM 推理系统、经典强化学习、LLM RL/RLHF、Agentic RL、verl。
> 选材原则：优先论文原文、会议页面、官方文档和上游源码；不把二手博客作为事实依据。论文中的性能数字只代表论文给定的模型、硬件和负载，不应直接当作生产承诺。

## 建议阅读顺序

1. Transformer 与 Attention：先能手算一次 causal self-attention，再理解 MHA、GQA 和 FlashAttention。
2. 单机推理：先区分 prefill/decode，再学习 KV Cache、连续批处理、PagedAttention、量化与投机解码。
3. 分布式推理：把 TP/PP/DP 与 TTFT/TPOT 联系起来，再进入 P/D 分离和跨节点 KV 传输。
4. 经典 RL：MDP → Bellman 方程 → Policy Gradient Theorem → REINFORCE → GAE/PPO。
5. LLM RL：RLHF 数据流 → GRPO → DAPO → SPIN；同时对照实现，避免只记公式名称。
6. Agentic RL 与 verl：把单轮 token 生成扩展为带工具、环境反馈、长轨迹信用分配的 MDP，最后沿 verl 数据流阅读源码。

---

## 1. Transformer 与 Attention

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| Attention Is All You Need | Transformer 用 scaled dot-product attention、multi-head attention、位置编码、残差连接和前馈网络替代循环/卷积；自注意力允许训练阶段对序列位置并行计算。 | [论文（arXiv）](https://arxiv.org/abs/1706.03762) | 精读 §3 和图 1；自己推导 `softmax(QK^T/sqrt(d_k))V` 的张量形状、causal mask 与复杂度。 |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | MQA 让所有 query heads 共享一组 K/V，可显著减少 decode 时的 KV 容量和带宽；GQA 使用介于 MHA 与 MQA 之间的 KV heads 数量，在质量与推理效率间折中。 | [论文（arXiv）](https://arxiv.org/abs/2305.13245) | 建立 `num_attention_heads`、`num_key_value_heads` 与每 token KV 字节数之间的定量关系。 |
| FlashAttention | 标准 attention 的实际瓶颈不仅是 FLOPs，还包括 HBM 与片上 SRAM 之间的 IO；通过 tiling 和重计算，可在保持精确 attention 的同时减少 HBM 读写。 | [论文（arXiv）](https://arxiv.org/abs/2205.14135)；[官方源码](https://github.com/Dao-AILab/flash-attention) | 区分“计算复杂度”“显存占用”“IO 复杂度”；理解为什么 kernel 优化不等于稀疏/近似 attention。 |
| FlashAttention-2 | 通过减少非矩阵乘法 FLOPs、改善 sequence 维并行和 warp 工作划分，进一步提升 attention kernel 利用率。 | [论文（arXiv）](https://arxiv.org/abs/2307.08691) | 用于进阶 kernel 课：从算法层 tiling 进入 GPU work partitioning，而不是只调用现成算子。 |

### 本主题应形成的课程图

- Transformer block：RMSNorm/LayerNorm、Attention、MLP、残差的数据流。
- MHA → GQA → MQA：Q heads 与 KV heads 的对应关系，以及 KV Cache 容量变化。
- 普通 attention 与 FlashAttention：HBM/SRAM 数据移动对比图。

---

## 2. 自回归推理、KV Cache、连续批处理与 PagedAttention

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| Hugging Face Transformers：KV Cache 官方说明 | 自回归 decode 若不缓存历史 K/V，会在每一步重复计算前缀；缓存后只计算新 token 的投影，但 cache 随序列长度线性增长，并在训练时通常不使用。 | [官方文档](https://huggingface.co/docs/transformers/main/en/cache_explanation) | 先用最小 PyTorch 代码比较有/无 cache 的 logits 一致性和张量形状；明确 cache 减少的是重复计算，不会消除对历史 K/V 的读取。 |
| Orca | 请求级静态批处理不能适配不同输出长度；iteration-level scheduling 在每个模型迭代重新组成 batch，已完成请求可退出、新请求可加入，是 continuous batching 的系统基础。 | [OSDI 2022 论文与演讲页](https://www.usenix.org/conference/osdi22/presentation/yu) | 理解吞吐、排队延迟和 head-of-line blocking；画出 static batching 与 iteration-level batching 的时间线。 |
| PagedAttention / vLLM | KV Cache 长度动态变化会产生碎片和冗余；PagedAttention 借鉴虚拟内存分页，用逻辑块到物理块映射实现按需分配、共享与 copy-on-write。论文报告其在给定实验下相对若干基线提升吞吐，但数值不可脱离负载复用。 | [SOSP 2023 论文（arXiv）](https://arxiv.org/abs/2309.06180)；[vLLM 官方源码](https://github.com/vllm-project/vllm) | 精读 block table、KV block 分配与共享；把“PagedAttention 算子”和“vLLM 整体调度器”分开理解。 |
| vLLM 调度器实现 | 生产实现把 token budget、running/waiting 请求、preemption、prefix cache 和 KV block 管理放进统一调度循环。 | [scheduler.py（上游源码）](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) | 代码阅读入口：沿 `schedule()` 追踪一个请求从排队、分配 blocks、运行到释放 cache 的生命周期。 |
| Sarathi-Serve | 将长 prefill 切成 chunks，并与 decode 请求 piggyback，可缓解长 prefill 对 decode 延迟的干扰，改善吞吐—延迟权衡。 | [论文（arXiv）](https://arxiv.org/abs/2403.02310) | 为 chunked prefill 与后续 P/D 分离建立对照：前者仍可共置，后者把两个阶段放到不同设备。 |

### 必须明确的计算口径

- Prefill：一次处理提示 tokens，通常矩阵规模较大、并行度高；主要指标是 TTFT。
- Decode：每步产生一个或少量 token，持续读取权重和历史 KV；主要指标是 TPOT/ITL。
- 每层 KV Cache 近似容量：`2 × batch × sequence_length × num_kv_heads × head_dim × bytes_per_element`；模型总量再乘层数。
- “使用 KV Cache 后每步是 O(1)”只适用于不再重复做历史 token 的 K/V/MLP 投影；标准 attention 对历史 cache 的读取和新 query 与历史 keys 的点积仍随上下文长度增长。

---

## 3. Speculative Decoding

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| Fast Inference from Transformers via Speculative Decoding | 小 draft model 连续提出多个候选 token，大 target model 并行验证；通过接受/拒绝与修正采样，可在不改变目标分布的前提下减少 target model 串行调用次数。论文在其 T5-XXL 设置中报告约 2–3× 加速。 | [论文（arXiv）](https://arxiv.org/abs/2211.17192) | 精读 acceptance rule，证明输出分布保持不变；测量 acceptance rate，而不只报告 tokens/s。 |
| Accelerating Large Language Model Decoding with Speculative Sampling | 独立提出 speculative sampling，用近似模型采样候选、目标模型并行评分，在保持原模型输出分布的条件下加速。 | [论文（arXiv）](https://arxiv.org/abs/2302.01318) | 与上一论文对照符号和算法步骤，理解 greedy、sampling 场景的不同接受逻辑。 |
| Medusa | 在目标模型上增加多个 decoding heads 并行预测后续 token，以 tree attention 一次验证多个候选；避免维护独立 draft model，但需要特定训练/微调。 | [论文（arXiv）](https://arxiv.org/abs/2401.10774)；[官方源码](https://github.com/FasterDecoding/Medusa) | 对比独立 draft、self-speculation、多头 draft 的训练成本、内存和接受率。 |
| vLLM speculative decoding | vLLM 的实现文档列出 draft-model、n-gram、Medusa/EAGLE 等配置与兼容限制；实际收益依模型、序列和并发而变。 | [官方文档](https://docs.vllm.ai/en/stable/features/spec_decode/) | 作为实验入口：固定输出分布/质量，分离 draft 开销、verify 开销、接受长度与端到端延迟。 |

---

## 4. 量化与低精度推理

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| LLM.int8() | 对大多数矩阵乘使用 8-bit，同时把少量 activation outlier 通道保留在更高精度路径，以降低量化误差。 | [论文（arXiv）](https://arxiv.org/abs/2208.07339) | 理解 weight-only 与 weight-activation quantization 的不同瓶颈，以及 outlier 对 scale 的影响。 |
| SmoothQuant | 用数学等价的通道缩放把 activation 的量化难度迁移到 weights，使 W8A8 PTQ 更可行；不需要重新训练。 | [论文（arXiv）](https://arxiv.org/abs/2211.10438)；[官方源码](https://github.com/mit-han-lab/smoothquant) | 手推 `XW = (XS^-1)(SW)`；讨论 calibration 数据和缩放超参数如何影响误差。 |
| GPTQ | 使用近似二阶信息逐列补偿误差，实现一次性低比特 weight-only PTQ；论文覆盖 3/4-bit 大模型量化。 | [论文（arXiv）](https://arxiv.org/abs/2210.17323)；[官方源码](https://github.com/IST-DASLab/gptq) | 理解“权重文件更小”不自动等于“端到端更快”：还取决于 dequant/fused kernel、batch size 和硬件。 |
| AWQ | 用 activation 统计识别少量显著权重通道，并通过等价缩放保护这些通道，避免硬件不友好的混合精度拆分。 | [论文（arXiv）](https://arxiv.org/abs/2306.00978)；[官方源码](https://github.com/mit-han-lab/llm-awq) | 与 GPTQ 对比：二阶重构 vs activation-aware scaling；实验同时记录 perplexity、峰值显存、prefill/decode 延迟。 |
| vLLM Quantization | 官方兼容矩阵与配置说明展示不同量化后端、GPU 架构和模型格式的支持边界。 | [官方文档](https://docs.vllm.ai/en/stable/features/quantization/) | 选实现时先查 kernel/硬件支持，不根据算法名假设可用；课程代码应固定 vLLM、CUDA 与模型版本。 |

---

## 5. 分布式推理与 Prefill/Decode 分离

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| Megatron-LM 模型并行 | Tensor parallel 把单层矩阵运算分片到多个设备，通过 collective 合并；它解决单卡容量/算力问题，但引入高频通信。 | [论文（arXiv）](https://arxiv.org/abs/1909.08053)；[官方源码](https://github.com/NVIDIA/Megatron-LM) | 手画 column/row parallel linear 的通信；建立 TP degree、通信量和单卡 kernel 尺寸之间的权衡。 |
| 3D Parallelism / Megatron-LM | TP、PP、DP 可组合；pipeline schedule 的 bubble、microbatch 数和跨节点拓扑会影响效率。 | [论文（arXiv）](https://arxiv.org/abs/2104.04473) | 虽以训练为主，但用于掌握 TP/PP/DP 词汇和拓扑原则；推理课应另行测量 decode 的小 batch 行为。 |
| DeepSpeed Inference | 将 kernel 优化、模型并行以及 CPU/NVMe heterogeneous memory 组合，用于模型可放入多 GPU或需 offload 的不同情形。 | [论文（arXiv）](https://arxiv.org/abs/2207.00032)；[官方源码](https://github.com/deepspeedai/DeepSpeed) | 对比分片计算与 offload：前者依赖互联带宽，后者常受 PCIe/NVMe 数据移动限制。 |
| vLLM distributed serving | vLLM 支持 TP/PP、多节点 Ray 或 multiprocessing；官方建议通常从单节点 TP 基线开始，并结合节点内/节点间拓扑选 TP/PP。 | [官方文档](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) | 实操 `--tensor-parallel-size`、`--pipeline-parallel-size`；验证 world size、显存和通信，而非只验证服务能启动。 |
| Splitwise | Prefill 偏 compute-intensive、decode 偏 memory-intensive；将两阶段放在独立机器，可分别选硬件和扩容，但必须传输 KV 状态。 | [论文（arXiv）](https://arxiv.org/abs/2311.18677) | 建立 P/D 分离的第一张成本表：TTFT、TPOT、KV 字节数、传输带宽、队列与资源比例。 |
| DistServe | 共置 prefill/decode 会相互干扰并耦合两阶段的资源与并行配置；DistServe 分配到不同 GPU，并以 TTFT/TPOT SLO 下的 goodput 为优化目标。 | [OSDI 2024 论文与演讲页](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | 精读 goodput 定义、placement 与并行策略搜索；理解吞吐高但违反尾延迟 SLO 的系统并不等价。 |
| Mooncake | 以 KVCache 为中心，把 prefill/decode 集群和 CPU DRAM/SSD cache 资源解耦，调度器同时考虑 SLO、cache 命中与过载拒绝。 | [论文（arXiv）](https://arxiv.org/abs/2407.00079)；[官方源码](https://github.com/kvcache-ai/Mooncake) | 从“两类 GPU 池”进阶到分层 KV 存储与传输；画出 KV 在 GPU、CPU、SSD、网络间的生命周期。 |
| vLLM disaggregated serving example | 官方示例展示独立 prefill/decoder 实例、KV connector 与 proxy 的基本部署形态；文档明确该功能目标不是自动提高所有负载吞吐，而是独立调 TTFT/ITL 与故障域。 | [官方示例](https://docs.vllm.ai/en/stable/examples/disaggregated/disaggregated_serving/) | 作为代码实验入口；记录 connector、传输网络、P:D 比、prefix 长度和输出长度。 |

---

## 6. 经典强化学习：MDP、Policy Gradient、REINFORCE、GAE、PPO

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| Reinforcement Learning: An Introduction（第 2 版） | 统一定义 agent、environment、state、action、reward、return、policy、value function、Bellman equation、on/off-policy 与 prediction/control。 | [作者官方 PDF](http://incompleteideas.net/book/RLbook2020.pdf) | 精读第 3、4、9、13 章；先在 tabular MDP 上验证 Bellman expectation/optimality equation，再进入函数逼近。 |
| Policy Gradient Methods for RL with Function Approximation | Policy Gradient Theorem 把目标函数梯度写成可由经验采样估计的形式，并允许以 action-value/advantage 辅助估计；奠定 actor-critic 理论。 | [NeurIPS 论文页](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) | 推导 score-function identity，理解为什么状态分布对参数的导数不在最终估计式中显式出现。 |
| REINFORCE | Williams 提出一类随机策略的梯度跟随算法；更新方向与期望回报梯度一致。实际估计方差高，baseline 可在不引入偏差的条件下降方差。 | [作者论文列表](https://ccs.neu.edu/home/rjw/pubs.html)；[DOI](https://doi.org/10.1007/BF00992696) | 从 `-G_t log π(a_t|s_t)` 写最小实现；用多随机种子观察高方差和 baseline 的作用。 |
| Generalized Advantage Estimation | GAE 用参数 λ 在低方差的一步 TD 与高方差 Monte Carlo return 之间折中；与 γ 一起决定 advantage 估计偏差—方差。 | [论文（arXiv）](https://arxiv.org/abs/1506.02438) | 手算一条短 trajectory 的 δ、GAE 与 returns；检查终止状态 mask，避免把 padding/bootstrap 写错。 |
| Trust Region Policy Optimization | TRPO 以策略分布的 KL 约束限制更新幅度，并给出单调改进相关理论动机；PPO 的“proximal”思想由此而来。 | [论文（arXiv）](https://arxiv.org/abs/1502.05477) | 不要求先实现二阶优化，但要理解 importance ratio、surrogate objective 与 KL trust region 的关系。 |
| Proximal Policy Optimization | PPO 交替采样与多轮 minibatch 更新；clipped surrogate 限制过大的 policy ratio 收益，使实现比 TRPO 简单。 | [论文（arXiv）](https://arxiv.org/abs/1707.06347) | 手画正/负 advantage 下的 clipped objective；实现时同时监控 approx KL、clip fraction、entropy、value loss。 |

### 经典 RL 到 LLM RL 的映射

| 经典 RL 元素 | 单轮 LLM 生成中的常用映射 | Agentic RL 中的扩展 |
|---|---|---|
| state | prompt 与已生成 token 前缀 | 对话历史、工具结果、环境观测、隐藏执行状态 |
| action | 下一个 token | token、结构化 tool call，或一个语义级动作 |
| transition | 拼接 token | 外部工具/网页/代码执行环境改变状态并返回观测 |
| reward | response 末尾的偏好/可验证分数，或 token-level reward | 稀疏任务成功、过程奖励、成本/安全约束、多回合 reward |
| episode | 一条 completion | 从任务开始到成功、失败、超时或预算耗尽的完整 trajectory |

---

## 7. RLHF 与 Preference Optimization

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| Deep RL from Human Preferences | 当奖励函数难以手写时，可用人类对 trajectory segments 的成对偏好训练 reward predictor，再以该模型提供奖励训练策略。 | [论文（arXiv）](https://arxiv.org/abs/1706.03741) | 理解 preference data → reward model → policy optimization 的原始闭环，以及 reward hacking/分布漂移风险。 |
| Learning to Summarize from Human Feedback | 将人类比较数据训练为 reward model，再用 RL 优化摘要策略，展示偏好学习用于语言生成的完整流程。 | [论文（arXiv）](https://arxiv.org/abs/2009.01325) | 对比 supervised label 与 pairwise preference；关注 reward model validation 和人评设计。 |
| InstructGPT | 典型三阶段：收集示范做 SFT；对多候选排序训练 reward model；用 PPO 优化策略并以 KL 约束其不要远离 SFT/reference policy。 | [论文（arXiv）](https://arxiv.org/abs/2203.02155) | 画出 actor、reference、reward model、value model 的数据流与显存需求；区分训练奖励和最终人评。 |
| Constitutional AI / RLAIF | 用书面原则指导模型自我批评与修订，并用 AI preferences 训练 preference model，再做 RL；减少逐样本人类有害性标签需求。 | [论文（arXiv）](https://arxiv.org/abs/2212.08073) | 理解“反馈来源”与“优化算法”是两个维度；RLAIF 仍需检查原则覆盖、judge 偏差和可操纵性。 |
| Direct Preference Optimization | 对标准 KL-regularized RLHF 目标重新参数化，可将偏好优化化为 policy 与 reference 的对数概率比分类损失，无需显式 reward model 和在线 RL rollout。 | [论文（arXiv）](https://arxiv.org/abs/2305.18290)；[官方源码](https://github.com/eric-mitchell/direct-preference-optimization) | 与 PPO-RLHF 对照数据需求：DPO 是离线 preference optimization，不应与 on-policy、环境交互 RL 混为一谈。 |

---

## 8. GRPO、DAPO 与 SPIN

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| DeepSeekMath / GRPO | GRPO 对同一 prompt 采样一组 responses，用组内 reward 相对值估计 advantage，省去 PPO 的独立 critic/value model；论文将其用于数学推理。 | [论文（arXiv）](https://arxiv.org/abs/2402.03300) | 从 PPO 数据流中删掉 critic 后重画显存与计算；手算 group mean/std、advantage、ratio clipping 与 KL 项。 |
| DeepSeek-R1 | 展示在可验证任务上以大规模 RL 激发 reasoning 行为，并给出 R1-Zero 与多阶段训练路线；技术报告也是理解 outcome reward、格式 reward 与冷启动数据的重要入口。 | [论文（arXiv）](https://arxiv.org/abs/2501.12948)；[官方仓库](https://github.com/deepseek-ai/DeepSeek-R1) | 区分报告事实与外界推测；分析奖励可验证性、语言混杂、可读性和蒸馏数据的角色。 |
| DAPO | 在 GRPO 风格基础上提出 decoupled clipping、dynamic sampling、token-level policy-gradient loss 与 overlong reward shaping，目标是改善大规模 reasoning RL 的稳定性与有效样本利用。 | [论文（arXiv）](https://arxiv.org/abs/2503.14476)；[官方项目页](https://dapo-sia.github.io/)；[官方仓库](https://github.com/BytedTsinghua-SIA/DAPO) | 每项技术单独做 ablation 解读；特别追踪全 0/全 1 reward group 被动态采样过滤后的数据分布。 |
| SPIN | 从 SFT model 出发，让当前模型区分 target demonstrations 与上一轮模型生成，迭代 self-play fine-tuning；不依赖新增人类偏好或更强 teacher。 | [论文（arXiv）](https://arxiv.org/abs/2401.01335)；[官方源码](https://github.com/uclaml/SPIN) | 明确 SPIN 的“self-play”是分布匹配/判别式目标，不等同于 PPO/GRPO 的环境回报优化。 |

### 重要辨析

- PPO：通常需要 critic/value estimates；可在同一批 on-policy trajectories 上做多轮更新，但需控制 policy drift。
- GRPO：用同 prompt 的组内结果构造 baseline/advantage，降低 critic 内存成本；有效性依赖每组多样性与 reward 信号。
- DAPO：不是 GRPO 的简单改名，而是一组针对大规模 reasoning RL 训练动态和 loss aggregation 的系统化修改。
- SPIN：属于迭代自博弈式 fine-tuning；课程中应与 RL 算法并列比较，而不应直接标注成标准 policy-gradient RL。

---

## 9. Agentic RL

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| ReAct | 让模型交错生成 reasoning traces 与 environment actions；行动获取的新观测反过来更新计划，是 agent trajectory 的基础范式。 | [论文（arXiv）](https://arxiv.org/abs/2210.03629)；[项目页/代码](https://react-lm.github.io/) | 先实现不训练的 agent loop；明确 observation 不应作为模型 action 计算 policy loss。 |
| WebArena | 提供可复现、功能完整的网站环境和长程任务，以最终功能状态验证成功，而不只做文本匹配；暴露真实 agent 的规划、恢复和长 horizon 难点。 | [论文（arXiv）](https://arxiv.org/abs/2307.13854)；[官方源码](https://github.com/web-arena-x/webarena) | 学会设计可自动验证的环境 reward、reset、超时和 side-effect 隔离，为在线 RL 采样做准备。 |
| Agent Lightning | 将 agent 执行与 RL 训练解耦，把多步骤 agent execution 表达为 MDP，并引入 trajectory/transition 与层级信用分配接口。 | [论文（arXiv）](https://arxiv.org/abs/2508.03680)；[微软官方源码](https://github.com/microsoft/agent-lightning) | 研究 observability、trajectory store、credit assignment 和 trainer-agent separation；特别检查动态 workflow 的 action 边界。 |
| Agent-R1 | 给出 LLM agent 的 MDP 形式化和端到端 RL 框架，将多轮工具/环境交互纳入训练，而不是只对最终 completion 做单轮优化。 | [论文（arXiv）](https://arxiv.org/abs/2511.14460) | 作为较新的实现案例阅读；对比单轮 GRPO 的 response mask、reward placement 与多轮 trajectory mask。 |
| INFERCEPT | Agent/tool use 会中断 LLM decode；若每次中断都丢弃上下文会产生 KV 重算。系统可依据中断时长选择保存、交换或丢弃 cache。 | [ICML 2024 论文（arXiv）](https://arxiv.org/abs/2402.01869) | 把 Agentic RL 与推理系统连接起来：训练吞吐不仅取决于 policy loss，还取决于工具等待、并发和 KV 生命周期。 |

### Agentic RL 课程必须覆盖的额外问题

- Action granularity：token-level、turn-level、tool-call-level 的 policy ratio 和 credit assignment 不相同。
- Masking：只对 policy 产生的 tokens 计 loss；system/user/tool observations 与 padding 必须正确遮罩。
- Rewards：终局成功奖励稀疏，过程奖励可能被钻漏洞；应记录 reward provenance 与 verifier 版本。
- On-policy freshness：工具调用很慢时，rollout 与训练并行可能造成策略滞后，需要衡量 version gap 和 importance weight。
- Environment safety：网页、shell、数据库环境必须可 reset、限权、限时并记录 side effects；离线 benchmark 成功不等价于可安全上线。

---

## 10. verl 官方论文、文档与固定版本源码

本仓库的 `RL_go/verl` 是外部源码 submodule；本次调研读取到的固定提交为：

```text
c4b389adadc58ce51cb2b63e70df497ca166d77f
```

课程材料应优先链接此提交，而不是随时间变化的 `main`。不要直接修改 submodule 源码；实验代码放在 `RL_go/code/`。

### 10.1 架构与入门

| 一手资料 | 核心结论 | 原始链接 | 建议阅读目的 |
|---|---|---|---|
| HybridFlow / verl 论文 | RLHF dataflow 的每个逻辑节点内部又是分布式训练/生成程序；HybridFlow 组合 single-controller 与 multi-controller，并以 3D-HybridEngine 处理 actor 在训练与 rollout 间的 resharding。 | [论文（arXiv）](https://arxiv.org/abs/2409.19256) | 先画 actor、rollout、reference、critic、reward 的依赖，再读 Ray workers；避免从单个 loss 函数误解整个系统。 |
| verl 官方文档首页 | verl 将训练后端（FSDP/Megatron 等）与 rollout 后端（vLLM/SGLang 等）解耦，并提供 PPO/GRPO、reward、agent loop 等入口。 | [官方文档](https://verl.readthedocs.io/en/latest/) | 作为最新功能索引；实际课程引用则用下方 pinned source，避免文档与 submodule 版本错位。 |
| Quickstart：PPO on GSM8K | 官方最小闭环包含数据预处理、reward function、配置和启动命令。 | [官方文档](https://verl.readthedocs.io/en/latest/start/quickstart.html)；[固定版本文档源](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/docs/start/quickstart.rst) | 第一个可运行实验应缩小 batch/model，并验证数据字段、reward、生成、advantage、update 与 checkpoint 全链路。 |
| Agentic RL quickstart | 官方把单轮生成扩展为 agent loop、多轮 tool calls 与自定义 reward。 | [固定版本文档源](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/docs/start/agentic_rl.rst) | 对照单轮 quickstart 标出新增的 environment、tool schema、trajectory 和 mask。 |

### 10.2 算法源码阅读路线

| 源码入口 | 核心结论 | 固定版本链接 | 建议阅读目的 |
|---|---|---|---|
| `core_algos.py` | 集中实现 GAE、GRPO/RLOO/ReMax 等 advantage estimator、policy/value loss、KL penalty 与聚合方式；同名算法的关键语义常由 config 决定。 | [固定提交源码](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/trainer/ppo/core_algos.py) | 重点追 `compute_gae_advantage_return`、`compute_grpo_outcome_advantage`、`compute_policy_loss`、`compute_value_loss`、`kl_penalty`；逐项记录输入 shape 和 mask。 |
| `ray_trainer.py` | 把 rollout、reward、old/ref log-prob、values、advantage 和参数更新编排为分布式训练循环。 | [固定提交源码](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/trainer/ppo/ray_trainer.py) | 从 `RayPPOTrainer` 和 `compute_advantage` 追一批 `DataProto` 的 key 变化；这是理解算法公式如何落地的主干。 |
| `main_ppo.py` | Hydra 入口负责配置、数据集、worker/resource pool 映射和 trainer 启动。 | [固定提交源码](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/trainer/main_ppo.py) | 从命令行 override 反查最终对象；课程代码应保存 resolved config，而非只保存启动脚本。 |
| PPO 配置 | actor、rollout、reference、critic、reward 与算法参数的默认值和组合关系在配置层表达。 | [固定提交配置](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/trainer/config/ppo_trainer.yaml) | 建立“论文符号 → config key → 源码函数 → 训练指标”映射表。 |
| PPO 文档 | 官方解释 PPO 目标和 verl 对应配置/实现。 | [固定版本文档源](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/docs/algo/ppo.md) | 与原始 PPO 论文对读，识别 LLM 场景中的 sequence masking、KL 与 loss aggregation。 |
| GRPO 文档 | 官方解释组相对 advantage 与相关配置。 | [固定版本文档源](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/docs/algo/grpo.md) | 对照 `compute_grpo_outcome_advantage` 验证 group id、reward 汇总、标准差归一化和广播到 token 的过程。 |
| DAPO 文档 | 官方记录 DAPO 在 verl 中的算法组件及其配置入口。 | [固定版本文档源](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/docs/algo/dapo.md) | 把 decoupled clip、dynamic sampling、token-level loss、overlong shaping 分别映射到代码/recipe；不要把整套算法压成一个开关。 |
| SPIN 文档 | 官方记录 SPIN 的数据与训练循环实现入口。 | [固定版本文档源](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/docs/algo/spin.md) | 与 SPIN 原论文对读，明确生成数据来自哪个迭代 checkpoint，以及 target/model samples 如何组成训练对。 |

### 10.3 Rollout、reward 与 agent loop

| 源码入口 | 核心结论 | 固定版本链接 | 建议阅读目的 |
|---|---|---|---|
| Rollout base/schema | 定义 generation request/response、序列、log probabilities 与后端接口，是 vLLM/SGLang/HF rollout 的公共契约。 | [base.py](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/workers/rollout/base.py)；[schemas.py](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/workers/rollout/schemas.py) | 先理解公共数据结构，再进入特定 engine；检查 prompt/response 长度、attention mask、position ids 和 EOS。 |
| vLLM rollout | 封装 vLLM 推理、权重同步与序列生成，使 actor 更新后的参数进入下一轮 rollout。 | [固定提交源码](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/workers/rollout/vllm_rollout/vllm_rollout.py) | 重点看 train/rollout 模式切换、显存释放/恢复、sampling params 和 weight update；这决定 on-policy 吞吐和一致性。 |
| Agent loop | 定义多轮 agent 执行与工具调用，将环境交互结果组织成可训练 trajectory。 | [agent_loop.py](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/experimental/agent_loop/agent_loop.py)；[tool_agent_loop.py](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/experimental/agent_loop/tool_agent_loop.py) | 沿一次 tool call 检查 state、action、observation、tokenization 和 loss mask；用失败/超时案例验证终止条件。 |
| Reward function 准备 | verl 支持 function-based/verifiable reward 与 model-based reward，数据字段和并发执行方式会影响正确性与吞吐。 | [固定版本文档源](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/docs/preparation/reward_function.rst)；[reward.py](https://github.com/verl-project/verl/blob/c4b389adadc58ce51cb2b63e70df497ca166d77f/verl/trainer/ppo/reward.py) | 课程实验要为 reward 写单元测试，包括正确、错误、格式异常、超时、空输出和恶意输出。 |

---

## 11. 建议的代码与图片产出对应表

本表不是新增事实来源，而是把上述一手资料转成可验证的学习制品。

| 模块 | 图片/图表 | 最小代码参考 | 验收点 |
|---|---|---|---|
| Attention | Q/K/V shape 流程图；MHA/GQA/MQA heads 图；FlashAttention IO 图 | NumPy/PyTorch causal attention；与 `scaled_dot_product_attention` 对齐 | 小张量 logits/outputs 一致；mask 无未来信息泄漏 |
| KV Cache | prefill/decode 时间线；每层 cache 布局；block table | 手写单 batch KV cache decoder；容量计算器 | cached/uncached logits 一致；容量公式与张量实际字节一致 |
| Serving | static/continuous batch 甘特图；PagedAttention 分页图 | vLLM 离线吞吐脚本和在线压测脚本 | 分开报告 TTFT、TPOT、吞吐、P50/P99、显存 |
| Spec decode | draft→verify→accept/reject 流程 | 小词表精确 speculative sampler | 固定种子统计分布与 target sampler 一致；记录 acceptance length |
| Quantization | per-tensor/per-channel scale 图；outlier 图 | 对线性层做 INT8/INT4 fake quant | 同时报误差、模型质量、内存与真实 kernel 延迟 |
| Distributed/P-D | TP collective 图；PP stage 图；P→KV→D 数据流 | vLLM TP 启动与两实例 disaggregated demo | 记录 GPU 拓扑、网络、KV 传输量、TTFT/TPOT 和失败恢复 |
| MDP/PG | trajectory 与 return 图；Bellman backup 图 | GridWorld、REINFORCE with baseline | 多种子曲线；数值梯度/解析梯度小例验证 |
| PPO/GRPO/DAPO | 四模型 RLHF 数据流；组内 advantage 图 | 纯 PyTorch loss 单元测试；verl 最小 GSM8K run | 手算 batch 与实现一致；mask、clipping、KL、loss aggregation 有测试 |
| Agentic RL | state/action/observation 多轮时序图 | 可 reset 的 calculator/search toy environment | 成功、失败、超时、工具异常均形成完整 trajectory；只训练 policy tokens |
| verl | `DataProto` key 演化图；worker/resource placement 图 | 固定 submodule commit 的小模型 smoke run | 保存 resolved config、版本、硬件、日志、checkpoint 路径与已知限制 |

## 12. 引用与版本纪律

- 论文结论引用 arXiv/会议页；实现细节引用官方源码或官方文档。
- 对上游源码使用 tag 或 commit SHA；本仓库 verl 基线固定为 `c4b389adadc58ce51cb2b63e70df497ca166d77f`。
- 性能表必须记录 GPU 型号/数量、互联、CUDA/驱动、模型与精度、输入/输出长度分布、并发、warm-up、batch 策略和统计分位数。
- 不把论文 abstract 的峰值加速数字拼成跨系统结论；每种优化可能改变不同瓶颈，也可能互相冲突。
- 任何 reward、judge 或 benchmark 都需要记录版本和失败案例；高 reward 不自动等价于真实能力或安全性提升。
