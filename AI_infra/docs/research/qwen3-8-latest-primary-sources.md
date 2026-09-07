# Qwen3.8 最新进展：面向 Qwen3.5 读者的一手资料核验

> 核验日期：2026-09-07（Asia/Singapore）
> 来源范围：Qwen 官方 GitHub、Qwen 官方 Hugging Face、Qwen/QwenCloud 官方文档与正式技术报告。本文刻意不采用媒体转述、社区跑分和第三方推测。

## 结论先行

1. **Qwen3.8 是正式产品与开源模型系列，不是 `Qwen3-8B` 的误写。** Qwen 官方仓库明确把 Qwen3.8 列为 Qwen3.5 开源模型大系列的最新一代，并列出了官方开源权重与发布时间。[Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)
2. 截至核验日，Qwen3.8 的两条主线要分开理解：
   - **能力旗舰线**：托管版 `qwen3.8-max`，以及开源的 `Qwen3.8-2.4T-A95B`、`Qwen3.8-27B`。重点是编码、办公/专业工作、研究和长程 Agent 执行能力。[Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)
   - **下一代效率架构线**：托管版 `qwen3.8-flash` 与开源 `Qwen3.8-Flash-Next`。后者是 Qwen4 架构的实验预览，引入 QSA、Gated Residual、N-gram Embedding 和 Muon 训练配方。[Qwen3.8-Flash-Next 官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)
3. **从 Qwen3.5 到 Qwen3.8-27B，并不是可见网络结构的大改版。** 两个官方模型卡列出的语言模型规格逐项相同：27B、hidden size 5120、64 层、`3 × Gated DeltaNet + 1 × Gated Attention` 的重复布局、相同 head 数、FFN 宽度和 MTP；因此可验证的增量主要落在新权重和后训练后的能力/行为，而非公开结构参数。[Qwen3.5-27B 模型卡](https://huggingface.co/Qwen/Qwen3.5-27B)；[Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
4. 真正的公开结构跃迁出现在 **Qwen3.8-Flash-Next**：它保留 GDN 的“压缩记忆”，把周期性的稠密 Gated Attention 换成 Qwen Sparse Attention（QSA）的“按 micro-block 检索”，同时重做残差流、embedding 扩容方式和优化器配方。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[官方技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

## 1. 官方发布时间线

| 日期 | 正式进展 | 可验证含义 |
|---|---|---|
| 2026-02-16 | Qwen3.5 首发 `397B-A17B` | 建立原生多模态、GDN + 稀疏 MoE、规模化 RL 与异步 Agent 环境训练底座。[Qwen3.5 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| 2026-04-16 / 04-22 | Qwen3.6 `35B-A3B` / `27B` | 官方定位是加强稳定性、真实编码体验与 thinking preservation；27B 的公开网络规格仍与 3.5-27B 相同。[Qwen3.8 仓库时间线](https://github.com/QwenLM/Qwen3.8)；[Qwen3.6-27B 模型卡](https://huggingface.co/Qwen/Qwen3.6-27B) |
| 2026-08-02 | `Qwen3.8-Max` 发布，宣布 Max 级模型首次开放权重 | 官方博客摘要称模型规模为 2.4T，开源权重随后发布。[Qwen 官方博客](https://qwen.ai/blog?id=qwen3.8) |
| 2026-08-12 | `Qwen3.8-2.4T-A95B` 开源 | 2.4T 总参数、95B 激活；开源权重本身是文本 Causal LM。[官方仓库](https://github.com/QwenLM/Qwen3.8)；[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) |
| 2026-08-14 | `Qwen3.8-27B` 开源 | 27B dense、原生图像/视频理解，强调部署友好。[官方仓库](https://github.com/QwenLM/Qwen3.8)；[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B) |
| 2026-08-26 | `qwen3.8-flash` 与开源 `Qwen3.8-Flash-Next` | Qwen4 架构预览；QwenCloud 将托管版定位为高速、低成本、多模态和 1M 上下文模型。[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)；[QwenCloud 发布日志](https://docs.qwencloud.com/changelog/models) |
| 2026-09-02 | `qwen3.8-max-0902` 快照 | 官方称其进一步强化工程级编码、长程自主开发、多工具协作与视觉理解，同时保留 1M 上下文、thinking 和工具生态。[QwenCloud 发布日志](https://docs.qwencloud.com/changelog/models) |

注意：`Qwen3.7-Max/Plus` 出现在 3.6 与 3.8 之间，但官方 Qwen3.8 仓库仍把开源谱系描述为 Qwen3.5、Qwen3.6、Qwen3.8；不要仅凭小数版本号推断每一代必然对应一次底层架构换代。[Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)

## 2. 当前正式型号与边界

| 型号 | 开放方式 | 结构/参数 | 多模态 | 上下文与定位 |
|---|---|---|---|---|
| `qwen3.8-max` / `qwen3.8-max-0902` | QwenCloud 托管 | 官方说明基于 2.4T MoE Max 系列；开源对应物是 `2.4T-A95B`，但两者功能不完全相等 | 文本、图像、视频输入 | 1M context、128K max output、256K thinking budget、函数调用/内置工具/结构化输出。[QwenCloud 模型表](https://docs.qwencloud.com/developer-guides/getting-started/text-generation-models)；[视觉模型表](https://docs.qwencloud.com/developer-guides/getting-started/vision-models) |
| `Qwen3.8-2.4T-A95B` | HF / ModelScope 开源权重 | 2.4T total、95B active、92 层、512 experts、每 token 激活 `10 routed + 1 shared` | **模型卡标为纯文本 Causal LM**；托管 Max 额外提供视觉输入和 non-thinking 等生产功能 | 原生 262,144，官方给出扩展至约 1.01M 的配置。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) |
| `Qwen3.8-27B` | HF / ModelScope 开源权重 | 27B dense、64 层，公开语言骨干规格与 3.5/3.6-27B 相同 | 原生图像和视频理解 | 原生 262,144，可用 YaRN 扩展至约 1M。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B) |
| `qwen3.8-flash` | QwenCloud 托管 | 基于 `Qwen3.8-Flash-Next`，带生产增强 | 文本、图像、视频 | 1M context、128K max output、262K thinking；官方报告 1M prefill 相对对照实现可达 7.6× 加速。[QwenCloud 最新模型页](https://docs.qwencloud.com/developer-guides/getting-started/latest-model) |
| `Qwen3.8-Flash-Next` | HF / ModelScope 开源权重 | 125B 主模型/6B active + 51B n-gram embedding + 4B MTP；HF 总体显示约 180B | 原生视觉编码器 | 开源模型原生 262,144；它是实验性的 Qwen4 架构预览，不应与普通 `Qwen3.8-Flash` 产品能力完全画等号。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |

### 最容易混淆的三组名字

- `Qwen3-8B`：Qwen3 系列中的 **8B 参数模型**，不是 Qwen3.8。
- `Qwen3.8-2.4T-A95B`：开源权重，模型卡标为文本模型；`Qwen3.8-Max` 是以它为基础、增加视觉输入、1M 默认上下文、non-thinking 和内置工具等能力的官方托管版本。[Qwen3.8-2.4T-A95B 模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
- `Qwen3.8-Flash-Next`：开源架构预览；`qwen3.8-flash` 是基于它的官方托管版本，具有默认 1M 上下文和内置工具等生产功能。[Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

## 3. 从 Qwen3.5 看 Qwen3.8：哪些没变，哪些真的变了

### 3.1 27B：公开骨架没变，能力与交互协议变了

Qwen3.5-27B、Qwen3.6-27B 和 Qwen3.8-27B 的官方模型卡列出完全相同的语言骨干：

```text
27B dense
hidden_size = 5120
layers = 64
layout = 16 × [3 × (Gated DeltaNet → FFN)
               + 1 × (Gated Attention → FFN)]
GDN heads = 48(V) / 16(QK), head_dim = 128
Attention heads = 24(Q) / 4(KV), head_dim = 256
FFN intermediate = 17,408
MTP = multi-step trained
native context = 262,144
```

逐项依据：[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)、[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)、[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。

因此，不能把 Qwen3.8-27B 的提升讲成“换了一套 Transformer 结构”。官方可验证的新重点是：

- 更强的 coding、professional work、research、长程 agent 任务与环境反馈处理；这是官方能力定位，并由模型卡中的官方评测支持。[Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
- thinking 默认开启，但可逐请求关闭；增加 `reasoning_effort` 控制推理深度，并通过 `preserve_thinking` 保留历史推理上下文。[Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
- 对常用 agent harness 与开发工具的兼容性被官方列为独立增强项。[Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)

在同一张官方 27B 评测表中，3.8 相对 3.6 的示例变化包括 Terminal Bench 2.1 `63.4 → 73.0`、SWE-bench Pro `53.5 → 61.7`、CoWorkBench `61.0 → 70.7`、IFBench `69.1 → 79.5`。这些是**官方自报结果**，部分测试使用 Claude Code harness 或 Qwen 内部 benchmark，适合证明官方训练目标的方向，不应当作完全独立的横向审计。[Qwen3.8-27B 模型卡及评测说明](https://huggingface.co/Qwen/Qwen3.8-27B)

### 3.2 Max 开源模型：在 3.5 混合骨架上纵向扩容

Qwen3.5-397B-A17B 与 Qwen3.8-2.4T-A95B 都采用 `3 × GDN + 1 × Gated Attention` 的周期和 512 个 experts、`10 routed + 1 shared` 激活模式；3.8 把层数从 60 提到 92、hidden size 从 4096 提到 8192、expert intermediate 从 1024 提到 2048，激活参数从 17B 提到 95B。[Qwen3.5-397B-A17B 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)；[Qwen3.8-2.4T-A95B 模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)

可以把它理解为：**3.5 已经完成“混合线性注意力 + 稀疏 MoE”的结构选择，3.8 Max 主要把这条路线扩到 Max 级容量，并围绕 agent 完成度继续后训练。** 这是由公开规格和官方定位共同支持的解释；官方尚未公开足够信息把每项能力增益精确归因到某个训练阶段。

### 3.3 Flash-Next：真正面向下一代的四项结构变化

1. **GDN + QSA**：GDN 负责低成本压缩历史；QSA 的轻量 indexer 以 micro-block 为单位选出重要上下文，避免每层都对完整长上下文做稠密注意力。官方模型规格给出的选择预算为 512 blocks / 2048 tokens。[Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
2. **Gated Residual（GR）**：把 residual stream 扩成 4 条分支，用数据依赖的逐元素 read gate 与每分支 scalar write gate 控制信息读写，以增强跨层表达并保持训练稳定。[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)；[技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
3. **N-gram Embedding**：用局部 bigram/trigram 索引大表，新增约 51B 参数；表可卸载到 host memory，并通过异步预取与模型计算重叠。它是在不按比例增加主干 FLOPs 的前提下扩大容量。[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)
4. **Muon + AdamW 分工**：不同权重类别采用 Muon 或 AdamW；官方还称重新拟合 scaling law、取消传统 batch-size warmup，直接从目标 batch size 开始以减少 optimizer steps。[Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[官方博客](https://qwen.ai/blog?id=qwen3.8-flash-next)

官方称 Flash-Next 的训练成本约为 Qwen3.7-Plus 的 `1/9`，并在 1M-token prefill 上达到 `7.6×` 加速；这些属于 Qwen 官方对特定对照与环境的报告，写作时应保留“官方报告/官方测得”的限定语。[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)；[QwenCloud 最新模型页](https://docs.qwencloud.com/developer-guides/getting-started/latest-model)

## 4. Thinking 与 Agent 接口：3.8 最值得工程关注的变化

Qwen3.8 把“推理深度”从单纯开关扩展为分级控制。QwenCloud 对 `qwen3.8-max` 定义 `reasoning_effort = low | medium | xhigh`，分别映射到约 4096、16384、262144 的 thinking budget；默认等级是 `xhigh`，但默认 budget 文档同时给出 131072。`reasoning_effort` 与 `thinking_budget` 不能同时设置。[QwenCloud Chat API](https://docs.qwencloud.com/api-reference/chat/dashscope)

`preserve_thinking=true` 会把历史 assistant 的 `reasoning_content` 再次送入模型；在 `qwen3.8-max` 和 `qwen3.8-flash` 上默认开启。调用方必须完整回传历史 reasoning content，它会计入输入 token 与费用。[QwenCloud Chat API](https://docs.qwencloud.com/api-reference/chat/dashscope)；[QwenCloud thinking 指南](https://docs.qwencloud.com/developer-guides/text-generation/thinking)

这带来一个对 Qwen3.5 用户很实际的迁移提醒：**多轮 Agent 不再只需要保存 `content` 与 `tool_calls`，还应把 `reasoning_content` 当成协议状态的一部分。** 否则可能丢失决策上下文、重复推理，并降低工具调用准确性。[Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

官方还提醒：在多轮 agent 任务里，较低 reasoning effort 不保证端到端更快，因为分析不足可能产生更多失败和重试；评估应看“完成一次任务的总 tokens/总时间”，不能只看单轮首 token 或单轮输出长度。[Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

## 5. 推理与部署变化

- 官方示例继续支持 SGLang、vLLM 和 TokenSpeed；Qwen3.8-27B 的标准启动示例使用 262,144 context、`qwen3` reasoning parser 和 `qwen3_coder` tool-call parser。[Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)
- 开源 27B/2.4T 的 262K 是**原生上下文**，接近 1M 需要 YaRN 等 RoPE scaling；模型卡明确警告多数开源框架实现的是 static YaRN，长上下文 scaling 会影响短文本表现，因此只在确实需要时启用，并按目标长度选 factor。[Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
- 托管版 Max/Flash 默认提供 1M context、函数调用、结构化输出和内置工具；这些是服务层能力，不应直接算作开源 checkpoint 的网络结构能力。[QwenCloud 模型表](https://docs.qwencloud.com/developer-guides/getting-started/text-generation-models)
- QwenCloud 对 Qwen3.8 开源模型、Max 和 Flash 支持 implicit context cache；满足共同前缀至少 1024 tokens 只是具备写入/命中的技术条件，实际命中并不保证。[QwenCloud Context Cache](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache)

## 6. 当前不能下结论的地方

以下内容截至核验日没有在所检查的一手资料中得到足够披露，文章不应把它们写成确定事实：

- Qwen3.8-27B 相对 3.5/3.6 的具体继续预训练 token 数、数据混合比例、数据截止日期。
- Qwen3.8 Max/27B 每个后训练阶段具体采用了 PPO、GRPO、DAPO 或其他哪一种 RL 算法，以及各自占比。
- 27B 是否由 2.4T 模型蒸馏而来；官方材料没有明确给出该因果链。
- 官方 benchmark 的全部 prompts、内部 benchmark 数据与所有竞争模型的完全同条件复现结果。
- `qwen3.8-max-0902` 的底层权重/结构是否变化；官方发布日志只描述能力升级，没有给出结构 diff。[QwenCloud 发布日志](https://docs.qwencloud.com/changelog/models)

## 7. 给主文章与指导图的建议

### 图 1：版本路线，不要画成单一升级箭头

```text
Qwen3.5（GDN + Attention + MoE，多模态底座）
    ├── Qwen3.6：同骨架，稳定性 / coding / thinking preservation
    ├── Qwen3.8 Max & 27B：同代骨架扩容 + agent 后训练增强
    └── Qwen3.8-Flash-Next：QSA + GR + N-gram + Muon → Qwen4 preview
```

### 图 2：27B 的“结构不变、行为升级”

左侧并排放 Qwen3.5/3.6/3.8-27B 相同的 64 层骨架；中间用等号标出公开 architecture config 一致；右侧只把可证实的增量放在 post-training / interface 层：coding、long-horizon agent、`reasoning_effort`、`preserve_thinking`、harness compatibility。

### 图 3：Flash-Next 的“记忆—检索—容量—优化”

```text
GDN：压缩历史记忆
  + QSA：从 micro-block 中稀疏检索
  + GR：4 路残差流的动态读写门
  + N-gram：低计算成本的容量扩张
  + Muon/AdamW：按权重类型分工训练
```

### 图 4：部署选择树

```text
最强复杂 Agent / 工程编码 → qwen3.8-max(-0902)
高并发、成本敏感、多模态 → qwen3.8-flash
可本地部署、单机/多卡实验 → Qwen3.8-27B
研究 Max 级开放权重 → Qwen3.8-2.4T-A95B
研究下一代稀疏注意力架构 → Qwen3.8-Flash-Next
```

## 8. 推荐优先阅读的一手资料

1. [Qwen3.8 官方 GitHub 总览](https://github.com/QwenLM/Qwen3.8)
2. [Qwen3.8-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
3. [Qwen3.8-2.4T-A95B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
4. [Qwen3.8-Flash-Next 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
5. [Qwen3.8-Flash-Next 正式技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
6. [QwenCloud 模型发布日志](https://docs.qwencloud.com/changelog/models)
7. [QwenCloud Qwen3.8-Flash 最新模型指南](https://docs.qwencloud.com/developer-guides/getting-started/latest-model)
8. [QwenCloud Thinking 参数文档](https://docs.qwencloud.com/developer-guides/text-generation/thinking)
