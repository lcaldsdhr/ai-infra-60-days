# Qwen3.8-Flash-Next 深度核验：配置、架构、训练与工程代价

> 核验日期：2026-09-08（Asia/Singapore）
>
> 只采用 Qwen 官方 GitHub、官方 Hugging Face、官方博客、QwenCloud 文档与 Qwen 团队技术报告。社区实测、媒体文章和第三方推测均不作为事实来源。

## 一句话结论

`Qwen3.8-Flash-Next` 不是 Qwen3.8-27B 的轻量量化版，而是一条实验性的新架构分支：用 125B 稀疏 MoE 主模型（每 token 激活 6B）、51B 可卸载 N-gram embedding 和 4B MTP 组成约 180B 的完整 checkpoint，并引入 QSA、四路 Gated Residual 与 Muon 训练配方。Qwen 官方明确称它是未来 Qwen4 架构的预览。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[官方技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

## 1. 参数口径：125B、6B、+51B、+4B 分别是什么

```text
完整 checkpoint 约 180B
├── 主模型 125B
│   └── MoE 路由后每 token 激活约 6B
├── N-gram embedding 51B
│   └── 稀疏查表；官方方案放在 accelerator 外并异步预取
└── MTP 4B
    └── 1 层预测模块；用于多步预测/推测解码
```

官方模型卡的原文口径是“125B with 6B activated, plus 51B n-gram embedding and 4B MTP”，Hugging Face 页面显示的模型总规模约为 180B，二者正好对应 `125 + 51 + 4 = 180B`。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

需要避免三个误读：

- **6B 不是 checkpoint 大小。** 它描述主干稀疏 MoE 在一个 token 上参与计算的参数量级；权重仍需存储、分片或卸载。[官方技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
- **51B 不是每个 token 都做一次 51B dense matmul。** N-gram 表通过确定性地址稀疏查找，官方把它作为“以很少额外 FLOPs 扩容量”的路径，并设计为 host-memory offload + asynchronous prefetch。[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- **4B MTP 不是主模型 48 层中的额外 4B 激活。** 配置中 `mtp_num_hidden_layers=1`，模型卡也写明 MTP 为 1 层；它服务于多 token 预测，是否在推理中付出这部分成本取决于是否启用相应 speculative/MTP 路径。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

相对 Qwen3.5-397B-A17B，Flash-Next 的主模型总参数从 397B 降到 125B，激活参数从 17B 降到 6B；但它额外引入 51B N-gram 表和 4B MTP，因此比较存储、训练 FLOPs、单 token 计算量时必须分别使用 total、activated、offloaded-table、draft-module 四种口径。[Qwen3.5-397B-A17B 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)；[Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

## 2. 完整语言模型配置逐项核对

| 模块 | Flash-Next 官方配置 | 字段含义与工程影响 | 相对 Qwen3.5-397B-A17B |
|---|---|---|---|
| 模型类型 | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` | Transformers 需要识别新的实验架构；不能假定旧 Qwen3.5 runtime 无修改即可加载 | Qwen3.5 是 `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`。[Flash 配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[3.5 配置](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) |
| 层数与宽度 | 48 层，`hidden_size=2560` | 更窄、更少层可降低 dense 部分和激活内存，但不能仅靠这两个数估算 MoE/GR/QSA 总成本 | Qwen3.5 为 60 层、hidden 4096。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) |
| 层布局 | `12 × [3 × (GDN → MoE) + 1 × (QSA → MoE)]` | 36 个 GDN 层与 12 个 QSA 层；全局检索并未消失，而是每四层出现一次稀疏全局检索 | Qwen3.5 是相同的 3:1 节拍，但第四层为 Gated Attention，不是 QSA。[Flash 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| GDN | V heads 48，QK heads 16，head dim 128，short-conv kernel 4 | GDN 把历史压入固定尺寸 recurrent state，长序列 token mixing 近线性；状态更新仍需专用 fused kernel | Qwen3.5 旗舰为 V heads 64、QK heads 16、dim 128、kernel 4。[Flash 配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[3.5 配置](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) |
| QSA core attention | Q heads 24、KV heads 2、head dim 256、RoPE dim 64 | 这是 GQA 形式的稀疏核心注意力；KV heads 少于 Q heads，可降低 KV 存储 | Qwen3.5 Gated Attention 为 Q heads 32、KV heads 2、dim 256、RoPE dim 64，但对全上下文做稠密注意力。[Flash 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| QSA indexer | MQA：4 Q heads + 1 shared K head，indexer dim 128 | 独立轻量索引器先为 micro-block 打分，再让 core attention 读取选中的 token；新增索引计算、top-k 与稀疏 gather | Qwen3.5 没有 QSA indexer。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| QSA budget | `indexer_compress_ratio=4`，`indexer_budget=2048`，即最多 512 个完整 blocks | 每个压缩 block 代表 4 tokens；选中 512 blocks 后展开为最多 2048 tokens，当前未完成尾块额外保留 | Qwen3.5 的周期性注意力层访问全部可见上下文，不存在该固定检索预算。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[技术报告 §2.1.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) |
| MoE | 512 experts，`top-10 routed + 1 shared`，expert intermediate 640 | 512 个专家仍需存储/分片；每 token 只算 10 个 routed experts 加 shared expert，通信拓扑仍是主要工程成本 | Qwen3.5 也是 512、top-10 + shared，但 expert intermediate 为 1024，因此其总量和激活量更大。[Flash 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| GR | `hc_count=4`、`hc_lowrank=320` | residual state 扩成 4 个分支；read gate 是逐通道动态门，write gate 是每分支标量；rank 320 等于 `hidden_size / 8` | Qwen3.5 是单 residual stream，没有 GR 的四路状态与低秩门。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[技术报告 §2.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) |
| N-gram | `ngram_vocab_size_base=20,000,000`、`ngram_size=3`、layer id 2、bigram/trigram | 用截至当前 token 的局部二元/三元组确定查表地址，并在第 2 层注入；大表适合从 host 异步预取 | Qwen3.5 没有 N-gram embedding。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| MTP | 1 hidden layer，`hybrid=true`，不使用 dedicated embeddings，约 4B 参数 | 可生成多个 draft token；QSA top-k indices 可跨多个预测步复用以降低 draft 成本 | Qwen3.5 也有 1 层、多步训练的 MTP，但 Flash-Next 将 MTP attention 改为 QSA，并明确复用 indices。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[技术报告 §2.1.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) |
| 词表/输出 | vocab 与 LM output 均为 padded 248,320，untied embeddings | 与旧模型的 tokenizer/output 宽度对齐有助生态兼容，但新架构和 chat template 仍需 runtime 支持 | Qwen3.5 同为 248,320 且不 tied。[Flash 配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[3.5 配置](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) |
| 位置与 context | native 262,144，`rope_theta=10,000,000`，partial rotary factor 0.25；可扩展至 1,000,000 | 开源 checkpoint 的 1M 不是原生窗口，需要 RoPE scaling/YaRN；短上下文可能受 static scaling 影响 | Qwen3.5 旗舰也是 native 262,144、可扩至约 1.01M，因而“1M”不是 Flash-Next 独有结构增量。[Flash 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |

配置中还出现 `heads_per_ngram=8`、`split_ngram_parts=128`、`ple_*` 等实现字段。官方模型卡与技术报告没有逐字段给出稳定的公共语义契约；可以在源码级实现分析中引用它们，但不宜仅凭字段名推断“8 张表”或“128 路必然对应某种并行度”。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)

## 3. QSA：不是直接把全注意力砍成固定窗口

### 3.1 完整读取路径

```text
当前 query token
    │
    ├─ indexer Q：4 heads × 128d
    │
历史 K ── 每 4 token AvgPool ── block key（1 shared K head）
    │
    ├─ partial RoPE：128d 中 64d 旋转
    ├─ ReLU(q·k)，跨 4 个 indexer heads 求和
    ├─ causal mask：只给已完整出现的 blocks 打分
    ├─ TopK 512 blocks
    └─ 展开成最多 2048 tokens + 当前尾部未完成 block
                              │
                         sparse core GQA
```

技术报告明确说明先对连续 4-token key 做 average pooling、再施加 block-level partial RoPE；这样避免直接平均不同旋转相位。每个 query 的 block 分数来自 4 个 indexer heads 的 ReLU 相似度之和，且只允许打分已经完整可见的 block。[技术报告公式 12–16](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

这与 sliding-window attention 不同：窗口只保留“最近”，QSA 可以从完整历史中选“相关”；它也与 token-level sparse indexer 不同：先把 key sequence 压缩 4 倍，再做索引，官方给出的 indexer 复杂度由 `O(n²)` 降为 `O(n²/r)`，此处 `r=4`。核心稀疏 attention 对每个 query 只处理最多 2048 个被选 token，但 indexer 本身仍随上下文增长，并不是完全 `O(n)`。[技术报告 §2.1.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

### 3.2 QSA 不是训练后直接硬裁剪

官方采用两阶段继续预训练：

1. **Dense distillation**：从全注意力 teacher 的 token-level attention distribution 构造 block teacher；用 max pooling 保留块内显著信号，再以 KL loss 训练 indexer。仅训练 indexer 1,000 steps，每步 8 条 256K 序列，合计约 2B tokens。[技术报告 §2.1.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
2. **Sparse training**：indexer top-k 真正控制 backbone 的稀疏 attention，backbone 与 indexer 联合适应 8,000 steps，每步 96 条 256K 序列，约 200B tokens；KL 只在已选择 blocks 上重新归一化计算。[技术报告 §2.1.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

官方实验显示，完成 dense initialization 后如果直接稀疏化会明显掉点，短暂 joint training 才恢复到 full-attention 水平。这意味着把 Qwen3.5 checkpoint 的 attention kernel 简单替换为 top-k sparse kernel，不能等价得到 Flash-Next。[技术报告 Figure 5](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

### 3.3 QSA 的真实性能边界

官方 kernel-level 测试包含 indexer 和 sparse core attention：1M context 时，QSA 相对 FlashInfer paged GQA 在 attention module 上达到 7.6× prefill 与 4.9× decode 加速。测试设置是 chunked prefill 的最后 16K chunk、batch size 1；decode 为 batch size 4、`next_n=4`。这是模块级、特定硬件/内核/批量下的结果，不能直接解释为整台服务吞吐提升 7.6×。[技术报告 Figure 6](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

QSA 的实际代价包括：轻量 indexer、top-k、稀疏地址 gather、专用 fused kernel、约 200B-token 稀疏适应训练，以及固定 2048-token budget 可能带来的召回上限。官方通过 RULER/MRCR 和通用 benchmark 验证其 checkpoint，但没有证明所有任务、所有超过 1M 的上下文或所有部署框架都无损。[技术报告 Tables 2–3](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

## 4. Gated Residual：四条信息高速公路，但要付内存流量

### 4.1 读写机制

每个 residual state 从单个 `d=2560` 向量扩为 `n_r=4` 个分支。每个 attention block 和 MLP block 都有独立 GR：先对各分支做 RMSNorm，再通过 rank `r=d/8=320` 的低秩投影预测逐分支、逐通道 sigmoid read gate，把 4 个分支加权平均成 block 输入；block 输出再通过每分支一个动态标量写回所有分支。[技术报告公式 30–34](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

与 Hyper-Connections 相比，GR 删除了分支间 `H_res` mixing matrix，把表达能力集中在 element-wise read gate 上。官方称这样少一次完整 residual state 读取，并避免为 `H_res` 额外施加稳定性约束。[技术报告 §2.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

### 4.2 为什么值得做

技术报告的路径分解显示，一条分支倾向保存早层信息并跨很多层送到周期性注意力层，另外三条更偏局部路径。它不是简单“残差复制四份”，而是允许不同历史信息具有不同的保留与读取强度。[技术报告 Figure 7](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

GR 也提供显式重缩放：在高学习率 stress test 中，加入 gate 显著减少 gradient-norm spikes 与 activation outliers；完整大规模训练据官方报告没有出现 loss spike 或异常 gradient-norm 波动，且未依赖 qk-clip / SwiGLU-clip。[技术报告 §3.3](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

### 4.3 工程代价

- 四分支 residual 会增加 activation 存储和读写流量；报告明确称 GR 的推理成本由 widened residual state 的 memory traffic 主导。[技术报告 §2.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
- 官方尝试只读取 gate 最大的两个分支，预训练 loss 几乎不变，但后训练质量明显下降，因此最终仍读取全部四个分支。[技术报告 §2.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
- residual branches 可用 FP8 存储，相对 BF16 把这部分搬运字节减半且官方观察到几乎无质量损失；read、RMSNorm、write 分别融合为内核以减少遍历。[技术报告 §2.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

由数据类型宽度可以推导：四路 FP8 residual 的裸 payload 仍约为单路 BF16 residual 的两倍；这只是数据宽度层面的估算，不包含 kernel fusion、缓存命中、对齐和临时 buffer，不能替代官方实测。

## 5. N-gram Embedding：把容量放到计算主干之外

配置给出一个 `20,000,000 × 2560` 量级的逻辑 embedding 容量，对应约 51.2B 参数，与模型卡的 51B 一致。模型使用截至当前 token 的 bigram/trigram 作为确定性 key，在第 2 层注入检索向量。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)；[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

选择第 2 层不是随意的：官方 ablation 发现多个深度差距不稳定，固定参数预算下多层放置也没有一致收益；第 2 层允许 host-memory prefetch 与第 1 层计算重叠，因此最终只放一层。[技术报告 §2.3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

这条路线的工程本质是“FLOPs 换内存层级”：

- 优点：每 token 只查少量槽位，不需要像增加 MoE experts 那样进行对应规模的矩阵乘；确定性地址便于预取和卸载。[技术报告 §2.3](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
- 代价：需要约 51B 参数的 host storage、稳定的地址构造、异步预取队列和足够的 host-to-device 带宽；预取若不能被前一层计算掩盖，会直接成为 latency bottleneck。后半句是由官方 offload/prefetch 设计导出的系统约束，不是官方提供的统一延迟数值。
- 边界：报告发现 vocabulary 变大时 training loss 单调下降，但 downstream accuracy 会饱和或波动；因此“表更大”不等于“任务能力等比例增加”。[技术报告 Tables 8–9](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

若全部以 BF16 保存，51.2B 参数的理论裸数据约为 102.4 GB；若某实现安全地使用 1-byte 格式则约 51.2 GB。这里只是按每参数字节数计算的容量下界，官方未在模型卡中承诺 N-gram 表固定采用何种 host 存储格式，也未披露实际 pinned-memory、缓存和分片开销。

## 6. GDN、MoE 与 MTP：保留的结构和新的组合

### GDN

GDN 延续 Qwen3.5 的 fast-weight recurrent memory：`q/k/v` 经短因果卷积，`q/k` L2 normalize，decay gate 控制旧状态寿命，delta update 只写入 key 当前预测值与真实 value 之间的残差。它减少长序列 KV cache，但有限状态不能精确替代任意 token 级检索，所以每四层仍需 QSA。[技术报告公式 1–11](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

报告称 FlashQLA 相对 FLA Triton baseline 在 NVIDIA GPU 上带来 2–3× forward、约 2× backward 的 GDN kernel 加速；这是 GDN kernel 对比，不是整体训练速度。[FlashQLA 官方仓库](https://github.com/QwenLM/FlashQLA)；[技术报告 §2.1.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

### MoE

Flash-Next 与 Qwen3.5 旗舰都保持 512 experts、top-10 routed + 1 shared，说明优化重点不是继续提高 expert 数量，而是缩小 hidden/expert width、降低 activated parameters，并用 N-gram 表补充低 FLOPs 容量。[Flash 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[Qwen3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)

但 6B active 不意味着“只需要 6B 模型的集群”：512 experts 的 125B 主模型仍需跨设备存放，routing 仍会产生 token dispatch/combine 通信，N-gram 还引入 host memory 路径。官方没有公布通用的最小 GPU 数、EP/TP/PP 最优组合或不同硬件上的端到端带宽需求。

### MTP

MTP 是一层、约 4B 的附加预测模块，训练为多步预测。所有 MTP full-attention 也被替换为 QSA；做四步 speculative decoding 时，可以跨 prediction steps 复用同一组 QSA top-k indices。官方实验中平均 accepted length 由 full attention 的 4.06 变为 QSA 的 4.07，未观察到显著损失。[技术报告 Table 4](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

工程上，MTP 能否换来净收益取决于 draft 接受率、verification kernel、batch、上下文长度和 runtime 实现；模型卡提供了能力，但没有承诺任意服务场景都会加速。

## 7. 视觉配置：大体继承 Qwen3.5，不是本报告的主要创新点

Flash-Next 的 vision encoder 配置为 depth 27、hidden 1152、intermediate 4304、16 heads、patch size 16、spatial merge 2、temporal patch 2、position embeddings 2304、无 deepstack visual indexes；输出投影宽度为 2560，以对接新的文本 hidden size。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)

Qwen3.5-397B-A17B 的 vision encoder 除输出宽度为 4096 以外，上述公开字段相同。因此从 config 能证实的是“视觉编码器公开骨架基本延续，connector/output width 随语言主干变窄”，不能把 Flash-Next 的主要创新讲成全新视觉塔。[Qwen3.5 官方配置](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)

官方技术报告重点研究 token mixing、residual、embedding 和 optimizer，没有披露完整多模态训练数据、图像/视频 token 配方、视觉塔预训练过程或 modality mixture。模型卡展示了视觉/视频与 multimodal agent 评测，但这些结果不能反推出未公开训练细节。[Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)；[技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

## 8. Context 与服务层边界

开源 Flash-Next 原生 context 是 262,144，可通过 YaRN 等扩展到 1,000,000。模型卡提醒 static YaRN 的固定 factor 可能损伤较短文本，因此只应在真正处理超长上下文时改 RoPE 参数，并按目标长度选 factor。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

托管的 `qwen3.8-flash` 基于 Flash-Next，但默认提供 1M context、128K 最大输出、262K thinking budget、内置工具和 context cache；这些属于官方服务产品能力，不应全部归因于开源 checkpoint 本身。[QwenCloud 最新模型页](https://docs.qwencloud.com/developer-guides/getting-started/latest-model)；[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

官方仓库给出的自部署入口包括 Transformers、SGLang、vLLM 与 TokenSpeed，并在服务命令中使用 262,144 context、`qwen3` reasoning parser 和 `qwen3_coder` tool parser。由于 `model_type=qwen4_exp`、QSA、GR 和 host-offloaded N-gram 都是新路径，应优先使用官方模型卡当前建议的最新框架版本与专用 recipes。[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## 9. 训练优化器与稳定性

### 9.1 哪些权重用 Muon，哪些不用

技术报告将 Muon 用于真正充当二维线性映射的权重：attention q/k/v 与 output、GDN input/output、routed/shared expert 的 fc1/fc2、N-gram 层的 key/value projections。input embedding、LM head、MoE router、GR 的两个低秩投影保留 AdamW；N-gram embedding table 使用无 weight decay 的 Adam。[技术报告 §3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

原因不是“Muon 全面优于 AdamW”：router 各输出维度近似独立，Muon 在早期反而加剧波动；GR 投影矩阵形状过于细长；embedding table 的查表性质也不适合矩阵正交化。这个按参数语义分工是 Flash-Next 配方的一部分。[技术报告 §3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

Muon 使用 Nesterov momentum `μ=0.95`，Newton–Schulz 采用 Polar Express 系数、8 次迭代，归一化稳定项为 `1e-14`；更新缩放为 `0.2 × sqrt(max(A,B))`。QKV、SwiGLU fc1 和 GDN input 等 fused parameters 在 orthogonalization 前必须按语义子矩阵拆开，否则会混合无关 singular directions。[技术报告 §3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

### 9.2 分布式优化器代价

Muon 需要对完整矩阵操作，但 TP 后单 rank 不拥有完整矩阵；同时 Newton–Schulz FLOPs 与矩阵形状有关，按参数元素数平均切 DP 会产生 straggler。官方使用 Canzona 对完整参数按估算 NS FLOPs 静态重分配，并通过 fused All-to-All 在 TP ranks 间重建矩阵；大量拆分后的小 kernel 再用 CUDA Graph 捕获以减少 launch overhead。[技术报告 §3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

因此“Muon 更省训练”不等于“直接把 AdamW 类名替换掉”：它要求矩阵语义拆分、TP/DP 通信重组、负载平衡和 CUDA Graph 工程。

### 9.3 Scaling law 与 batch warmup

新架构与 Muon 把官方拟合的近优 learning rate 和 batch size 都推高。报告的实验中，从小 batch ramp 到目标 batch 没有改善最终结果，却为同一 token budget 增加 18.8% optimizer steps，所以 production run 不再使用 batch-size warmup。[技术报告 §3.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

报告称完整 Flash-Next base 相对 397B-A17B 的 Qwen3.7-Plus-Base，只用约 1/3 activated parameters、1/3 training tokens 和约 1/9 training FLOPs，并在 14 个 pretraining benchmarks 中领先 8 个，其余最多落后 2.6 分。这个 `1/9` 的明确对照是官方报告中的 Qwen3.7-Plus-Base/上一代 397B-A17B，而不是任意 Qwen3.5 训练 run。[技术报告 Table 11](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

## 10. 两种 FP8 必须分开

### A. FP8 checkpoint 权重量化

官方另发 `Qwen3.8-Flash-Next-FP8`，称使用 block size 128 的 fine-grained FP8 quantization，性能指标与原模型近乎相同，并支持 Transformers、vLLM、SGLang、TokenSpeed 等框架。[官方 FP8 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)

它主要减少权重存储与权重带宽，但实际显存、吞吐与可运行硬件取决于框架是否有对应 FP8 kernel、MoE 分片、N-gram offload 和 KV/activation 精度配置；官方没有给出跨硬件统一倍率。

### B. GR residual activation 的 FP8 存储

技术报告还讨论把四路 residual state 存为 FP8，以降低 widened residual 的 memory traffic。这是**中间激活/状态的数据格式设计**，与下载 FP8 权重 checkpoint 是两个维度；即便使用 BF16 checkpoint，runtime 仍可能对 residual state 采用低精度内核，反之亦然。[技术报告 §2.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

## 11. 官方没有披露或尚不能证明的事项

- 完整 pretraining token 总数、数据截止日期、语言/代码/视觉数据比例、去重与合成数据配方。
- post-training 使用的 SFT、RL、偏好优化算法及各阶段数据/算力；不能从模型能力反推采用了 GRPO、DAPO 或某个特定算法。
- 125B 主模型内部“6B activated”的逐模块精确核算表；官方给出总口径，但未逐 expert、shared expert、dense projection 分解。
- N-gram bigram/trigram 的完整 hashing、collision、table partition、cache eviction 与 host-memory 传输实现；配置字段不足以还原生产系统。
- FP8 checkpoint 对每类张量的精确量化白名单、scale 布局和所有 benchmark 的逐项 BF16/FP8 对照。
- 视觉塔的训练数据与训练过程，以及 Flash-Next 视觉能力提升可分别归因于哪些结构或数据因素。
- 各 GPU 代际、节点互联、batch/sequence mix 下的最小硬件与端到端 tokens/s；论文的 7.6×/4.9× 是特定 attention kernel 测试。
- Qwen4 最终版是否保持当前全部超参数。官方只称 Flash-Next 是 architecture preview，预览不构成最终接口承诺。[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## 12. 可直接用于主文章的图设计

### 图 A：参数不是一个数字——四种成本口径

```text
┌──────── 约 180B checkpoint ────────┐
│ 125B main MoE │ 51B N-gram │ 4B MTP │
│   ↓ top-10+shared    ↓ sparse lookup  ↓ optional draft
│ 约 6B active/token   host prefetch    speculative path
└─────────────────────────────────────┘
底栏：存储规模 ≠ 激活计算 ≠ GPU 常驻量 ≠ 每请求实际执行量
```

### 图 B：Qwen3.5 Gated Attention → Flash-Next QSA

左半：Qwen3.5 每四层一次 dense global attention，所有历史 K/V 进入 attention。右半：历史 token 每 4 个形成 micro-block，4-head MQA indexer 打分，top-512 blocks 展开为 2048 tokens，再由 24Q/2KV core attention 计算。底部做 With/Without：全检索成本高 vs 稀疏检索需训练适配且有召回预算。

### 图 C：GR 四分支读写

画 4 条平行 residual rails；每个 sublayer 前画 `RMSNorm → element-wise sigmoid read gate → average`，输出后画 4 个 scalar write gates。把“一条长期支路通往后续 QSA、三条偏局部”的官方观察用线宽体现；旁注 `rank=320`、`FP8 state`、`remove H_res`。

### 图 D：N-gram host prefetch 时间线

```text
GPU: Layer 1 compute ───────────────► Layer 2 add N-gram vector
CPU: hash bigram/trigram → table lookup → async H2D ───────▲
目标：传输藏在 Layer 1 计算后面
失败：prefetch 未完成 → Layer 2 stall
```

### 图 E：训练从“一个 optimizer”变成按参数语义路由

Muon 路径：QKV、GDN projections、expert fc1/fc2、N-gram projections → split fused tensors → TP All-to-All reconstruct → NS×8 → CUDA Graph。AdamW/Adam 路径：embedding、LM head、router、GR low-rank、N-gram table。底栏注明“不做 batch-size warmup，官方实验减少 18.8% steps”。

### 图 F：两种 FP8

上下双泳道：上层 `BF16 weights → block-128 FP8 checkpoint`，下层 `4× BF16 residual → 4× FP8 residual`；中间醒目标注“权重量化”和“激活/状态存储”相互独立。

## 13. 一手资料索引

1. [Qwen3.8-Flash-Next 官方 GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next)
2. [Qwen 团队正式技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
3. [BF16 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
4. [BF16 官方 config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
5. [官方 generation_config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/generation_config.json)
6. [FP8 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
7. [官方发布博客](https://qwen.ai/blog?id=qwen3.8-flash-next)
8. [QwenCloud 托管 Qwen3.8-Flash 指南](https://docs.qwencloud.com/developer-guides/getting-started/latest-model)
9. [Qwen3.5-397B-A17B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
10. [Qwen3.5-397B-A17B 官方 config.json](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)
