# Qwen3.5 的创新和网络结构

> **本篇是总览 + 速查表**。要深入理解 GDN 请看 [博客 1《GDN 深度：从 Mamba 2 到 GDN》](gdn-deep-dive.md)；要看其余 4 个创新的详细数据请看 [博客 2《Qwen3.5 其余创新》](qwen3.5-remaining-innovations.md)。

---

## 速览

5 大创新速查：

| # | 创新 | 核心数据 | 详见 |
|---|------|---------|------|
| 1 | **GDN** | 3:1 混合 / 8.6x 提速 / d² 替代 L² | [博客 1 § 6](gdn-deep-dive.md) |
| 2 | **极致稀疏 MoE** | 397B 总 / 17B 激活 / < 5% 激活率 | [博客 2 § 2](qwen3.5-remaining-innovations.md) |
| 3 | **Early Fusion 原生多模态** | 27 层 ViT → 49 token → 60 层 LLM | [博客 2 § 3](qwen3.5-remaining-innovations.md) |
| 4 | **mRoPE 3 维位置编码** | [11,11,10] 切片 / partial 0.25 | [博客 2 § 4](qwen3.5-remaining-innovations.md) |
| 5 | **MTP 多 Token 预测** | K=1 头 / 复用主 embedding | [博客 2 § 5](qwen3.5-remaining-innovations.md) |

## 简介

> Qwen3.5 是阿里通义千问团队 2026 年 2 月发布的开源大模型家族，包含 0.8B 到 397B-A17B 共 8 个尺寸。最大亮点是混合注意力架构：**3 层 GDN（线性 attention）+ 1 层 GA（标准 attention 升级版）= 3:1 比例**。这一设计让 Qwen3.5 在 32K 上下文下的解码速度达到 Qwen3-Max 的 8.6 倍，256K 下是 19 倍。本文将逐个拆解 Qwen3.5 的核心创新点和完整网络结构。

---

## 创新点 1：GDN

### GDN 是什么

> **GDN = Mamba2 门控 + Delta Rule**

### GA 和 GDN 的组成

> **Qwen3.5 的 attention = 3 层 GDN + 1 层 GA**（3:1 比例混合）

其中 GA 是标准 Attention 的升级版 = 标准 Attention + 3 个小补丁：

- **Output Gate（输出门）**：相当于"调光开关"，模型自己学每个 token 的 attention 该开多强（0~1 之间），重要的开大些、不重要的开小些。解决"注意力沉没"问题
- **QK Norm**：算 attention 之前先归一化 Q、K，防止数值爆炸导致训练不稳定
- **Partial RoPE**：位置编码只应用到 25% 维度（partial_rotary_factor=0.25），节省 75% 位置编码计算

GDN 是另一种 attention（线性 attention），用状态递推替代了 Q·K^T 矩阵乘法，复杂度从 O(L²·d) 降到 O(L·d²)，适合长序列和长期模式。

### 为什么需要 GDN

普通 attention 的核心问题是 **L² 复杂度**。每对 (Q_i, K_j) 都要算相似度，L 个 token 两两组合就是 L² 次。当上下文达到 128K 时就是 160 亿次，1M 时是 1 万亿次——单卡根本装不下。

GDN 用一个"状态矩阵 S"代替"全量历史相似度矩阵"。S 的大小是 d×d（d 是 hidden size，Qwen3.5 397B-A17B 是 4096），**跟序列长度 L 无关**。这样每来一个新 token，只需要更新 d×d 的小矩阵（增量更新），不用重算整段历史。

这是 GDN 最重要的洞察：**把"对 L 算 L² 次"转化为"对 d 算 d² 次"**。d 是模型架构决定的常数，所以对 L 是线性的。

### GDN 的工作原理

GDN 的核心运算是"先 K·V 再 Q"的顺序，跟普通 attention 颠倒：

```
S = K · V^T      ← 第一步：建立 K-V 关联表（大小 d×d）
O = Q · S        ← 第二步：用 Q 一次性查表
```

这利用了矩阵乘法的结合律：普通 attention 算 (Q·K^T)·V 必须先形成 L×L 矩阵，而 GDN 算 Q·(K^T·V) 先形成 d×d 矩阵。这样 S 的大小跟 L 无关，只跟 d 相关。

**两个门控机制**让 GDN 能学会怎么用状态矩阵 S：

**α（衰减门控）**：控制历史信息的遗忘速度。α = exp(-exp(A_log))，A_log 初始化为 0，所以 α 约等于 0.37，含义是"每个时间步保留 37% 的历史"。α 接近 0 表示快速遗忘，接近 1 表示长期记忆，模型自己学每个头（head）应该用多大的 α。

**β（输入门控）**：控制当前 token 多少进 S。β = sigmoid(linear(x))，linear 初始化为 0，所以 β 等于 0.5，含义是"每个 token 一半进 S、一半不进"。β 接近 0 表示当前 token 完全不进状态，接近 1 表示完全进状态。

**官方推荐的初始值**（来自 Qwen3.5 配置文件）：A_log = 0（α ≈ 0.37），dt_bias = 0（β 范围 [0, 1]），不需要 warm-up。模型通过训练自己学习 α 和 β 的最优组合，代码层面两者独立，但训练中会自动学到协同（α 小的时候 β 大，反之亦然）。实际训练中 α 和 β 的稳定区间都在 [0.1, 0.9]，极端值会导致训练不稳定。

**d 的含义**：d 是 hidden size，代表模型的"信息容量"或"抽象能力"。d 越大，每个 token 的 embedding 维度越高，能表达的语义越丰富。S 的大小 = d²（Qwen3.5 397B-A17B 的 S 是 4096×4096 ≈ 16M 元素，跟 128K 长度的 L² = 16G 元素相比，节省了 1000 倍存储）。

### 直观理解：S 是查询指南

经过前面三节，S、K、V、Q 的符号你大概熟了，但 d×d 怎么装得下 L 大小的信息，还是容易绕。
这里用三个递进的视角把 GDN 的"压缩笔记本质"讲清楚。

#### 视角 1：S 不是档案库，是聚合统计量

S 存的不是"每个 token 长什么样"，而是**两两之间的关联强度**。
用现实类比：
- **KV Cache** 像录音机——逐字完整记录 L 个 token
- **GDN 的 S** 像速记本——只记"谁跟谁关联强"这个模式

GDN 的 S 是 lossy 压缩，但足够用来做 attention——因为 attention 真正需要的就是关联模式。

#### 视角 2：d² 是流动的关联池，不是静态档案库

回到核心公式：

```
S_t = α·S_{t-1} + β·(k_t ⊗ v_t)   ← 增量更新 + 衰减
o_t = q_t · S_t                    ← Q 查表拿走关联
```

S 永远只装 d² 个数字——L 再长，旧关联也会被 α 自动稀释/覆盖。
对比数字（Qwen3.5 397B-A17B，d=4096）：
- KV Cache：L × d = 128K × 4096 ≈ 5 亿个浮点数
- GDN 的 S：d² = 4096² ≈ 1600 万个浮点数
- **差距约 30 倍**

为什么这种压缩"够用"？因为 attention 真正需要的是"谁跟谁相关"，不是"每个 token 的完整坐标"。

#### 视角 3：α 和 β 是"记多久、记多深"的可学习开关

α（衰减门）和 β（输入门）都是可学习的，模型自己决定：

| 项 | 作用 |
|---|---|
| α·S_{t-1} | 旧关联保留比例（α 接近 1 = 长期记忆，接近 0 = 短时记忆） |
| β·(k_t ⊗ v_t) | 新关联注入强度（β 接近 1 = 完全进 S，接近 0 = 完全不进） |

**α 小的时候 β 大，反之亦然**——模型通过训练学到这种"记多久 vs 记多深"的协同。

#### 你的一句话总结（实战公式版）

> **S 是 d×d 大小的"压缩笔记"**（存关联模式），
> **Q 是"查询问题"**（这次要哪些关联），
> **Q·S → O** 是"按需查表拿到答案"。

这个视角回答了一个常见疑问："S 只有 d×d，怎么装得下 L 个 token 的信息？"
答案：S 装的是**聚合关联**，不是**完整记录**。d² 是付得起的常数，L 是付不起的变量——GDN 选择保留前者。

### 性能对比（论文数据）

| 模型 | 注意力机制 | 32K 上下文解码速度 | 256K 上下文解码速度 |
|------|----------|------------------|------------------|
| Qwen3-Max | 标准 Attention | 1x（基准）| 1x（基准）|
| Qwen3.5 | GDN + GA (3:1) | **8.6x** | **19x** |
| Llama 3.1 405B | Full MHA | 较慢 | 不支持 |
| Kimi K2.5 | MLA + DSA | 中等 | 中等 |

> 数据来源：[Qwen3.5 发布博客](https://qwen.ai/blog?id=qwen3.5) / [GDN 论文](https://arxiv.org/abs/2412.06464)

### 3:1 混合的具体含义

Qwen3.5 397B-A17B 总共 60 层 = 15 组 × 4 层。每一组的 4 层中，**前 3 层用 GDN，最后 1 层用 GA**。所以**按层数算，75% 是 GDN，25% 是 GA**。但因为 GDN 的计算复杂度远低于 GA（O(L·d²) vs O(L²·d)），实际绝大部分 token 走 GDN 处理，GA 只用来抓 GDN 处理不好的"复杂依赖"。

这是 Qwen Team 调出来的效率-精度 sweet spot：3:1 比例平衡了效率和精度。

### 工程实现：FLA 优化

朴素 GDN 是逐 token 递推（S = α·S + β·(k⊗v)），GPU 利用率很低。**FLA（Flash Linear Attention）** 通过 **chunking** 把递推改为块状并行计算：把 L 个 token 切成多个 chunk（如 64），chunk 内并行用矩阵乘，chunk 间串行保留因果性。这样训练速度能快 5-10 倍。

FLA 的官方实现：[flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)，感兴趣的可以自己看源码。

---

## 创新点 2：极致稀疏 MoE

> 397B-A17B：397B 总参 / 17B 激活（< 5% 激活率）｜512 专家 / 10 routed + 1 shared / 中间维 1024

### 一句话理解

> **千亿级知识容量 + 十亿级推理成本**：参数大（397B 装得下 397B 份知识），算量小（每次只激活 17B）

### 27B-Dense vs 397B-MoE 结构对比

> **数据来源**：HF 模型 config（`Qwen/Qwen3.5-27B/config.json` + `Qwen/Qwen3.5-397B-A17B/config.json`）+ Qwen 官方发布博客。本地存档：`docs/qwen3_5-mindsped-mm/config-27B.json`、`config-397B.json`

| 维度 | **27B-Dense** | **397B-A17B MoE** | **config 字段** |
|------|---------------|-------------------|-----------------|
| 总参 / 激活 | 27B / 27B | 397B / **17B** | Qwen 博客（"397B-A17B" 命名） |
| 激活率 | 100% | **< 5%** | 17/397 ≈ 4.28% |
| 层数 | **64** | **60** | `num_hidden_layers` |
| Hidden Dim | **5120** | **4096** | `hidden_size` |
| FFN/MoE 中间维 | **17408** | **1024**（×10 routed + ×1 shared） | `intermediate_size` / `moe_intermediate_size` + `shared_expert_intermediate_size` |
| 专家数 | — | 512 / 10 routed + 1 shared | `num_experts` / `num_experts_per_tok` + `shared_expert_intermediate_size` |
| 路由辅助损失 | — | 0.001 | `router_aux_loss_coef` |
| Block 结构 | `Norm → Attn(GDN/GA) → Norm → FFN` | `Norm → Attn(GDN/GA) → Norm → MoE` | `docs/qwen3_5-mindsped-mm/01-...md` L362-378 |
| 注意力 | 3 GDN + 1 GA（每 4 层） | 3 GDN + 1 GA（每 4 层） | `full_attention_interval: 4` + `layer_types` 数组 |
| GDN V 头 | **48** | **64** | `linear_num_value_heads` |
| GDN QK 头 | **16** | **16** | `linear_num_key_heads` |
| GDN 头维（K） | 128 | 128 | `linear_key_head_dim` |
| GDN 头维（V） | 128 | 128 | `linear_value_head_dim` |
| GA Q 头 | **24** | **32** | `num_attention_heads` |
| GA KV 头（GQA） | **4** | **2** | `num_key_value_heads` |
| 头维 | 256 | 256 | `head_dim` |
| Rotary 维 / 比例 | partial 0.25 / mRoPE [11,11,10] | partial 0.25 / mRoPE [11,11,10] | `rope_parameters.partial_rotary_factor` / `mrope_section` |
| 上下文 | 262144 原生 + 1M 扩展 | 262144 原生 + 1M 扩展 | `max_position_embeddings: 262144`（1M 扩展用 YaRN） |
| 词表 | 248320 | 248320 | `vocab_size` |

### 两个关键差异的"为什么"

**1. 为什么 Dense 可以 64 层、MoE 只用 60 层？**
> MoE 每层参数更多（512 专家装在 FFN 位），深度可以减少。同时 60 层 = 15 组 × 4 层，正好对齐 GDN+GA 的 3:1 比例。

**2. 为什么 MoE 的 hidden dim 反而小（4096 vs 5120）？**
> MoE 的"知识容量"在专家里（专家数 × 中间维），不在 hidden dim。所以 hidden dim 可以小、专家可以多。
> 27B-Dense 的知识全压在一个大 FFN（中间维 17408）里，hidden dim 必须大才能装下。
> 一个粗略估算：397B-MoE 的"等效知识"≈ 512 专家 × 1024 中间维 ≈ 50 万维的张量场；27B-Dense 靠 64 层 × 17408 中间维 ≈ 110 万维——**MoE 用"横向并列"换"纵向堆叠"**。

### 三个工程上的取舍

| 取舍点 | 27B-Dense | 397B-A17B MoE |
|--------|-----------|----------------|
| **训练** | 简单，标准 DP+TP+PP | 需专家并行（EP）+ 负载均衡损失 |
| **推理** | 简单，所有参数都跑 | 需 AllToAll 通信，路由抖动要处理 |
| **部署** | 27B 一张/几张卡就行 | 397B 必须多机多卡，激活 17B 也要大显存 |

> 一句话：MoE 用"工程复杂度"换"推理性价比"——总参数拉满知识，激活参数压低算量。

## 创新点 3：Early Fusion 原生多模态

> 文本 / 图像 / 视频在**预训练阶段**就一起训，**没有外挂视觉适配器**。

### 关键差异：原生 vs 后期添加

| 维度 | **Qwen3-VL（后期融合）** | **Qwen3.5（原生融合）** |
|------|------------------------|---------------------|
| 视觉模块 | 训练完 LLM **再挂上** ViT + 投影层 | 预训练**一开始**就把视觉 token 和文本 token 混训 |
| 视觉 token 注入点 | LLM 输入层做一次投影 | **贯穿整个 60 层**（每层 attention 都能看到视觉） |
| 训练范式 | LLM 冻住 / LoRA 调视觉对齐 | **端到端**从头训 |
| 视觉理解上限 | 受限于 LLM 冻结时的能力 | 跟 LLM 同步成长 |

### Vision Encoder 配置（397B-A17B）

> **数据来源**：HF config `Qwen/Qwen3.5-397B-A17B/config.json`（`vision_config` 字段）。本地存档：`docs/qwen3_5-mindsped-mm/config-397B.json`

| 项 | 值 | config 字段 |
|---|---|---|
| 层数 | 27 | `vision_config.depth` |
| Hidden | 1152 | `vision_config.hidden_size` |
| Patch | 16×16 | `vision_config.patch_size` |
| Spatial merge | 2×2 | `vision_config.spatial_merge_size` |
| Temporal patch | 2（视频） | `vision_config.temporal_patch_size` |
| 输出维度 | 4096（对齐 LLM hidden） | `vision_config.out_hidden_size` |
| 视频 token id | 248057 | `video_token_id` |
| 图像 token id | 248056 | `image_token_id` |

> 关键点：`out_hidden_size = 4096`，**直接对齐 LLM 的 hidden_size**——这是原生融合的工程基础，视觉 patch 投影后**形状直接 = 文本 token**，可以无障碍混进 LLM 60 层。

### mRoPE 多模态位置编码

视觉 token 也用 mRoPE（不是 1D RoPE），三段切片：

- `temporal: 11 维`（视频时间轴）
- `height:  11 维`（图像高度）
- `width:   10 维`（图像宽度）

> 文本 token 三段都填同一个位置（伪 1D），视觉 token 三段填实际坐标——**统一公式同时表达 1D/2D/3D 位置**。

### 端到端数据流：从一张图到 LLM 第 60 层

一张 224×224 图走完整个 pipeline（token 数随分辨率变，下面以 224×224 为例）：

| 步骤 | 转换 | 形状 |
|---|---|---|
| 1. 原始图像 | — | 224×224×3 |
| 2. Patch 切分（patch_size=16） | 224/16 = 14 → 14×14 | 196 个 768 维 patch |
| 3. Vision Transformer（27 层） | patch → 1152 维特征 | 196 × 1152 |
| 4. Spatial merge（merge=2） | 2×2 邻接 patch 合成一个 token | 49 × 1152 |
| 5. 投影层（1152 → 4096） | 单层 linear，对齐 LLM hidden | 49 × **4096** ← 形状 = 文本 token |
| 6. 拼接文本 token | 49 视觉 + N 文本 | (49+N) × 4096 |
| 7. mRoPE 位置编码 | 视觉填 3D 坐标 / 文本填伪 1D | (49+N) × 4096 |
| 8. Embedding → Layer 0..59 | 3 GDN + 1 GA 走 15 轮 | (49+N) × 4096 |
| 9. Layer 59 → LM Head | 投影到 vocab | (49+N) × 248320 |

> **关键观察 1**：第 5 步 `out_hidden_size = 4096` 跟 LLM `hidden_size = 4096` 完全相等——这是**单层 linear 投影**敢用的前提。视觉 patch 投影后**形状直接 = 文本 token**，无缝混进 LLM。
>
> **关键观察 2**：第 8 步——**每层 GDN/GA 都能 attend 到这 49 个视觉 token**。Qwen3.5 的 3:1 混合对视觉和文本**一视同仁**，没有"视觉层"概念。
>
> **关键观察 3**：49 这个数字 = (224/16/2)² = 7²。分辨率变了 token 数会变：448×448 = 196，896×896 = 784。视觉 token 数随输入大小自适应。

### 跟"挂个 ViT"的具体差别

| 维度 | **挂个 ViT**（Qwen3-VL 风格） | **Qwen3.5 原生融合** |
|------|------|------|
| 投影层 | 复杂 adapter（多层 MLP + 交叉注意力） | 单层 linear（1152→4096） |
| 视觉 token 数 | 多（196 个不 merge） | 少（49 个，靠 spatial merge 压缩） |
| LLM 权重调整 | 只调 adapter，LLM 冻结 | LLM 60 层权重**都改** |
| 视觉↔文本注意力 | 局部（adapter 内部） | 全局（每层都能） |
| 训练范式 | LLM 训完 → 挂 ViT → adapter 对齐 | ViT 和 LLM **一起从头训** |
| 信息流方向 | 视觉 → LLM（单向） | **双向**（视觉 token 也参与反向梯度） |
| 训练数据规模 | 图文对（百万级） | 图文/视频/交错混合（万亿级 token） |

> **核心区别一句话**：Qwen3.5 敢用 27 层 ViT + 简单投影 + 49 个 token 走 60 层 LLM，**前提是 LLM 权重在训练中被调过**——视觉信息不是"翻译"给 LLM 的，是 LLM **学会了怎么读**这些视觉 token。

## 创新点 4：mRoPE 3 维位置编码

> **数据来源**：HF config `Qwen/Qwen3.5-397B-A17B/config.json`（`rope_parameters` 字段）。本地存档：`docs/qwen3_5-mindsped-mm/config-397B.json`

### 核心机制：1 套公式表达 3 个维度

| 配置 | 值 | config 字段 |
|---|---|---|
| 总维度 | 64 | `head_dim * partial_rotary_factor = 256 * 0.25` |
| 三段切片 | [11, 11, 10] = 32 维 | `rope_parameters.mrope_section` |
| 应用比例 | 25%（只对 25% 维度做 RoPE） | `rope_parameters.partial_rotary_factor` |
| 基频 θ | 10000000 | `rope_parameters.rope_theta` |
| 排布模式 | **interleaved**（交织） | `rope_parameters.mrope_interleaved: true` |

> **直觉**：1D RoPE 把 64 维拆成 32 对 `(sin, cos)`，每对编码一个频率。mRoPE 把这 32 对**按维度切片**到 3 组——前 11 对编时间维、中间 11 对编高度维、最后 10 对编宽度维。**维度上看是切分，频率上看是三套独立 RoPE。**

### 跟 1D RoPE 的代码对比

```python
# 1D RoPE（标准文本）
def rope_1d(pos, dim=64):
    # 32 对 (sin, cos)，所有维度共享 pos
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    freqs = pos * inv_freq          # shape: (L, 32)
    return sin/freqs, cos/freqs     # 整段 64 维都用这 32 个频率

# mRoPE（Qwen3.5）
def mrope(pos_t, pos_h, pos_w, dim=64, section=[11, 11, 10]):
    # pos_t / pos_h / pos_w 都是 (L,) 形状
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    # 关键：32 个频率按 section 切 3 段，每段配一个维度
    freq_t = pos_t[:, None] * inv_freq[0:11]    # 11 个频率编时间
    freq_h = pos_h[:, None] * inv_freq[11:22]   # 11 个频率编高度
    freq_w = pos_w[:, None] * inv_freq[22:32]   # 10 个频率编宽度
    # 拼回 (L, 32)，再扩展到 64 维
    return sin/freqs, cos/freqs
```

> **关键差别**：1D RoPE 全程 1 个位置 `pos`；mRoPE 用 **3 个独立位置**（t/h/w），各编各的，最后拼回一个张量。

### 文本和视觉怎么"统一"

| Token 类型 | `pos_t` | `pos_h` | `pos_w` | 含义 |
|---|---|---|---|---|
| 文本 token | 真实 token 位置 | 复制 pos_t | 复制 pos_t | 退化成 1D RoPE |
| 图像 patch | 0 | patch 行号 | patch 列号 | 编码 2D 空间位置 |
| 视频 patch | 帧号 | patch 行号 | patch 列号 | 编码 3D 时空位置 |

> **统一公式同时表达 1D/2D/3D 位置**——文本 token 三段填同一个值，自然退化成 1D RoPE；视觉 token 三段填真实坐标，编码空间位置。**不需要为多模态单独设计 RoPE**。

### partial_rotary_factor = 0.25：只对 25% 维度做旋转

```python
# Qwen3.5：256 维 head_dim 中只有 64 维做 RoPE
# 剩下 192 维保持原样（不旋转）

x_rot, x_pass = x[..., :64], x[..., 64:]   # 切两段
x_rot = apply_rope(x_rot, freqs)            # 前 25% 旋转
output = concat(x_rot, x_pass, dim=-1)      # 拼回去
```

> **为什么是 25%？** 经验值。RoPE 主要是给 attention 引入"位置差"信息，理论上全维度做也行，但 25% 节省 75% 位置编码计算（Qwen3.5 GDN 和 GA 头维都是 256），**质量没掉，速度白拿**。

### interleaved 模式：不是简单"切 3 段"

config 里 `mrope_interleaved: true`——意味着三段频率**不是按 dim 顺序排列**（[11 个 t, 11 个 h, 10 个 w]），而是**交织**：

```
默认（non-interleaved）：ttttttttttt hhhhhhhhhh wwwwwwwwww
交织（interleaved）：   t h w t h w t h w t h w ...   （前 32 维）
```

> **工程意义**：交织后，每个"位置差异"维度上**同时编码 3 维信息**——更利于 attention 学到"图像 patch 在第 5 帧、第 3 行、第 7 列"这种复合位置关系。**视觉任务上比 non-interleaved 效果更好**。

## 创新点 5：MTP 多 Token 预测

> **数据来源**：HF config `Qwen/Qwen3.5-397B-A17B/config.json`（`mtp_*` 字段）。本地存档：`docs/qwen3_5-mindsped-mm/config-397B.json`

### 核心机制：一次预测多个 token

| 配置 | 值 | config 字段 |
|---|---|---|
| MTP 层数 | 1 | `mtp_num_hidden_layers` |
| 专用 embedding | 不用 | `mtp_use_dedicated_embeddings: false` |

> **直觉**：标准 LM 每个位置只预测下一个 token（1 个监督信号）。MTP 额外**预测后面第 k 个 token**（k=1..K，Qwen3.5 设 K=1 即 1 个额外预测头），**多一个监督信号**。

### 训练时 vs 推理时

| 阶段 | 用途 | 关键做法 |
|---|---|---|
| **训练** | **让模型学更深**——下一 token 预测是浅层特征，K+1 步预测迫使模型学**长距离依赖** | 正常前向，但在每个位置多算 K 个辅助 head，**加到主 loss 上** |
| **推理** | **Speculative Decoding 加速**——主模型用 MTP head 一次猜 K 个 token，小模型验证 | 1 次主模型前向 + K 次验证 ≈ 1.5-2x 加速 |

### 训练时数据流（K=1 示意）

```
主 token 位置 t:  预测 t+1
辅助 token 位置 t: 预测 t+2   ← MTP 头输出

主 loss = CE(t+1) + λ · CE(t+2)
         ↑ 标准 LM 损失     ↑ MTP 辅助损失
```

> **关键点**：`mtp_num_hidden_layers = 1` 表示 MTP head 只有 1 层（轻量），`mtp_use_dedicated_embeddings = false` 表示**复用主模型的 embedding**（不引入新参数表）——**工程上 MTP 的额外成本很低**。

### 推理加速原理

```
传统自回归：  1 步 → 1 token（N 次前向生成 N token）
MTP 投机解码：1 步 → 1 token + K 个猜测
              小模型验证 K 个猜测
              → 1 次主前向 + K 次小前向 ≈ 1.5-2x 加速
```

> **注意**：Qwen3.5 当前 config 只启用了 **K=1**（`mtp_num_hidden_layers=1`），所以辅助预测只有 1 个 token。加速比相对保守；如果 K=2/3，加速更明显但训练成本也涨。

### MTP 在 Qwen3.5 vs Qwen3 vs DeepSeek-V3

| 模型 | MTP 头数 | 训练 | 推理加速 |
|---|---|---|---|
| **DeepSeek-V3** | K=1 | ✅ | ✅ speculative |
| **Qwen3** | 无 | ❌ | ❌ |
| **Qwen3.5** | K=1 | ✅ | ✅ speculative（轻量） |

> Qwen3.5 没用 DeepSeek-V3 的全量 MTP（K=1 是保守选择），是"**以小博大**"的工程平衡：训练成本涨一点，推理加速白拿，质量不掉。

---

## 网络结构

### 整体架构

> **数据来源**：HF config `Qwen/Qwen3.5-397B-A17B/config.json`（`num_hidden_layers=60`、`full_attention_interval=4`、`layer_types` 数组）。本地存档：`docs/qwen3_5-mindsped-mm/config-397B.json`

| 项 | 值 | 说明 |
|---|---|---|
| 总层数 | **60** | `num_hidden_layers` |
| 分组 | 15 组 × 4 层 | 60 ÷ 4 = 15 |
| 每组结构 | 3 × GDN + 1 × GA | 严格 3:1 比例，无例外 |
| 排布模式 | 每组最后 1 层是 GA | `layer_types` 数组前 3 个 `linear_attention`、第 4 个 `full_attention`，循环 15 次 |

**60 层完整排布**（从 config `layer_types` 数组直接读出，60 个值循环 15 轮）：

| 组 | 层 0 | 层 1 | 层 2 | 层 3（GA） |
|---|---|---|---|---|
| 0  | GDN | GDN | GDN | **GA** |
| 1  | GDN | GDN | GDN | **GA** |
| 2  | GDN | GDN | GDN | **GA** |
| 3  | GDN | GDN | GDN | **GA** |
| 4  | GDN | GDN | GDN | **GA** |
| 5  | GDN | GDN | GDN | **GA** |
| 6  | GDN | GDN | GDN | **GA** |
| 7  | GDN | GDN | GDN | **GA** |
| 8  | GDN | GDN | GDN | **GA** |
| 9  | GDN | GDN | GDN | **GA** |
| 10 | GDN | GDN | GDN | **GA** |
| 11 | GDN | GDN | GDN | **GA** |
| 12 | GDN | GDN | GDN | **GA** |
| 13 | GDN | GDN | GDN | **GA** |
| 14 | GDN | GDN | GDN | **GA** |

> **关键观察**：layer_types 数组**严格 3:1**——前 3 层 linear（GDN）、第 4 层 full（GA），**15 轮循环无任何例外**。最后一层（59）是 GA，意味着 GA 始终是每组的"收尾层"。

### 一个 Block 的内部结构

> **数据来源**：`docs/qwen3_5-mindsped-mm/01-qwen3.5-architecture-overview.md` L362-378 + HF config 字段

| 子层 | 维度 | 输入 → 输出 | 备注 |
|---|---|---|---|
| 1. RMSNorm | 4096 | (B, L, 4096) → (B, L, 4096) | `rms_norm_eps=1e-6` |
| 2. Attention（GDN 或 GA） | 4096 | (B, L, 4096) → (B, L, 4096) | GDN=线性递推 / GA=标准 attention |
| 3. Residual Add | — | x + attn(x) | Pre-Norm 范式 |
| 4. RMSNorm | 4096 | 同上 | |
| 5. **MoE**（397B）/ FFN（27B-Dense） | 4096 → 中间维 → 4096 | 见下表 | 397B=MoE / 27B=Dense FFN |
| 6. Residual Add | — | x + ffn(x) | |

> **关键范式**：**Pre-Norm**（先 norm 再 sub-layer，再 residual）—— 现代 Transformer 主流做法，比 Post-Norm 训练更稳定。

#### MoE 子层细节（397B-A17B）

| 步骤 | 维度变化 | 备注 |
|---|---|---|
| 1. Router | (B·L, 4096) → (B·L, 10) | Top-10 路由选择 |
| 2. AllToAll | token 分散到对应 expert GPU | 通信瓶颈 |
| 3. 10 routed experts 并行 | 10 × (B·L/512·10, 1024) | 每个 expert 中间维 1024 |
| 4. 1 shared expert | (B·L, 1024) | 永远激活 |
| 5. 加权合并 | → (B·L, 4096) | 路由权重加权 |
| 6. AllToAll 回 | (B·L, 4096) | token 回到原 GPU |

> **工程观察**：MoE 的 AllToAll 通信是**推理和训练的主要瓶颈**——4 个步骤里 2 个是通信（step 2 + 6），计算本身反而快。

#### FFN 子层细节（27B-Dense，对比）

```
x (B, L, 5120) → linear → (B, L, 17408) → SiLU → linear → (B, L, 5120)
                    ↑ gate                 ↑ 激活              ↑ down_proj
                    ↑ 17408 = 5120 × 3.4（典型 SwiGLU 比例）
```

> **27B vs 397B 的 Block 差别**只有第 5 子层：27B = 标准 FFN（每个 token 走 1 个 FFN），397B = MoE（每个 token 走 10 routed + 1 shared = 11 个 expert 加权）。**前面 4 个子层（Norm + Attn + Residual ×2）完全相同**。

---

## 总结

### 一句话设计哲学

> **Qwen3.5 = 用 5 个"以小博大"的工程创新，把"千亿参数 + 多模态 + 长上下文"塞进单卡可推理的预算里。**

5 个创新**全部都是工程优化**——没有新理论、新架构，是 Qwen Team 把现有技术调到极致后的组合拳。

### 5 大创新总览

| # | 创新 | 核心数据 | 解决的痛点 | 工程代价 |
|---|---|---|---|---|
| 1 | **GDN** | 3:1 混合 / 8.6x 提速 | L² 复杂度长上下文慢 | 容量有限 + α 衰减 |
| 2 | **极致稀疏 MoE** | 397B 总 / 17B 激活 / < 5% 激活率 | 大模型推理贵 | AllToAll 通信 + 路由抖动 |
| 3 | **Early Fusion 原生多模态** | 27 层 ViT → 49 token → 60 层 LLM | 挂个 ViT 信息流单向 | 训练数据要万亿级图文视频 |
| 4 | **mRoPE 3 维位置编码** | [11,11,10] 切片 / partial 0.25 | 多模态位置难统一 | 3 套频率要协调 |
| 5 | **MTP 多 Token 预测** | K=1 头 / 复用主 embedding | 推理自回归慢 | 主 loss + 辅助 loss |

### 5 个创新的"共同 DNA"

| 共同点 | 体现 |
|---|---|
| **都以"小"搏"大"** | d² 装 L / 5% 激活率 / 49 token 走 60 层 / 25% 维度 RoPE / K=1 预测 |
| **都是"复用已有"** | MoE 复用 GDN 的 hidden 维度 / Early Fusion 复用 mRoPE / MTP 复用主 embedding |
| **都依赖"端到端训练"** | GDN 要学 α-β / MoE 要学路由 / Early Fusion 要 LLM 调权重 / mRoPE 要学 3 段频率 / MTP 要学辅助头 |
| **都增加了"通信/调度"开销** | GDN 的递推 / MoE 的 AllToAll / Early Fusion 的视觉特征注入 / mRoPE 的多维位置 / MTP 的多 head |

### 3 个"为什么 Qwen3.5 选了这个组合"

| 为什么 | 答 |
|--------|-----|
| **为什么 GDN+GA 混合，不全用 GDN？** | **精度保证**——25% 的 GA 层抓 GDN 处理不好的复杂依赖（参考创新 1 Q&A） |
| **为什么 MoE 这么稀疏（< 5%）？** | **推理成本**——千亿级知识容量配十亿级算量，部署成本降 60% |
| **为什么 mRoPE 切片是 [11,11,10] 不是均分？** | **工程经验**——高度/宽度维更重要（编视觉空间），时间维次之（编视频时序），所以高度宽度多 1 维 |

### 回到 Qwen3.5 的"3 个为什么"

| 为什么 | 答 |
|--------|-----|
| **为什么用 GDN 替代大部分 attention？** | **长上下文效率**（O(L) vs O(L²)），8.6-19x 提速 |
| **为什么混合 GDN+GA 不全用 GDN？** | **精度保证**（标准 attention 抓复杂依赖更强），3:1 平衡效率与质量 |
| **为什么 MoE 这么稀疏？** | **推理成本**（17B 激活 vs 397B 总参数），部署成本降低 60% |

### 推荐阅读路径

- 想理解 **长上下文效率**：创新 1（GDN）
- 想理解 **大模型推理降本**：创新 2（MoE）
- 想理解 **多模态原理**：创新 3（Early Fusion）+ 创新 4（mRoPE）
- 想理解 **推理加速**：创新 5（MTP）
- 想理解 **整体结构**：网络结构 + 总结

> 路径 2 博客《Qwen3.5 的创新和网络结构》完结。下一步推荐深入 GDN 源码（路径 2 文档 02）或者恢复路径 1 挂起的 AI Infra 60 天。

---

## Q&A

**Q1：S 只有 d×d，怎么装得下 L 个 token 的信息？**
> 装不下完整信息——S 是"压缩笔记"，存的是关联模式不是完整记录。详见上文「直观理解：S 是查询指南」三视角（聚合统计量 / 流动的关联池 / α-β 可学习开关）。

**Q2：那 GA（标准 attention）不是更准？为什么要 GDN？**
> GA 准但慢。Qwen3.5 的策略是"快为主，准为辅"。75% 的层用 GDN 处理大部分 token，25% 的层用 GA 抓 GDN 处理不好的复杂依赖。就像快递：90% 的小件用电动车（快），10% 的大件用卡车（准）。

**Q3：训练 GDN 难吗？**
> 相对简单：A_log 初始化为 0（α ≈ 0.37），β 由模型学，没有特殊的 warm-up 需要。难点在工程：怎么把逐 token 递推改成块状并行计算（FLA 优化）。这属于工程问题，不是算法问题。

**Q4：FLA 是怎么把递推改为并行的？**
> 核心是 chunking：把 L 个 token 切成多个 chunk（如 64），chunk 内用矩阵乘并行计算，chunk 间保持串行依赖。完整实现见 [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) 库。
