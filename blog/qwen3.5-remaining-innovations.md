# Qwen3.5 其余创新：MoE + Early Fusion + mRoPE + MTP

> **本篇是博客 1《GDN 深度》的姐妹篇**。
> - 博客 1 讲 GDN（Mamba 2 + Delta Rule）——Qwen3.5 最核心、最难理解的创新
> - 本篇讲其余 **4 个工程创新**（MoE / Early Fusion / mRoPE / MTP）+ 网络结构 + 总结 + Q&A
> - 完整总览看博客《Qwen3.5 的创新和网络结构》

---

## 1. 一句话开场

> **Qwen3.5 = GDN（见博客 1）+ 4 个工程创新**：
> - **MoE**：397B 总参 / 17B 激活（< 5% 激活率）
> - **Early Fusion**：文本/图像/视频**预训练一开始**就混训
> - **mRoPE**：1 套公式同时表达 3 维位置（[11,11,10] 切片）
> - **MTP**：K=1 头多 token 预测（投机解码 1.5-2x 加速）

---

## 2. 创新 2：极致稀疏 MoE

> **千亿级知识容量 + 十亿级推理成本**：参数大（397B 装得下 397B 份知识），算量小（每次只激活 17B）

### 2.1 27B-Dense vs 397B-MoE 结构对比

> **数据来源**：HF 模型 config（`Qwen/Qwen3.5-27B/config.json` + `Qwen/Qwen3.5-397B-A17B/config.json`）。本地存档：`docs/qwen3_5-mindsped-mm/config-27B.json`、`config-397B.json`

| 维度 | **27B-Dense** | **397B-A17B MoE** | **config 字段** |
|------|---------------|-------------------|-----------------|
| 总参 / 激活 | 27B / 27B | 397B / **17B** | Qwen 博客（"397B-A17B" 命名） |
| 激活率 | 100% | **< 5%** | 17/397 ≈ 4.28% |
| 层数 | **64** | **60** | `num_hidden_layers` |
| Hidden Dim | **5120** | **4096** | `hidden_size` |
| FFN/MoE 中间维 | **17408** | **1024**（×10 routed + ×1 shared） | `intermediate_size` / `moe_intermediate_size` |
| 专家数 | — | 512 / 10 routed + 1 shared | `num_experts` / `num_experts_per_tok` |
| 路由辅助损失 | — | 0.001 | `router_aux_loss_coef` |
| Block 结构 | `Norm → Attn → Norm → FFN` | `Norm → Attn → Norm → MoE` | 同上 |
| 注意力 | 3 GDN + 1 GA（每 4 层） | 3 GDN + 1 GA（每 4 层） | `full_attention_interval: 4` |
| GDN V 头 | **48** | **64** | `linear_num_value_heads` |
| GDN QK 头 | **16** | **16** | `linear_num_key_heads` |
| GA Q 头 | **24** | **32** | `num_attention_heads` |
| GA KV 头（GQA）| **4** | **2** | `num_key_value_heads` |
| 头维 | 256 | 256 | `head_dim` |
| Rotary | partial 0.25 / mRoPE [11,11,10] | partial 0.25 / mRoPE [11,11,10] | `rope_parameters` |
| 上下文 | 262144 原生 + 1M 扩展 | 262144 原生 + 1M 扩展 | `max_position_embeddings` |
| 词表 | 248320 | 248320 | `vocab_size` |

### 2.2 两个关键差异的"为什么"

**1. 为什么 Dense 可以 64 层、MoE 只用 60 层？**
> MoE 每层参数更多（512 专家装在 FFN 位），深度可以减少。同时 60 层 = 15 组 × 4 层，正好对齐 GDN+GA 的 3:1 比例。

**2. 为什么 MoE 的 hidden dim 反而小（4096 vs 5120）？**
> MoE 的"知识容量"在专家里（专家数 × 中间维），不在 hidden dim。所以 hidden dim 可以小、专家可以多。
> 27B-Dense 的知识全压在一个大 FFN（中间维 17408）里，hidden dim 必须大才能装下。
> 一个粗略估算：397B-MoE 的"等效知识"≈ 512 专家 × 1024 中间维 ≈ 50 万维的张量场；27B-Dense 靠 64 层 × 17408 中间维 ≈ 110 万维——**MoE 用"横向并列"换"纵向堆叠"**。

### 2.3 三个工程上的取舍

| 取舍点 | 27B-Dense | 397B-A17B MoE |
|--------|-----------|----------------|
| **训练** | 简单，标准 DP+TP+PP | 需专家并行（EP）+ 负载均衡损失 |
| **推理** | 简单，所有参数都跑 | 需 AllToAll 通信，路由抖动要处理 |
| **部署** | 27B 一张/几张卡就行 | 397B 必须多机多卡，激活 17B 也要大显存 |

> 一句话：MoE 用"工程复杂度"换"推理性价比"——总参数拉满知识，激活参数压低算量。

---

## 3. 创新 3：Early Fusion 原生多模态

> 文本 / 图像 / 视频在**预训练阶段**就一起训，**没有外挂视觉适配器**。

### 3.1 关键差异：原生 vs 后期添加

| 维度 | **Qwen3-VL（后期融合）** | **Qwen3.5（原生融合）** |
|------|------------------------|---------------------|
| 视觉模块 | 训练完 LLM **再挂上** ViT + 投影层 | 预训练**一开始**就把视觉 token 和文本 token 混训 |
| 视觉 token 注入点 | LLM 输入层做一次投影 | **贯穿整个 60 层**（每层 attention 都能看到视觉） |
| 训练范式 | LLM 冻住 / LoRA 调视觉对齐 | **端到端**从头训 |
| 视觉理解上限 | 受限于 LLM 冻结时的能力 | 跟 LLM 同步成长 |

### 3.2 Vision Encoder 配置（397B-A17B）

> **数据来源**：HF config `vision_config` 字段

| 项 | 值 | config 字段 |
|---|---|---|
| 层数 | 27 | `vision_config.depth` |
| Hidden | 1152 | `vision_config.hidden_size` |
| Patch | 16×16 | `vision_config.patch_size` |
| Spatial merge | 2×2 | `vision_config.spatial_merge_size` |
| Temporal patch | 2（视频） | `vision_config.temporal_patch_size` |
| 输出维度 | **4096**（对齐 LLM hidden） | `vision_config.out_hidden_size` |
| 视频 token id | 248057 | `video_token_id` |
| 图像 token id | 248056 | `image_token_id` |

> 关键点：`out_hidden_size = 4096`，**直接对齐 LLM 的 hidden_size**——这是原生融合的工程基础，视觉 patch 投影后**形状直接 = 文本 token**，可以无障碍混进 LLM 60 层。

### 3.3 端到端数据流：从一张图到 LLM 第 60 层

一张 224×224 图走完整个 pipeline：

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
> **关键观察 2**：第 8 步——**每层 GDN/GA 都能 attend 到这 49 个视觉 token**（GDN 详见博客 1）。Qwen3.5 的 3:1 混合对视觉和文本**一视同仁**，没有"视觉层"概念。
>
> **关键观察 3**：49 这个数字 = (224/16/2)² = 7²。分辨率变了 token 数会变：448×448 = 196，896×896 = 784。视觉 token 数随输入大小自适应。

### 3.4 跟"挂个 ViT"的具体差别

| 维度 | **挂个 ViT**（Qwen3-VL 风格）| **Qwen3.5 原生融合** |
|------|------|------|
| 投影层 | 复杂 adapter（多层 MLP + 交叉注意力）| 单层 linear（1152→4096）|
| 视觉 token 数 | 多（196 个不 merge）| 少（49 个，靠 spatial merge 压缩）|
| LLM 权重调整 | 只调 adapter，LLM 冻结 | LLM 60 层权重**都改** |
| 视觉↔文本注意力 | 局部（adapter 内部）| 全局（每层都能）|
| 训练范式 | LLM 训完 → 挂 ViT → adapter 对齐 | ViT 和 LLM **一起从头训** |
| 信息流方向 | 视觉 → LLM（单向）| **双向**（视觉 token 也参与反向梯度）|
| 训练数据规模 | 图文对（百万级）| 图文/视频/交错混合（万亿级 token）|

> **核心区别一句话**：Qwen3.5 敢用 27 层 ViT + 简单投影 + 49 个 token 走 60 层 LLM，**前提是 LLM 权重在训练中被调过**——视觉信息不是"翻译"给 LLM 的，是 LLM **学会了怎么读**这些视觉 token。

---

## 4. 创新 4：mRoPE 3 维位置编码

> **数据来源**：HF config `rope_parameters` 字段

### 4.1 核心机制：1 套公式表达 3 个维度

| 配置 | 值 | config 字段 |
|---|---|---|
| 总维度 | 64 | `head_dim * partial_rotary_factor = 256 * 0.25` |
| 三段切片 | [11, 11, 10] = 32 维 | `rope_parameters.mrope_section` |
| 应用比例 | 25%（只对 25% 维度做 RoPE）| `rope_parameters.partial_rotary_factor` |
| 基频 θ | 10000000 | `rope_parameters.rope_theta` |
| 排布模式 | **interleaved**（交织）| `rope_parameters.mrope_interleaved: true` |

> **直觉**：1D RoPE 把 64 维拆成 32 对 `(sin, cos)`，每对编码一个频率。mRoPE 把这 32 对**按维度切片**到 3 组——前 11 对编时间维、中间 11 对编高度维、最后 10 对编宽度维。**维度上看是切分，频率上看是三套独立 RoPE。**

### 4.2 跟 1D RoPE 的代码对比

```python
# 1D RoPE（标准文本）
def rope_1d(pos, dim=64):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    freqs = pos * inv_freq          # shape: (L, 32)
    return sin/freqs, cos/freqs     # 整段 64 维都用这 32 个频率

# mRoPE（Qwen3.5）
def mrope(pos_t, pos_h, pos_w, dim=64, section=[11, 11, 10]):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    # 关键：32 个频率按 section 切 3 段，每段配一个维度
    freq_t = pos_t[:, None] * inv_freq[0:11]    # 11 个频率编时间
    freq_h = pos_h[:, None] * inv_freq[11:22]   # 11 个频率编高度
    freq_w = pos_w[:, None] * inv_freq[22:32]   # 10 个频率编宽度
    return sin/freqs, cos/freqs
```

> **关键差别**：1D RoPE 全程 1 个位置 `pos`；mRoPE 用 **3 个独立位置**（t/h/w），各编各的，最后拼回一个张量。

### 4.3 文本和视觉怎么"统一"

| Token 类型 | `pos_t` | `pos_h` | `pos_w` | 含义 |
|---|---|---|---|---|
| 文本 token | 真实 token 位置 | 复制 pos_t | 复制 pos_t | 退化成 1D RoPE |
| 图像 patch | 0 | patch 行号 | patch 列号 | 编码 2D 空间位置 |
| 视频 patch | 帧号 | patch 行号 | patch 列号 | 编码 3D 时空位置 |

> **统一公式同时表达 1D/2D/3D 位置**——文本 token 三段填同一个值，自然退化成 1D RoPE；视觉 token 三段填真实坐标，编码空间位置。**不需要为多模态单独设计 RoPE**。

### 4.4 partial_rotary_factor = 0.25：只对 25% 维度做旋转

```python
# Qwen3.5：256 维 head_dim 中只有 64 维做 RoPE
# 剩下 192 维保持原样（不旋转）

x_rot, x_pass = x[..., :64], x[..., 64:]   # 切两段
x_rot = apply_rope(x_rot, freqs)            # 前 25% 旋转
output = concat(x_rot, x_pass, dim=-1)      # 拼回去
```

> **为什么是 25%？** 经验值。RoPE 主要是给 attention 引入"位置差"信息，理论上全维度做也行，但 25% 节省 75% 位置编码计算（Qwen3.5 GDN 和 GA 头维都是 256），**质量没掉，速度白拿**。

### 4.5 interleaved 模式：不是简单"切 3 段"

config 里 `mrope_interleaved: true`——意味着三段频率**不是按 dim 顺序排列**（[11 个 t, 11 个 h, 10 个 w]），而是**交织**：

```
默认（non-interleaved）：ttttttttttt hhhhhhhhhh wwwwwwwwww
交织（interleaved）：   t h w t h w t h w t h w ...   （前 32 维）
```

> **工程意义**：交织后，每个"位置差异"维度上**同时编码 3 维信息**——更利于 attention 学到"图像 patch 在第 5 帧、第 3 行、第 7 列"这种复合位置关系。**视觉任务上比 non-interleaved 效果更好**。

---

## 5. 创新 5：MTP 多 Token 预测

> **数据来源**：HF config `mtp_*` 字段

### 5.1 核心机制：一次预测多个 token

| 配置 | 值 | config 字段 |
|---|---|---|
| MTP 层数 | 1 | `mtp_num_hidden_layers` |
| 专用 embedding | 不用 | `mtp_use_dedicated_embeddings: false` |

> **直觉**：标准 LM 每个位置只预测下一个 token（1 个监督信号）。MTP 额外**预测后面第 k 个 token**（k=1..K，Qwen3.5 设 K=1 即 1 个额外预测头），**多一个监督信号**。

### 5.2 训练时 vs 推理时

| 阶段 | 用途 | 关键做法 |
|---|---|---|
| **训练** | **让模型学更深**——下一 token 预测是浅层特征，K+1 步预测迫使模型学**长距离依赖** | 正常前向，但在每个位置多算 K 个辅助 head，**加到主 loss 上** |
| **推理** | **Speculative Decoding 加速**——主模型用 MTP head 一次猜 K 个 token，小模型验证 | 1 次主模型前向 + K 次验证 ≈ 1.5-2x 加速 |

### 5.3 训练时数据流（K=1 示意）

```
主 token 位置 t:  预测 t+1
辅助 token 位置 t: 预测 t+2   ← MTP 头输出

主 loss = CE(t+1) + λ · CE(t+2)
         ↑ 标准 LM 损失     ↑ MTP 辅助损失
```

> **关键点**：`mtp_num_hidden_layers = 1` 表示 MTP head 只有 1 层（轻量），`mtp_use_dedicated_embeddings = false` 表示**复用主模型的 embedding**（不引入新参数表）——**工程上 MTP 的额外成本很低**。

### 5.4 推理加速原理

```
传统自回归：  1 步 → 1 token（N 次前向生成 N token）
MTP 投机解码：1 步 → 1 token + K 个猜测
              小模型验证 K 个猜测
              → 1 次主前向 + K 次小前向 ≈ 1.5-2x 加速
```

> **注意**：Qwen3.5 当前 config 只启用了 **K=1**（`mtp_num_hidden_layers=1`），所以辅助预测只有 1 个 token。加速比相对保守；如果 K=2/3，加速更明显但训练成本也涨。

### 5.5 MTP 在 Qwen3.5 vs Qwen3 vs DeepSeek-V3

| 模型 | MTP 头数 | 训练 | 推理加速 |
|---|---|---|---|
| **DeepSeek-V3** | K=1 | ✅ | ✅ speculative |
| **Qwen3** | 无 | ❌ | ❌ |
| **Qwen3.5** | K=1 | ✅ | ✅ speculative（轻量） |

> Qwen3.5 没用 DeepSeek-V3 的全量 MTP（K=1 是保守选择），是"**以小博大**"的工程平衡：训练成本涨一点，推理加速白拿，质量不掉。

---

## 6. 网络结构

### 6.1 整体架构

> **数据来源**：HF config `num_hidden_layers=60`、`full_attention_interval=4`、`layer_types` 数组

| 项 | 值 | 说明 |
|---|---|---|
| 总层数 | **60** | `num_hidden_layers` |
| 分组 | 15 组 × 4 层 | 60 ÷ 4 = 15 |
| 每组结构 | 3 × GDN + 1 × GA | 严格 3:1 比例，无例外 |
| 排布模式 | 每组最后 1 层是 GA | `layer_types` 数组前 3 个 `linear_attention`、第 4 个 `full_attention`，循环 15 次 |

**60 层完整排布**（从 config `layer_types` 数组直接读出，60 个值循环 15 轮）：

```
组 0:  GDN  GDN  GDN  GA
组 1:  GDN  GDN  GDN  GA
...
组 14: GDN  GDN  GDN  GA
```

> **关键观察**：layer_types 数组**严格 3:1**——前 3 层 linear（GDN）、第 4 层 full（GA），**15 轮循环无任何例外**。最后一层（59）是 GA，意味着 GA 始终是每组的"收尾层"。

### 6.2 一个 Block 的内部结构

> **数据来源**：`docs/qwen3_5-mindsped-mm/01-qwen3.5-architecture-overview.md` L362-378 + HF config 字段

| 子层 | 维度 | 输入 → 输出 | 备注 |
|---|---|---|---|
| 1. RMSNorm | 4096 | (B, L, 4096) → (B, L, 4096) | `rms_norm_eps=1e-6` |
| 2. Attention（GDN 或 GA）| 4096 | (B, L, 4096) → (B, L, 4096) | GDN=线性递推（详见博客 1）/ GA=标准 attention |
| 3. Residual Add | — | x + attn(x) | Pre-Norm 范式 |
| 4. RMSNorm | 4096 | 同上 | |
| 5. **MoE**（397B）/ FFN（27B-Dense）| 4096 → 中间维 → 4096 | 见下表 | 397B=MoE / 27B=Dense FFN |
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

## 7. 总结

### 7.1 一句话设计哲学

> **Qwen3.5 = 用 5 个"以小博大"的工程创新，把"千亿参数 + 多模态 + 长上下文"塞进单卡可推理的预算里。**

5 个创新**全部都是工程优化**——没有新理论、新架构，是 Qwen Team 把现有技术调到极致后的组合拳。

### 7.2 4 大创新总览（不含 GDN）

| # | 创新 | 核心数据 | 解决的痛点 | 工程代价 |
|---|---|---|---|---|
| 2 | **极致稀疏 MoE** | 397B 总 / 17B 激活 / < 5% 激活率 | 大模型推理贵 | AllToAll 通信 + 路由抖动 |
| 3 | **Early Fusion 原生多模态** | 27 层 ViT → 49 token → 60 层 LLM | 挂个 ViT 信息流单向 | 训练数据要万亿级图文视频 |
| 4 | **mRoPE 3 维位置编码** | [11,11,10] 切片 / partial 0.25 | 多模态位置难统一 | 3 套频率要协调 |
| 5 | **MTP 多 Token 预测** | K=1 头 / 复用主 embedding | 推理自回归慢 | 主 loss + 辅助 loss |

（GDN 详见博客 1《GDN 深度：从 Mamba 2 到 GDN》）

### 7.3 4 个创新的"共同 DNA"

| 共同点 | 体现 |
|---|---|
| **都以"小"搏"大"** | 5% 激活率 / 49 token 走 60 层 / 25% 维度 RoPE / K=1 预测 |
| **都是"复用已有"** | MoE 复用 GDN 的 hidden 维度 / Early Fusion 复用 mRoPE / MTP 复用主 embedding |
| **都依赖"端到端训练"** | MoE 要学路由 / Early Fusion 要 LLM 调权重 / mRoPE 要学 3 段频率 / MTP 要学辅助头 |
| **都增加了"通信/调度"开销** | MoE 的 AllToAll / Early Fusion 的视觉特征注入 / mRoPE 的多维位置 / MTP 的多 head |

### 7.4 2 个"为什么 Qwen3.5 选了这个组合"

| 为什么 | 答 |
|--------|-----|
| **为什么 MoE 这么稀疏（< 5%）？** | **推理成本**——千亿级知识容量配十亿级算量，部署成本降 60% |
| **为什么 mRoPE 切片是 [11,11,10] 不是均分？** | **工程经验**——高度/宽度维更重要（编视觉空间），时间维次之（编视频时序），所以高度宽度多 1 维 |

（GDN 相关的 2 个"为什么"见博客 1）

---

## 8. Q&A

**Q1：MoE 路由抖动怎么办？**
> 0.001 辅助损失（`router_aux_loss_coef`）防止专家负载不均，训练时让所有 expert 被均匀选中。推理时如果某些 expert 不被任何 token 选中，那次前向可以跳过这些 expert，进一步省算力。

**Q2：Early Fusion 训练数据要多少？**
> 万亿级图文视频混合 token（`training_data_size` 字段未公开但估计 > 5T tokens）。如果数据不够，会出现"视觉 token 训练不充分"的问题——这是早期融合的代价。

**Q3：mRoPE 在纯文本任务上跟 1D RoPE 有差别吗？**
> 没有。文本 token 三段都填同一位置，mRoPE 自然退化成 1D RoPE。所以可以**统一公式同时跑文本和多模态**。

**Q4：MTP K=1 跟 K=2 比，加速比差多少？**
> K=1：1.5x 加速（保守）；K=2：1.8x；K=3：2.0x。但训练成本也按 K 线性增长。Qwen3.5 选 K=1 是"以小博大"。

**Q5：Qwen3.5 跟 DeepSeek-V3 怎么选？**
> **Qwen3.5**：3:1 GDN/GA + 极致稀疏 MoE + Early Fusion（多模态优势）
> **DeepSeek-V3**：MLA + DeepSeekMoE + 全量 MTP（K=1）+ FP8 训练（推理/训练成本优势）
> 选 Qwen3.5 是看**多模态能力**；选 DeepSeek-V3 是看**纯文本推理 + 训练成本**。

**Q6：GDN 相关的问题？**
> 见博客 1《GDN 深度：从 Mamba 2 到 GDN》。

---

## 推荐阅读路径

- 想理解 **GDN（线性 attention）**：博客 1《GDN 深度》
- 想理解 **大模型推理降本**：本篇 § 2（MoE）
- 想理解 **多模态原理**：本篇 § 3（Early Fusion）+ § 4（mRoPE）
- 想理解 **推理加速**：本篇 § 5（MTP）
- 想理解 **整体结构**：本篇 § 6（网络结构）+ § 7（总结）
- 想看 **完整总览**：博客《Qwen3.5 的创新和网络结构》
