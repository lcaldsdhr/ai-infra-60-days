# Qwen3.5 架构总览

> **来源**：Qwen 官方 GitHub（[QwenLM/Qwen3.5](https://github.com/QwenLM/Qwen3.5)）+ HF Transformers + 多份技术报告
> **整理日期**：2026-06-22
> **适用人群**：AI Infra / 大模型架构学习者
> **状态**：第 1 份（架构总览），后续会有 GDN 详解、算子分析、对比等

---

## 📚 目录

1. [模型族总览](#-模型族总览-8-个尺寸)
2. [4 大核心创新](#-4-大核心创新)
3. [397B-A17B 详细参数](#-397b-a17b-架构详细参数)
4. [Gated DeltaNet (GDN) 详解](#-gated-deltanet-gdn--最核心创新)
5. [Gated Attention (GA) 详解](#-gated-attention-ga--31-比例中的-1)
6. [mRoPE 多模态位置编码](#-mrope多模态-rope)
7. [完整层级结构图](#-完整层级结构图)
8. [5 大关键算子](#-5-大关键算子)
9. [与同类模型对比](#-与同类模型对比)
10. [总结：3 个"为什么"](#-总结qwen35-的-3-个为什么)
11. [参考资源](#-参考资源)
12. [下一步计划](#-下一步计划)

---

## 🏛 模型族总览（8 个尺寸）

| 类型 | 尺寸 | 发布时间 | 架构 |
|------|------|---------|------|
| **MoE** | **397B-A17B** ⭐ 首发 | 2026-02-16 | Hybrid GDN+GA + MoE |
| **MoE** | 122B-A10B | 2026-02-24 | Hybrid GDN+GA + MoE |
| **MoE** | 35B-A3B | 2026-02-24 | Hybrid GDN+GA + MoE |
| **Dense** | 27B | 2026-02-24 | Hybrid GDN+GA + FFN |
| **Dense** | 9B | 2026-03-02 | Hybrid GDN+GA + FFN |
| **Dense** | 4B | 2026-03-02 | Hybrid GDN+GA + FFN |
| **Dense** | 2B | 2026-03-02 | Hybrid GDN+GA + FFN |
| **Dense** | 0.8B | 2026-03-02 | Hybrid GDN+GA + FFN |

> **所有尺寸** 都基于 **Qwen3-Next 架构**（2025-09-11 的 80B-A3B 是技术祖先）
> **首发模型**：Qwen3.5-397B-A17B（也称 Qwen3.5-Plus，闭源 API 版本）
> **协议**：Apache 2.0
> **发布时间**：2026-02-16（除夕夜）

---

## 🎯 4 大核心创新

### 创新 1️⃣ **混合注意力机制（Hybrid Attention）** ⭐⭐⭐

**Qwen3.5 最核心的创新**

```
每 4 层 = 3 × Gated DeltaNet (GDN) + 1 × Gated Attention (GA)
比例 3:1
```

- **GDN**：线性注意力，复杂度 O(L·d²)，处理大多数 token
- **GA**：标准 softmax 注意力，每 4 层 1 次，保证复杂依赖捕捉
- **论文基础**：[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

### 创新 2️⃣ **极致稀疏 MoE**

```
397B-A17B:  397B 总参数 / 17B 激活
激活率:     < 5%（普通 MoE 是 10-15%）
512 专家，10 routed + 1 shared
专家中间维 1024
```

**工程意义**：千亿级参数的"知识容量"，但推理成本仅相当于 17B 模型。

### 创新 3️⃣ **原生多模态早期融合（Early Fusion）**

```
文本 + 图像 + 视频 原生融合训练
不挂视觉适配器（区别于 Qwen3-VL）
mRoPE 多模态位置编码
视觉理解超越 Qwen3-VL
```

**核心差异**：与 Qwen3-VL 的"语言模型 + 后期添加视觉"不同，Qwen3.5 在**预训练阶段**就把视觉 token 和文本 token 一起训练。

### 创新 4️⃣ **极致性能优化**

```
FP8 训练（Qwen3 是 BF16/FP16）
32K 上下文：解码吞吐量 8.6x Qwen3-Max
256K 上下文：19x Qwen3-Max
部署成本降低 60%
```

---

## 📐 397B-A17B 架构详细参数

### 整体参数

| 项 | 值 |
|----|-----|
| **总参数** | 397B |
| **激活参数** | 17B |
| **层数** | 60 |
| **Hidden Dim** | 4096 |
| **Token Embed** | 248,320（已填充） |
| **上下文** | 原生 262,144 / 扩展 1,010,000（YaRN） |
| **词汇表** | 248,320 tokens |
| **支持语言** | 201 种 |

### 层级结构

```
60 层 = 15 × ( 3 × (GDN → MoE)  +  1 × (GA → MoE) )
                  ↓                     ↓
              线性注意力           标准 Attention
              专家路由            专家路由
```

### 关键模块配置

| 模块 | Q 头 | K/V 头 | 头维 | 总维度 | 备注 |
|------|------|--------|------|--------|------|
| **GDN** | 16 | 16 | 128 | V 头 64（**K/V 不对称**） | 线性注意力 |
| **GA** | 32 | 2（GQA） | 256 | Q 头 32 + KV 头 2 | 标准 attention |
| **MoE** | — | — | — | 512 专家 / 10 routed + 1 shared | 中间维 1024 |
| **mRoPE** | — | — | — | 64 维 | partial_rotary_factor=0.25 |

### FFN / MoE 共享专家

- 397B-A17B 的 MoE 中有 **1 个 shared expert**（永远激活）+ **10 个 routed experts**（按路由选）
- **共享专家**保证基础能力，**路由专家**保证专业能力

### 27B-Dense 版（用于对比）

| 项 | 值 |
|----|-----|
| 总参数 | 27B |
| 隐藏层维度 | 5120 |
| 层数 | 64 |
| 层级结构 | 16 × (3 × GDN-FFN + 1 × GA-FFN) |
| **GDN** | V 头 48 / QK 头 16 / 头维 128 |
| **GA** | Q 头 24 / KV 头 4 / 头维 256 / RoPE 维 64 |
| **FFN** | 中间维 17408 |

---

## 🧠 Gated DeltaNet (GDN) — **最核心创新**

### 核心思想

> 用**线性递推状态**代替标准 attention 的 Q·K^T 矩阵

### 数学形式对比

```
标准 attention:
   O = softmax(QK^T / √d) · V      ← 复杂度 O(L²·d)

GDN:
   S_t = α·S_{t-1} + β·(k_t ⊗ v_t)   ← 状态递推，复杂度 O(L·d²)
   o_t = q_t · S_t
```

其中：
- `S_t` 是 d×d 的状态矩阵
- `α` 是衰减门控（防止状态无限增长）
- `β` 是输入门控
- `k_t ⊗ v_t` 是 outer product

### 关键参数

```python
# Qwen3.5 27B-Dense 的 GDN 配置（来自配置）
linear_conv_kernel_dim: 4        # 因果卷积核大小
linear_key_head_dim: 128         # K 头维
linear_value_head_dim: 128       # V 头维
linear_num_key_heads: 16         # K 头数
linear_num_value_heads: 48       # V 头数（**K/V 头数不一样**）
```

### GDN 的两个门控（**重点**）

```python
# 1. 衰减门控 α = exp(g)
#    g = -exp(A_log)  ← 可学习参数
#    控制历史信息的衰减率

# 2. 输入门控 β = sigmoid(b)
#    b = linear(x)  ← 由输入决定
#    控制当前 token 的贡献
```

### 因果卷积（Causal Conv1d）

```python
# 在 GDN 之前先做 1D 因果卷积（卷积核 = 4）
# 让 Q/K/V 共享局部上下文（局部特征提取）
mixed_qkv = causal_conv1d(x, conv_weight)  # shape (B, S, D)
q, k, v = split(mixed_qkv)  # 拆分
```

### 完整流程图

```
Input x (B, L, D)
    │
    ├──► in_proj_qkv ──► [Q, K, V] packed (B, L, 2*K + V)
    │                              │
    │                              ▼
    │                       causal_conv1d (kernel=4)
    │                              │
    │                              ▼
    │                          Q, K, V split
    │                              │
    │                              ▼
    │              ┌─── apply_gate (α = exp(g))  ← ★ 门控1：衰减
    │              │
    │              ├─── Gated Delta Rule: S_t = α·S + β·(k⊗v) ← ★ 门控2：输入
    │              │
    │              ├─── output: o = q · S
    │              │
    │              ▼
    │         out_proj (back to hidden_size)
    │
    └─► output (B, L, D)
```

### 解决什么问题

| 问题 | 解决方案 |
|------|---------|
| 标准 attention O(L²) 长序列慢 | GDN 线性递推 O(L·d²) |
| Attention sinks（注意力沉没） | 衰减门控 α 让历史信息自然衰减 |
| 巨大激活值 | 门控机制限制状态矩阵增长 |
| 训练不稳定 | Delta rule 增量更新比 full softmax 稳定 |

---

## 🔍 Gated Attention (GA) — **3:1 比例中的 "1"**

### 与标准 attention 的 3 个差异

| 项 | 标准 Attention | **Gated Attention (Qwen3.5)** |
|----|--------------|-------------------------------|
| **输出门** | 无 | **sigmoid gate** 缩放 attention 输出 |
| **QK 归一化** | 无 | **zero-centered RMSNorm** |
| **RoPE** | 全 | **partial**（partial_rotary_factor=0.25） |

### 数学形式

```python
# 标准 attention
attn = softmax(Q @ K^T / √d) @ V
output = attn  # 直接输出

# Qwen3.5 Gated Attention
attn = softmax(Q @ K^T / √d) @ V
output = attn * sigmoid(gate_proj(x))  # ★ 输出门
#         ↑ 注意力计算       ↑ 输出门控
```

### 输出门的作用

> 解决"attention sinks"（注意力沉没）问题 + 抑制巨大激活
>
> 在 Qwen2/Qwen3 中也出现过巨大激活值（outlier features），GA 的输出门让模型**自适应控制**每层 attention 的输出强度。

### QK zero-centered RMSNorm

```python
# 标准 RMSNorm
output = x / RMS(x) * weight
# 均值不是 0，训练时有偏移

# Zero-centered RMSNorm（Qwen3.5 用的）
output = x * (1 + weight) / RMS(x)
#       ↑ 加 1 让 weight 可以为负
#       ↑ 鼓励"零中心"分布
```

---

## 🌀 mRoPE（多模态 RoPE）

### 与标准 RoPE 对比

```python
# 标准 RoPE（1 维）
PE(pos) = [sin(pos·θ_0), cos(pos·θ_0), sin(pos·θ_1), cos(pos·θ_1), ...]

# mRoPE（3 维，多模态）
PE(t, h, w) = [
    sin(t·θ_0), cos(t·θ_0),    # 时间维 (temporal)
    sin(h·θ_1), cos(h·θ_1),    # 高度维 (height)
    sin(w·θ_2), cos(w·θ_2),    # 宽度维 (width)
    ... (重复)
]
```

### Qwen3.5 的 mRoPE 切片

```python
mrope_section: [11, 11, 10]  # 共 32 维
#              [temporal, height, width]
#              共 11+11+10 = 32 维（这就是 64 维 rotary 中的一半）
```

### mRoPE 解决什么问题

| 任务 | 位置编码需求 |
|------|------------|
| 文本 | 1 维（token 位置） |
| 图像 | 2 维（height × width） |
| 视频 | 3 维（time × height × width） |

mRoPE 用统一公式表达 1D/2D/3D 位置。

---

## 🧩 完整层级结构图

```
Input Tokens (B, L)
    │
    ▼
Embedding (B, L, 4096)
    │
    ▼
mRoPE Position Encoding
    │
    ▼
┌─[Layer 0] GDN → MoE──────────────────┐
│                                        │
│[Layer 1] GDN → MoE                    │
│[Layer 2] GDN → MoE                    │  Block 1
│[Layer 3] GA → MoE     ← ★ 每 4 层 1 个│
│                                        │  Gated Attention
├─[Layer 4] GDN → MoE──────────────────┤
│[Layer 5] GDN → MoE                    │
│[Layer 6] GDN → MoE                    │  Block 2
│[Layer 7] GA → MoE     ← ★             │
│                                        │
├─...                                   │  Block 3-14
│                                        │
├─[Layer 56] GDN → MoE─────────────────┤
│[Layer 57] GDN → MoE                   │
│[Layer 58] GDN → MoE                   │  Block 15
│[Layer 59] GA → MoE    ← ★            │
│                                        │
└────────────────────────────────────────┘
    │
    ▼
Final Norm
    │
    ▼
LM Head (B, L, vocab_size)
    │
    ▼
Logits
```

### Block 内部细节

每个 Block 内部：
```
Input (B, L, D)
    │
    ├── Norm
    ├── Attention (GDN 或 GA)
    ├── Residual Add
    │
    ├── Norm
    ├── MoE / FFN
    └── Residual Add
    │
    ▼
Output (B, L, D)
```

---

## 🔧 5 大关键算子

| 算子 | 作用 | 复杂度 | 关键优化 |
|------|------|--------|---------|
| **GDN** | 线性 attention | O(L·d²) | FLA / Triton kernel |
| **GA** | 标准 attention | O(L²·d) | Flash Attention 2/3 |
| **causal_conv1d** | GDN 前置局部卷积 | O(L·k·d) | cuDNN / Triton |
| **MoE Router** | 专家路由 | O(E·d) | AllToAll 通信 |
| **mRoPE** | 多模态位置编码 | O(L·d) | 预计算 cos/sin 表 |

### 算子性能关键点

1. **GDN 的 FLA 实现**（Flash Linear Attention）
   - 通过 chunking 把递推改成块计算
   - 利用 GPU 张量并行
   - 显存 O(L) 而不是 O(L²)

2. **causal_conv1d 的 Triton kernel**
   - 避免逐个 token 计算
   - 用 shared memory 缓存

3. **MoE 的 AllToAll 通信**
   - 把 token 路由到对应专家
   - 计算完再 gather 回来
   - 通信是主要瓶颈

---

## 🚀 与同类模型对比

| 模型 | 注意力 | 激活参数 | 上下文 | 发布时间 |
|------|--------|---------|--------|---------|
| **Qwen3.5-397B-A17B** ⭐ | GDN + GA (3:1) | 17B | 262K (1M ext) | 2026-02 |
| Kimi K2.5 | MLA + DSA | 32B | 200K | 2026-01 |
| GLM-5 | MLA + DSA | — | — | 2026-02 |
| DeepSeek V3 | MLA | 22B | 128K | 2024-12 |
| Llama 3.1 405B | Full MHA | 405B | 128K | 2024-07 |
| MiniMax M2.5 | Full MHA | 10B | 200K | 2026-02 |

### Qwen3.5 的差异化

- **唯一使用 GDN**（其他用 MLA 或 Full MHA）
- **极致稀疏**（< 5% 激活率）
- **原生多模态早期融合**（区别于 Qwen3-VL 的后期添加）
- **FP8 训练**（Qwen3 是 BF16/FP16）

---

## 💎 总结：Qwen3.5 的 3 个"为什么"

| 为什么 | 答 |
|--------|-----|
| **为什么用 GDN 替代大部分 attention？** | **长上下文效率**（O(L) vs O(L²)），8.6-19x 提速 |
| **为什么混合 GDN+GA 不全用 GDN？** | **精度保证**（标准 attention 抓复杂依赖更强），3:1 平衡效率与质量 |
| **为什么 MoE 这么稀疏（< 5%）？** | **推理成本**（17B 激活 vs 397B 总参数），部署成本降低 60% |

---

## 📖 参考资源

### 官方资源

- [Qwen 官方 GitHub](https://github.com/QwenLM/Qwen3.5)
- [Qwen3.5 发布博客](https://qwen.ai/blog?id=qwen3.5)
- [HF 模型页 Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [HF 模型页 Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- [HF 模型集合 Qwen3.5](https://huggingface.co/collections/Qwen/qwen35)

### 论文

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) - GDN 理论基础
- Mamba 系列论文（GDN 借鉴了 Mamba2 的门控机制）

### HF Transformers 源码

- `transformers/models/qwen3_5/modeling_qwen3_5.py` - 主模型 + GatedDeltaNet 实现
- `transformers/models/qwen3_5/configuration_qwen3_5.py` - 配置类
- `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` - MoE 版本
- `transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py` - MoE modular

### 中文技术博客

- [CSDN: Qwen3.5 混合注意力架构全解析](https://blog.csdn.net/tekin_cn/article/details/158773402)
- [HF Blog: Qwen3.5: Nobody Agrees on Attention Anymore](https://huggingface.co/blog/mlabonne/qwen35)

---

## 🎯 下一步计划

本文档是 **路径 2：Qwen3.5 + MindSpeed-MM 深度研究** 的第 1 份。

后续会输出：

| 序号 | 文档 | 内容 | 状态 |
|------|------|------|------|
| 01 | **架构总览**（本文） | Qwen3.5 整体架构 | ✅ |
| 02 | GDN 详解 | Gated DeltaNet 数学 + PyTorch 实现 | ⬜ |
| 03 | Gated Attention 源码 | QK Norm + Output Gate 完整实现 | ⬜ |
| 04 | mRoPE 详解 | 多模态位置编码原理 + 实现 | ⬜ |
| 05 | 极致稀疏 MoE 工程 | AllToAll + Token 调度 | ⬜ |
| 06 | MTP（多 Token 预测） | 训练时多预测几个 token | ⬜ |
| 07 | MindSpeed-MM 实现 | NPU 适配 + 性能优化 | ⬜ |
| 08 | Qwen3.5 vs Qwen3.6 | 演进对比 | ⬜ |

---

> **更新日志**
> - 2026-06-22: 初版整理（基于 Qwen 官方 GitHub + HF Transformers 源码 + 多份技术报告）
