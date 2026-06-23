# Gated DeltaNet (GDN) 详解

> **目标**：彻底吃透 GDN 的设计动机 + 数学原理 + 代码实现
> **附加用途**：周五给组员技术分享的素材库（可直接用）
> **前置阅读**：[01-架构总览](./01-qwen3.5-architecture-overview.md)
> **整理日期**：2026-06-22
> **预计演讲时长**：60 min（含 Q&A）

---

## 📚 目录

### 第一部分：演讲准备
1. [🎤 演讲 PPT 大纲](#-演讲-ptt-大纲-60-min)
2. [❓ Q&A 预期问题清单](#-qa-预期问题清单)

### 第二部分：核心内容
3. [🌊 引入：为什么需要 GDN？](#-引入为什么需要-gdn)
4. [📐 Delta Rule 数学基础](#-delta-rule-数学基础)
5. [🔐 Gated DeltaNet 完整数学](#-gated-deltanet-完整数学)
6. [💻 PyTorch 白板实现](#-pytorch-白板实现)
7. [🏛 HF Transformers 官方实现解析](#-hf-transformers-官方实现解析)
8. [⚡ FLA / causal_conv1d 工程优化](#-fla--causal_conv1d-工程优化)
9. [🎯 实验对比与适用场景](#-实验对比与适用场景)

### 第三部分：附录
10. [📖 推荐阅读 + 引用](#-推荐阅读--引用)

---

## 🎤 演讲 PPT 大纲（60 min）

> **可直接复制为 PPT 标题**。每个 PPT 标题后面写了"内容要点"+"建议时长"+"演讲提示"。

### Part 1: 引子（5 min）— 用故事抓住注意力

| # | PPT 标题 | 内容要点 | 时长 | 演讲提示 |
|---|---------|---------|------|---------|
| 1 | Qwen3.5 凭什么跑这么快？ | 32K 上下文 8.6x Qwen3-Max；256K 19x | 1 min | 抛悬念：不是参数更大，而是架构创新 |
| 2 | 一个反直觉的数据 | 397B 总参 / 17B 激活，**只用 5% 参数**做推理 | 1 min | 引出 MoE |
| 3 | 今天的主角：Gated DeltaNet | Qwen3.5 的 75% 层用这个 | 1 min | 标黄、放大字体 |
| 4 | GDN 是什么？ | 3 个关键词：线性 / 递推 / 门控 | 1 min | 用 3 个 PPT 翻页展示 |
| 5 | 为什么要发明它？ | 标准 attention O(L²) 撑不住 100K+ 上下文 | 1 min | 提个问题：你手机能跑 1M 上下文吗？ |

### Part 2: 数学原理（15 min）— 让大家真的懂

| # | PPT 标题 | 内容要点 | 时长 | 演讲提示 |
|---|---------|---------|------|---------|
| 6 | 标准 attention 的瓶颈 | 写公式：O = softmax(QK^T/√d)·V | 2 min | 突出"QK^T"是 L×L 矩阵 |
| 7 | 复杂度对比表 | L=128K 时，attention 需要 256 倍计算量 | 2 min | 用具体数字 |
| 8 | 线性注意力的核心思想 | 用核函数 φ 让 QK^T 拆开：φ(Q)(φ(K)^T·V) | 2 min | **核心洞察**：改变乘法顺序 |
| 9 | 线性注意力的 2 个问题 | (1) 表达力差 (2) 训练不稳定 | 2 min | 这是 Mamba 解决的问题 |
| 10 | Delta Rule 登场 | 借鉴数据库"增量更新"思想 | 2 min | 类比 git commit |
| 11 | GDN 的两个门控 | α（衰减）+ β（输入） | 2 min | 用动画演示 |
| 12 | GDN 的完整数学 | S_t = α·S_{t-1} + β·(k_t⊗v_t) | 1 min | 写在黑板 / 投影 |

### Part 3: 代码实现（20 min）— 现场跑代码

| # | PPT 标题 | 内容要点 | 时长 | 演讲提示 |
|---|---------|---------|------|---------|
| 13 | PyTorch 简化版（30 行） | 在白板 / Jupyter 现场写 | 5 min | 边写边讲 |
| 14 | 4 个关键参数 | linear_conv_kernel_dim / K 头 / V 头 / K-V 不对称 | 3 min | 列出 Qwen3.5 实际值 |
| 15 | HF Transformers 真实代码 | modeling_qwen3_5.py 精讲 | 5 min | 提前截图、逐行 walk |
| 16 | 训练 vs 推理的差异 | cache_params、conv_state、recurrent_state | 4 min | 画图讲解 |
| 17 | FLA 优化 | chunking 把递推变块计算 | 2 min | 性能数据 |
| 18 | causal_conv1d 的作用 | 局部特征提取 | 1 min | 类比 CNN 1D 卷积 |

### Part 4: 实战 + 总结（10 min）

| # | PPT 标题 | 内容要点 | 时长 | 演讲提示 |
|---|---------|---------|------|---------|
| 19 | 与同类架构对比 | Mamba2 / RetNet / GDN 性能表 | 3 min | 表格对比 |
| 20 | Qwen3.5 的整体架构 | 3 GDN + 1 GA（3:1 混合）| 2 min | 回到全景图 |
| 21 | GDN 的 3 大优势 | 速度 / 稳定性 / 长上下文 | 2 min | 总结 |
| 22 | GDN 的 3 大限制 | 表达力 / 训练成本 / 生态 | 2 min | 客观分析 |
| 23 | 总结一句话 | GDN = Mamba2 门控 + Delta Rule | 1 min | 强结束语 |

### Part 5: Q&A（10 min）

准备 Q&A 见下面 👇

---

## ❓ Q&A 预期问题清单

> 现场可能被问到的问题 + 建议回答

### Q1：GDN 和 Mamba 是什么关系？
**建议回答**：
> GDN 的"门控"部分借鉴 Mamba2（α = exp(g)），但状态更新用 Delta Rule 而非 SSM。  
> 一句话：**GDN = Mamba2 门控 + Delta Rule**

### Q2：GDN 和标准 attention 比，到底快多少？
**建议回答**：
> 复杂度：attention O(L²·d) vs GDN O(L·d²)  
> 实际数据：Qwen3.5 32K 上下文下，GDN 比 attention **快 8.6-19 倍**  
> 但准确率略低于 attention（这就是为什么 3:1 混合）

### Q3：为什么是 3:1 比例？不能 1:1 或 4:1？
**建议回答**：
> 3:1 是 Qwen Team 调出来的最优比例  
> 1:1 = 速度慢，4:1 = 准确率掉  
> 关键洞察：**GDN 抓"主要模式"，GA 抓"复杂依赖"**  
> 3:1 是效率与精度的 sweet spot

### Q4：GDN 能完全替代 attention 吗？
**建议回答**：
> **目前不能**。Qwen3.5 也保留 25% 标准 attention 层。  
> 原因：(1) 复杂推理任务 GDN 精度不够 (2) GDN 生态不成熟 (3) KV cache 优势仍存在  
> 未来如果 GDN 表现进一步提升，可能会更激进（比如 7:1 或纯 GDN）

### Q5：训练 GDN 有什么坑？
**建议回答**：
> (1) 数值稳定性：α 接近 0 或 1 都会爆，需初始化 A_log  
> (2) 因果卷积 padding 要正确  
> (3) chunk size 太小会慢，太大会爆显存  
> (4) FSDP 切分要小心 recurent state 的一致性

### Q6：NPU 上跑 GDN 跟 GPU 一样吗？
**建议回答**：
> **不一样**。NPU 用 AscendC 算子（`fla-npu`），GPU 用 Triton/CUDA  
> 主要差异：(1) 内存模型 (2) 通信原语 (3) 性能调优策略  
> MindSpeed-MM 仓库有专门的 NPU 适配代码

### Q7：GDN 跟 MoE 怎么配合？
**建议回答**：
> Qwen3.5 的设计是：  
> ```
> 每 4 层 = 3 × (GDN + MoE) + 1 × (GA + MoE)
> ```  
> GDN 处理长序列，MoE 提供稀疏专家。两者**正交**：GDN 是 attention 替代，MoE 是 FFN 替代。

### Q8：K/V 头数不一样（16 vs 64）是什么设计？
**建议回答**：
> 这是 GDN 的"非对称设计"，**只 Qwen3.5 / Qwen3-Next 这么干**  
> V 头更多 = 每个 V 头只负责更细粒度的语义（参数量大）  
> K 头更少 = 减少 query-key 匹配的复杂度（计算量小）  
> 实际效果：精度持平，训练速度快

---

## 🌊 引入：为什么需要 GDN？

### 标准 Attention 的 3 大瓶颈

假设一个上下文 L=128K（Qwen3.5 原生支持 262K）：

| 项 | 数值 | 问题 |
|----|------|------|
| Q·K^T 矩阵 | 128K × 128K = **160 亿**个浮点数 | 显存 OOM |
| 计算量 | O(L²·d) = O(128K²·4096) ≈ 2×10¹⁴ | 太慢 |
| 推理时延 | 32K 时每秒 ~38 token（Qwen3-Max） | 用户体验差 |

### 1M 上下文会怎样？

- Q·K^T 矩阵：1M × 1M = **1 万亿**个浮点数（FP16 = 2TB）→ 单卡根本装不下
- 计算量：比 128K 多 64 倍 → 单次 attention 要 30 分钟

**结论：标准 attention 无法支撑长上下文**。

### 解决思路：3 条路线

```
┌─────────────────────────────────────────────────┐
│ 解决长上下文问题                                 │
├─────────────────────────────────────────────────┤
│ (1) 稀疏 Attention  ← DeepSeek DSA, Mistral     │
│ (2) 线性 Attention  ← GDN, Mamba, RetNet        │
│ (3) 状态空间模型    ← Mamba1/2, RWKV, S4        │
└─────────────────────────────────────────────────┘
```

Qwen3.5 选了**路线 2**的 GDN。

---

## 📐 Delta Rule 数学基础

### 数据库视角的"增量更新"

```
想象一个数据库表，存储所有历史的"key-value"对：
   (k_1, v_1), (k_2, v_2), ..., (k_t, v_t)

Delta Rule 思想：
   "如果新的 k 跟某个历史 k 相似，就更新那行的 v，
    而不是在表里追加新行"
```

**为什么这样？**
- 数据库追加行会越来越慢
- 增量更新保持表大小不变 → 速度恒定

### 标准 attention：每次都"扫描全表"

```
对于 query q_t：
   1. 计算 q_t 跟所有历史 k_i 的相似度
   2. 加权求和所有 v_i
   → 复杂度 O(L)
```

### Delta Rule：只"更新相关行"

```
对于 query q_t：
   1. 找到最相关的历史行 i*（q_t · k_{i*} 最大）
   2. 只更新这一行：v_{i*} ← v_{i*} + (q_t · v_{t-1})
   → 复杂度 O(1) per token
```

**但这有个问题**：Delta Rule 假设只有一个相关行，实际可能有多个。

---

## 🔐 Gated DeltaNet 完整数学

### 状态递推公式

```
S_t = α_t · S_{t-1} + β_t · (k_t ⊗ v_t)    ← 状态更新
o_t = q_t · S_t                              ← 输出
```

其中：
- `S_t`：状态矩阵，shape `(d_k, d_v)`，存储"历史信息"
- `α_t`：衰减门控（scalar per head），防止状态无限增长
- `β_t`：输入门控（scalar per head），控制当前 token 的贡献
- `k_t ⊗ v_t`：outer product，把"k 对应什么 v"加到状态里
- `q_t · S_t`：matrix-vector multiply，输出当前 token 的表示

### 与标准 Attention 的对比

| 维度 | 标准 Attention | **GDN** |
|------|---------------|---------|
| 存储 | Q·K^T 矩阵（L×L）| 状态矩阵 S（d×d） |
| 复杂度 | O(L²·d) | O(L·d²) |
| 记忆机制 | 全局 softmax | 衰减累加 |
| 表达力 | 强（任意 token 都能 attend）| 中（受状态容量限制）|
| 训练稳定性 | 一般 | 较好（门控机制）|

### 两个门控的设计动机

**门控 1：α（衰减门控）**

```python
α = exp(g)        # g = -exp(A_log) 是可学习参数
                  # A_log 初始化为 0 → α ≈ 0.37
                  # 模型可以学 α 接近 0（快速遗忘）或接近 1（长期记忆）
```

**作用**：
- 防止状态矩阵 S 无限增长
- 模拟"人类记忆"——重要的事记很久，无关的事快速忘
- 解决 attention sinks 问题

**门控 2：β（输入门控）**

```python
β = sigmoid(b)    # b = linear(x) 由输入决定
                  # 输出范围 [0, 1]
```

**作用**：
- 控制当前 token 对状态的"贡献度"
- 输入不重要的 token，β 接近 0，状态几乎不变
- 解决了"巨大激活值"问题

### 与 Mamba2 的关系

```
Mamba2:  S_t = α·S_{t-1} + β·(k_t·v_t)       ← 元素级乘
GDN:     S_t = α·S_{t-1} + β·(k_t ⊗ v_t)     ← outer product
```

**核心差异**：
- Mamba2 用 **元素级乘**（Hadamard product），等价于把 v 加权累加
- GDN 用 **outer product**（k ⊗ v），把"k 决定往哪更新、v 决定更新什么"分开

**outer product 的好处**：
- 状态矩阵每个元素都有意义（k[i] 和 v[j] 的关联）
- 类似 attention 中的 score 矩阵
- 表达力比 Mamba2 强

### 完整数据流（一个 token）

```
输入 x_t (1, D)
    │
    ├──► k_proj → k_t (1, d_k)        ← K 头
    ├──► v_proj → v_t (1, d_v)        ← V 头
    ├──► q_proj → q_t (1, d_k)        ← Q 头
    │
    ├──► a_proj → a_t (1, 1)           ← α = exp(-exp(a_t))
    ├──► b_proj → b_t (1, 1)           ← β = sigmoid(b_t)
    │
    ├──► S_{t-1} = 状态矩阵 (d_k, d_v)
    │
    ├──► 状态更新: S_t = α·S_{t-1} + β·(k_t ⊗ v_t)   ← ★ 核心
    │
    ├──► 输出: o_t = q_t · S_t (1, d_v)
    │
    └──► out_proj → output (1, D)
```

---

## 💻 PyTorch 白板实现

> **目标**：写一个能跑通的最小版 GDN（不优化，便于理解）

### 完整代码（60 行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNet(nn.Module):
    """最小化 GDN 实现（教学版）"""
    
    def __init__(self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        
        # K/V 头数不同（K/V 不对称）
        assert num_v_heads % num_k_heads == 0
        self.head_group = num_v_heads // num_k_heads  # 1 个 K 头对应几个 V 头
        
        # 4 个投影
        self.q_proj = nn.Linear(hidden_size, num_k_heads * head_k_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_k_heads * head_k_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_v_heads * head_v_dim, bias=False)
        self.out_proj = nn.Linear(num_v_heads * head_v_dim, hidden_size, bias=False)
        
        # ★ 衰减参数 A_log（每个 K 头 1 个）
        # 初始化为 0 → α = exp(-exp(0)) = exp(-1) ≈ 0.37
        self.A_log = nn.Parameter(torch.zeros(num_k_heads))
        
        # ★ 输入门控
        self.b_proj = nn.Linear(hidden_size, num_k_heads, bias=False)
    
    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        
        # 1) 投影
        q = self.q_proj(x).view(B, L, self.num_k_heads, self.head_k_dim)  # (B, L, Hk, dk)
        k = self.k_proj(x).view(B, L, self.num_k_heads, self.head_k_dim)
        v = self.v_proj(x).view(B, L, self.num_v_heads, self.head_v_dim)  # (B, L, Hv, dv)
        b = self.b_proj(x)                                                # (B, L, Hk)
        
        # 2) 计算衰减门控 α (每个 K 头 1 个，broadcast 到所有 L)
        # alpha = exp(-exp(A_log))
        alpha = torch.exp(-torch.exp(self.A_log))                          # (Hk,)
        alpha = alpha.view(1, 1, self.num_k_heads, 1)                      # (1, 1, Hk, 1)
        
        # 3) 计算输入门控 β
        beta = torch.sigmoid(b)                                            # (B, L, Hk)
        beta = beta.unsqueeze(-1)                                          # (B, L, Hk, 1)
        
        # 4) ★ 状态递推（核心！）
        # 初始化状态 S_0 = 0
        S = torch.zeros(
            B, self.num_k_heads, self.head_k_dim, self.head_v_dim,
            device=x.device, dtype=x.dtype
        )  # (B, Hk, dk, dv)
        
        outputs = []
        for t in range(L):
            q_t = q[:, t, :, :]              # (B, Hk, dk)
            k_t = k[:, t, :, :]              # (B, Hk, dk)
            v_t = v[:, t, :, :]              # (B, Hv, dv)
            beta_t = beta[:, t, :, :]        # (B, Hk, 1)
            
            # 关键：把 V 头扩展到 K 头（一个 K 头对应多个 V 头）
            # (B, Hv, dv) → (B, Hk, head_group, dv) → (B, Hk, dv) 用 repeat
            v_t_grouped = v_t.view(B, self.num_k_heads, self.head_group, self.head_v_dim)
            v_t_grouped = v_t_grouped.mean(dim=2)  # 简化：取平均
            
            # 状态更新
            # S_t = alpha · S_{t-1} + beta · (k_t ⊗ v_t)
            kv_outer = k_t.unsqueeze(-1) * v_t_grouped.unsqueeze(-2)  # (B, Hk, dk, dv)
            S = alpha * S + beta_t.unsqueeze(-1) * kv_outer
            
            # 输出
            o_t = torch.einsum('bhdk,bhkd->bhd', q_t, S)                # (B, Hk, dv)
            outputs.append(o_t)
        
        output = torch.stack(outputs, dim=1)  # (B, L, Hk, dv)
        output = output.reshape(B, L, -1)      # (B, L, Hk*dv)
        output = self.out_proj(output)         # (B, L, D)
        
        return output


# ============== 测试 ==============
if __name__ == "__main__":
    B, L, D = 2, 16, 128
    num_k_heads, num_v_heads = 4, 8
    head_k_dim, head_v_dim = 16, 16
    
    model = GatedDeltaNet(D, num_k_heads, num_v_heads, head_k_dim, head_v_dim)
    x = torch.randn(B, L, D)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    # Input: torch.Size([2, 16, 128]) → Output: torch.Size([2, 16, 128])
```

### 关键代码解读

#### 1. K/V 头数不对称（重点）

```python
# Qwen3.5 397B-A17B: K 头 16, V 头 64
# 比例: 1 K 头对应 4 个 V 头
self.head_group = num_v_heads // num_k_heads  # = 4
```

**为什么这么设计？**
- K 负责"匹配"，计算量应该小
- V 负责"内容"，参数量应该大
- 1 个 K 头 = 4 个 V 头 → 节省 75% 的 K 计算

#### 2. 状态矩阵的物理意义

```python
# S shape: (B, Hk, dk, dv)
# 想象成 dk x dv 的"关联表"
# S[i, j] = "历史上 k 的第 i 维 和 v 的第 j 维 的关联强度"
```

#### 3. 因果性自动保证

```python
# 因为 S_t 只依赖 S_{t-1} 和当前 token
# 所以 token t 的输出只依赖 t 之前的信息
# 不需要额外 mask（相比标准 attention 简化了）
```

#### 4. α 初始化的数学含义

```python
# A_log 初始化为 0
# α = exp(-exp(0)) = exp(-1) ≈ 0.3679
# 含义：历史信息保留约 37%
# 模型可以学 A_log 变大（α 接近 0，遗忘更快）
# 或 A_log 变小（α 接近 1，长期记忆）
```

---

## 🏛 HF Transformers 官方实现解析

> **目标**：理解 Qwen3.5 真实工程代码的关键设计

### 关键文件

`transformers/models/qwen3_5/modeling_qwen3_5.py` 中：
- 类 `Qwen3_5GatedDeltaNet` — 主类
- `causal_conv1d` — 因果卷积（前置）
- `fla_chunk_gated_delta_rule` — FLA 优化的核心

### 简化版真实代码（去优化版）

```python
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads      # 16 (397B)
        self.num_v_heads = config.linear_num_value_heads    # 64 (397B)
        self.head_k_dim = config.linear_key_head_dim        # 128
        self.head_v_dim = config.linear_value_head_dim      # 128
        self.conv_kernel_size = config.linear_conv_kernel_dim  # 4
        
        # 关键参数
        self.head_group = self.num_v_heads // self.num_k_heads  # 4
        
        # Q/K/V/β 一次性投影（节省计算）
        qkv_dim = self.num_k_heads * self.head_k_dim * 2 + self.num_v_heads * self.head_v_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, qkv_dim, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(
            self.num_v_heads * self.head_v_dim, self.hidden_size, bias=False
        )
        
        # 衰减参数
        self.A_log = nn.Parameter(torch.zeros(self.num_k_heads))
        
        # β 门控投影（已经合并到 in_proj_qkv 中）
        # dt_bias 用来调节 β 的偏置
        self.dt_bias = nn.Parameter(torch.zeros(self.num_k_heads))
    
    def forward(self, hidden_states, cache_params=None, cache_position=None):
        B, L, D = hidden_states.shape
        
        # 1) 一次性 QKV 投影
        qkv = self.in_proj_qkv(hidden_states)  # (B, L, qkv_dim)
        
        # 2) ★ 因果卷积（Q/K/V 共享局部上下文）
        if cache_params is not None:
            # 推理时：用缓存的 conv state
            conv_state = cache_params.conv_states[self.layer_idx]
            qkv = causal_conv1d_update(qkv, conv_state, ...)
        else:
            # 训练时：完整因果卷积
            qkv = causal_conv1d(qkv, conv_weight, kernel_size=4)
        
        # 3) 拆分 Q/K/V/β
        q, k, v, beta = self._split_qkv(qkv)  # 4 个 tensor
        # q: (B, L, Hk, dk)
        # k: (B, L, Hk, dk)
        # v: (B, L, Hv, dv)
        # beta: (B, L, Hk)
        
        # 4) reshape 成多头
        q = q.transpose(1, 2)  # (B, Hk, L, dk)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5) ★ 核心：FLA 优化的 Delta Rule 计算
        if cache_params is not None and cache_position[0] > 0:
            # 推理：增量更新
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q, k, v, beta=beta.sigmoid() + self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=True,
            )
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
        else:
            # 训练：chunked FLA
            o = fla_chunk_gated_delta_rule(
                q, k, v, beta=beta.sigmoid() + self.dt_bias,
                initial_state=None,  # 训练时无状态
            )
        
        # 6) 输出 reshape + 投影
        o = o.transpose(1, 2).contiguous()  # (B, L, Hv, dv)
        o = o.reshape(B, L, -1)
        o = self.out_proj(o)
        
        return o
    
    def _split_qkv(self, qkv):
        """把 QKV+beta 拆开"""
        # 假设 dims = [Hk*dk, Hk*dk, Hv*dv, Hk]
        k_dim = self.num_k_heads * self.head_k_dim
        v_dim = self.num_v_heads * self.head_v_dim
        q = qkv[..., :k_dim]
        k = qkv[..., k_dim:2*k_dim]
        v = qkv[..., 2*k_dim:2*k_dim+v_dim]
        beta = qkv[..., 2*k_dim+v_dim:]
        return q, k, v, beta
```

### 关键工程设计

#### 1. QKV 一次性投影（性能关键）

```python
# Karpathy nanoGPT 风格：1 个 Linear 输出 QKV
self.in_proj_qkv = nn.Linear(hidden_size, qkv_dim)

# 比分开 4 个 Linear 快 30%+（GPU kernel launch 开销）
```

#### 2. 因果卷积在前（关键预处理）

```python
# causal_conv1d(kernel_size=4)
# 让 Q/K/V 共享"局部上下文"（最近 4 个 token）
# 弥补 GDN 缺乏"局部建模"的弱点
```

**为什么需要？**
- GDN 状态递推擅长"长期依赖"
- 但对"最近几个 token"不够敏感
- 用 1D 卷积补这个能力

#### 3. 训练 vs 推理路径分离

```python
# 训练：fla_chunk_gated_delta_rule
#   - 整段序列并行计算
#   - 用 chunking 把递推改成矩阵运算
#   - 显存 O(L) 不 O(L²)

# 推理：fused_recurrent_gated_delta_rule  
#   - 增量更新（每次只处理 1 个 token）
#   - 用 recurrent_state 缓存历史
#   - 显存 O(1) per step
```

#### 4. β 偏置 dt_bias

```python
# β = sigmoid(linear(x)) + dt_bias
# dt_bias 让 β 可以超出 [0, 1] 范围
# 默认初始化为 0 → β ∈ [0, 1]
# 训练时 dt_bias 可以被优化
```

---

## ⚡ FLA / causal_conv1d 工程优化

### 什么是 FLA？

**FLA = Flash Linear Attention**（[flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)）

> 把 GDN 的"逐 token 递推"改成"块状并行计算"

### 为什么需要 FLA？

```python
# 朴素 GDN（前面 PyTorch 实现）
for t in range(L):
    S = alpha * S + beta * (k_t ⊗ v_t)
    output[t] = q_t @ S

# 问题：循环 L 次，GPU 没用上
# 训练 1 个 60 层 × 100K token 的模型
# = 600 万次循环 = 几天
```

### FLA 的核心思想：Chunking

把 L 个 token 切成多个 chunk（如 chunk_size=64）：

```
Token 序列: [t0, t1, ..., t63, t64, ..., t127, ...]
                ↓
Chunk 0:    [t0 ~ t63]      ← 块内并行计算
Chunk 1:    [t64 ~ t127]    ← 块内并行计算
...                             块间串行（但可以流水）

# 关键：块内用矩阵乘（GPU 友好）
# 块间用递推（保留因果性）
```

### FLA 的 3 大优势

| 优势 | 数值 |
|------|------|
| **训练速度** | 比朴素 GDN **快 5-10x** |
| **显存占用** | O(L) 不 O(L²) |
| **支持 Backward** | 通过重计算策略 |

### causal_conv1d 的作用

```python
# 等价于：
# output[t] = sum_{i=0}^{kernel_size-1} weight[i] * input[t-i] (if t-i >= 0)
# 即：每个 token 看前面 kernel_size 个 token

# kernel_size=4 的卷积核
# 让 Q/K/V 共享 4 个 token 的局部上下文
```

**为什么 GDN 需要 causal_conv1d？**
- GDN 状态递推擅长"长期模式"
- 但对"最近 4 个 token" 的关系不够敏感
- causal_conv1d 补这个能力
- 类似 CNN 早期层的作用

### NPU 上的实现差异

| 项 | GPU（CUDA） | NPU（Ascend） |
|----|------------|--------------|
| 算子库 | Triton / FLA | AscendC / fla-npu |
| 内存模型 | HBM 统一 | HBM + L2 Cache 分层 |
| 通信 | NCCL | HCCL |
| 优化策略 | Tensor Core | Cube / Vector Unit |

**MindSpeed-MM 仓库**：
- 用了 `fla-npu` 库（NPU 版本的 FLA）
- 见 [README](https://github.com/mindspore-lab/MindSpeed-MM) 提到的 `flash-linear-attention-npu`

---

## 🎯 实验对比与适用场景

### 与同类架构对比

| 架构 | 模型 | 复杂度 | 表达力 | 长上下文 | 训练稳定性 | 推理速度 |
|------|------|--------|--------|---------|----------|---------|
| **Standard Attn** | Llama 3 | O(L²·d) | 高 | 差 | 中 | 慢 |
| **Mamba2** | Jamba | O(L·d²) | 中 | 好 | 中 | 快 |
| **RetNet** | Yandex | O(L·d²) | 中 | 好 | 好 | 快 |
| **GDN** | Qwen3.5 | O(L·d²) | 中高 | 好 | 好 | **最快** |
| **MLA** | DeepSeek V3 | O(L²·d) 压缩 | 高 | 中 | 好 | 中 |
| **DSA** | GLM-5 | O(L²·d) 稀疏 | 高 | 中 | 好 | 中 |

### GDN 的 3 大优势

| 优势 | 说明 |
|------|------|
| **速度** | 32K 上下文 8.6x Qwen3-Max |
| **稳定性** | 门控机制解决 attention sinks |
| **长上下文** | 原生支持 262K，扩展 1M |

### GDN 的 3 大限制

| 限制 | 说明 |
|------|------|
| **表达力** | 比标准 attention 弱（需混合） |
| **生态** | 库不成熟（FLA / fla-npu 还在演进） |
| **调优难** | A_log / dt_bias 等参数需精调 |

### 适用场景

| ✅ 适合 | ❌ 不适合 |
|--------|---------|
| 长上下文（100K+） | 短文本（< 4K） |
| 推理密集任务 | 精确模式匹配 |
| 移动端 / 边缘部署 | 训练小模型（不划算） |
| 高吞吐推理 | 需要强 KV cache 的场景 |

---

## 📖 推荐阅读 + 引用

### 论文

1. **[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)** ⭐⭐⭐
   - GDN 理论基础
   - 必读，建议完整读一遍

2. **[Mamba2: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2412.08852)**
   - GDN 借鉴的对象
   - 重点读"门控机制"部分

3. **[Qwen3-Next Technical Report](https://qwen.ai/blog?id=qwen3-next)**
   - Qwen3.5 的技术祖先
   - 详细解释为什么 3:1 混合

### 代码库

- [flash-linear-attention (FLA)](https://github.com/sustcsonglin/flash-linear-attention) — GDN 高效实现
- [Qwen3.5 官方 GitHub](https://github.com/QwenLM/Qwen3.5) — 完整模型仓库
- [HF Transformers Qwen3.5 源码](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5)

### 中文博客

- [CSDN: Qwen3.5 混合注意力架构全解析](https://blog.csdn.net/tekin_cn/article/details/158773402)
- [HF Blog: Qwen3.5: Nobody Agrees on Attention Anymore](https://huggingface.co/blog/mlabonne/qwen35)

### 演讲引用

```bibtex
@misc{qwen3.5,
    title  = {{Qwen3.5}: Towards Native Multimodal Agents},
    author = {{Qwen Team}},
    year   = {2026},
    month  = {February},
    url    = {https://qwen.ai/blog?id=qwen3.5}
}

@misc{yang2024gated,
      title={Gated Delta Networks: Improving Mamba2 with Delta Rule}, 
      author={Songlin Yang and Jan Kautz and Ali Hatamizadeh},
      year={2024},
      eprint={2412.06464},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

---

## 🎯 演讲收尾（讲给组员时用）

> **开场引入**（1 min）：
> "今天讲 Qwen3.5 为什么跑这么快。答案是 75% 的层不用标准 attention 了，用一个叫 GDN 的新东西。"

> **核心要点**（3 句话）：
> 1. GDN = Mamba2 门控 + Delta Rule
> 2. 状态递推 O(L·d²) vs attention O(L²·d)
> 3. 两个门控解决 attention sinks + 巨大激活

> **结束语**（1 min）：
> "GDN 不是银弹。Qwen3.5 还是保留了 25% 标准 attention 来抓复杂依赖。3:1 是目前最优的效率-精度 sweet spot。这条路还很长，未来怎么走，大家可以一起想想。"

---

> **更新日志**
> - 2026-06-22: 初版（深度版，含演讲 PPT 大纲 + Q&A 准备）
