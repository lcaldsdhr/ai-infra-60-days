# GDN 深度：从 Mamba 2 到 GDN

> 要真正理解 Qwen3.5 的核心创新 GDN（Gated DeltaNet），需要从 Mamba 2 开始讲起。本文按技术演进路径讲解 GDN：
> - **Mamba 2**（2024.5）用 SSD 理论统一 SSM 和 Attention，提供**门控 α**（时间维度遗忘）
> - **Delta Rule**（Anderson 1961 / DeltaNet 2024）用"错误修正"实现**强关联记忆**，引入**输入门 β**
> - **GDN = 两者结合** = "既跟时间走（α），又跟错误走（β）"

---

## 速览

**GDN = Mamba 2 门控 + Delta Rule**

读完本文，你将能：
1. 解释 Mamba 2 的 SSD 理论为什么重要
2. 写出 Delta Rule 的公式并解释其优势
3. 推导 GDN 完整公式
4. 解释 α 和 β 各自的作用

---

## 前置知识

### 1. 矩阵乘法的结合律

```
(A · B) · C = A · (B · C)
```

这就是 GDN 的核心魔法。普通 attention 算 `(Q·K^T)·V`（先形成 L×L），GDN 算 `Q·(K^T·V)`（先形成 d×d）。

### 2. Attention 公式回顾

```
O = (Q · K^T) · V
```

- Q, K, V 都是 [L, d] 形状
- Q·K^T 形成 [L, L] 矩阵（**L² 个格子**）
- 这是 GDN 要解决的核心问题

### 3. 门控（Gating）概念

门控 = "通过一个可学习的 0~1 之间的数来控制信息流"。类比：手机音量旋钮（0=静音，1=最大）。

---

## 第 1 章：Mamba 2 是什么

### 1.1 状态空间模型（SSM）

SSM 是控制论的经典模型，1960 年代就有：

```
h_t = A · h_{t-1} + B · x_t    ← 状态递推
y_t = C · h_t                   ← 输出
```

| 符号 | 含义 |
|------|------|
| `h_t` | 时刻 t 的隐藏状态（**d 维向量**）|
| `x_t` | 时刻 t 的输入 |
| `y_t` | 时刻 t 的输出 |
| `A, B, C` | 系统矩阵（参数）|

**对比 RNN**：
- RNN: `h_t = f(W_h · h_{t-1} + W_x · x_t)`（稠密，难并行）
- SSM: `h_t = A · h_{t-1} + B · x_t`（**A 对角**，并行友好）

### 1.2 Mamba 1（2023.12）的突破

**核心创新**：让 B、C 矩阵**依赖输入**（之前是固定的）：

```
B_t, C_t = f(x_t)    ← 输入相关
h_t = A · h_{t-1} + B_t · x_t
y_t = C_t · h_t
```

**问题**：B, C 每步都变，**没法预计算 → 没法并行**，需要自定义 CUDA kernel。

### 1.3 Mamba 2（2024.5）的 SSD 理论

**核心洞察**（State Space Duality, SSD）：**SSM 和 Attention 是同一类东西的对偶**。

证明过程（简化）：
```
SSM 输出:    y_t = C · h_t,  h_t = A·h_{t-1} + B·x_t
展开成矩阵:  Y = (Q·K^T)·V 形式！

关键洞察：
- Attention 算 S = K^T·V 在每个位置都重算 → O(L²d)
- SSM 增量更新 S → O(Ld²)
- 两者数学等价但实现完全不同！
```

Mamba 2 用 **4 个关键改进**实现 5-10x 加速：

| 改进 | 作用 |
|------|------|
| **结构化 S4** | A 矩阵用对角/低秩结构 |
| **状态矩阵化** | 把 d 维向量 h_t 升级为 d×d 矩阵 S |
| **SSD 并行** | SSM 算子可用**矩阵乘法**实现 |
| **门控 α** | 引入时间维度遗忘（见下）|

### 1.4 Mamba 2 的门控 α（关键！）

```
h_t = α · h_{t-1} + B_t · x_t    ← α 控制历史保留比例
y_t = C_t · h_t
```

`α` 用 `exp(-exp(A_log))` 计算：
- A_log = 0（**官方初始值**）→ α ≈ **0.37**（保留 37% 历史）
- A_log 越大 → α 越小（**快速遗忘**）
- A_log 越小 → α 越大（**长期记忆**）

**α 的核心作用**：让模型自己学"该记住多久"。每个 head 独立学 α（不同头不同遗忘速度）。

### 1.5 Mamba 2 的局限

Mamba 2 的状态是 d 维向量，**关联记忆能力弱**——它能记住"最近发生什么"，但记不住"谁跟谁相关"。

这正是 Delta Rule 要补的短板。

---

## 第 2 章：Delta Rule 是什么

### 2.1 起源：线性关联记忆

Delta Rule 来自**线性关联记忆**（Linear Associative Memory）理论，1961 年 Anderson 提出，1971 年 Kohonen 也独立提出。

**核心问题**：给定一对 (k_t, v_t)，怎么更新"记忆矩阵 S"使得 `S · k_t = v_t`？

### 2.2 朴素做法：Hebbian Learning

```
S_new = S_old + k_t · v_t^T    ← 直接加
```

**问题**：S 会无限累加，"过度记忆"，跟新数据冲突。

### 2.3 Delta Rule 的智慧

**只更新"错误"部分**：

```
err = v_t - S_old · k_t                ← 算错误
S_new = S_old + β · k_t · err^T        ← 用错误修正
S_new = S_old + β · k_t · (v_t - S·k_t)^T
```

**展开**：

```
S_new = S_old + β·k·v^T - β·(S·k)·k^T
      = (I - β·k·k^T) · S_old + β·k·v^T
```

### 2.4 Hebbian vs Delta 对比

| 维度 | Hebbian | Delta Rule |
|------|---------|-----------|
| 公式 | `S + k·v^T` | `S + β·k·(v - S·k)^T` |
| 累加 | 无限累加 | **减去 S·k 项，不会爆炸** |
| 关联记忆 | 弱 | **强**（用错误修正）|
| 收敛性 | 不保证 | **保证 S·k → v** |
| 类比 | 傻瓜（"记住所有事"）| **聪明学生（"错的才要改"）**|

### 2.5 DeltaNet 2024 的现代形式

Yang 等人 2024 年的 DeltaNet 论文把 Delta Rule 写成**线性 attention 形式**：

```
S_t = S_{t-1} + β_t · k_t · (v_t - S_{t-1}·k_t)^T
o_t = q_t · S_t
```

证明 DeltaNet 的 S 矩阵有**最优子空间性质**——只记住最相关的关联。

### 2.6 DeltaNet 的局限

DeltaNet **不遗忘**——所有历史都进 S，长序列会"过载"。这正是 Mamba 2 门控 α 要补的短板。

---

## 第 3 章：GDN = Mamba 2 + Delta Rule

### 3.1 GDN 完整公式（首秀）

```
S_t = α · S_{t-1} + β_t · k_t · (v_t - S_{t-1}·k_t)^T
o_t = q_t · S_t
```

### 3.2 拆解：Mamba 2 + Delta 各负责什么

```
S_t =  [α · S_{t-1}]              +  [β_t · k_t · (v_t - S_{t-1}·k_t)^T]
       ↑ Mamba 2 门控                ↑ Delta Rule 修正
       控制历史遗忘速度                控制关联记忆修正
       时间维度（α）                  关联维度（β）
```

### 3.3 展开成矩阵形式

```
S_t = α · S_{t-1}
    + β_t · k_t · v_t^T            ← Delta 新关联
    - β_t · (S_{t-1}·k_t) · k_t^T  ← Delta 减去旧错误

  = (α·I - β_t·k_t·k_t^T) · S_{t-1}
    + β_t · k_t · v_t^T
```

**关键洞察**：`α·I - β·k·k^T` 是**自适应的衰减矩阵**——
- `α·I` 是**时间维度**的均匀衰减
- `β·k·k^T` 是**key 维度**的正交化（让新 key 跟旧 key 方向正交时不被衰减）

两者组合 = "既按时序衰减，又按 key 相关性修正"。

### 3.4 对比：三种架构的公式

| 架构 | 公式 | 状态 | 遗忘 | 关联 |
|------|------|------|------|------|
| **纯 Mamba 2 SSM** | `h_t = α·h + B·x` | d 维向量 | ✅ α | ❌ 弱 |
| **纯 Delta Rule** | `S = S + β·k·(v-S·k)^T` | d×d 矩阵 | ❌ 无 | ✅ 强 |
| **GDN** | `S = α·S + β·k·(v-S·k)^T` | **d×d 矩阵** | **✅ α** | **✅ β** |

**GDN 用 d×d 矩阵（不是 d 维向量），同时拥有 Mamba 2 的遗忘和 Delta 的关联**。

### 3.5 GDN 输出（跟 Attention 一致）

```
o_t = q_t · S_t
```

Q 查表 S 拿到输出，**跟 Attention 的 `O = Q·K^T·V` 数学等价**（结合律），但计算复杂度 O(Ld²) 远低于 O(L²d)。

---

## 第 4 章：α 和 β 详解

### 4.1 官方初始值

| 参数 | 公式 | 初始值 | 含义 |
|------|------|--------|------|
| **α** | `exp(-exp(A_log))` | A_log=0 → **α ≈ 0.37** | 衰减门（保留 37% 历史）|
| **β** | `sigmoid(linear(x))` | linear=0 → **β = 0.5** | 输入门（50% 注入）|

A_log 和 linear 都初始化为 0，**不需要 warm-up**。

### 4.2 α 和 β 的协同

两者数学上独立，但训练中**自动学到协同**：

| α 趋势 | β 趋势 | 模型行为 |
|--------|--------|---------|
| α 小（快遗忘）| β 大（强注入）| 短时关注当下信息 |
| α 大（慢遗忘）| β 小（弱注入）| 长期关注历史模式 |
| α ≈ β | 中等 | 平衡模式 |

### 4.3 调参经验

- 实际训练中 α 和 β 的稳定区间都在 **[0.1, 0.9]**
- 极端值（< 0.1 或 > 0.9）会导致训练不稳定
- 每个 head 独立学 α（不同头不同遗忘速度）
- β 跟输入相关（不同 token 不同 β），更灵活

### 4.4 跟 GRU/LSTM 的门控对比

| 模型 | 门控 | 维度 | 来源 |
|------|------|------|------|
| **GRU** | reset + update | 2 个标量门 | 跟输入相关 |
| **LSTM** | forget + input + output | 3 个标量门 | 跟输入相关 |
| **GDN α** | 衰减门 | **1 个标量**（per head）| **不依赖输入** |
| **GDN β** | 输入门 | 1 个标量 | 跟输入相关 |

**GDN 简化了**：α 不依赖输入（每个 head 一个固定值），β 依赖输入（每个 token 一个值）。

---

## 第 5 章：三视角理解 S

### 5.1 视角 1：S 是聚合统计量（不是档案库）

S 存的**不是**"每个 token 长什么样"，而是**两两之间的关联强度**。

类比：
- **KV Cache** = 录音机（逐字完整记录 L 个 token）
- **GDN 的 S** = 速记本（只记"谁跟谁关联强"这个模式）

S 是 lossy 压缩，但**够用**——attention 真正需要的就是关联模式。

### 5.2 视角 2：S 是流动的关联池

S 永远只装 d² 个数字——L 再长，旧关联也会被 α 自动稀释/覆盖。

数字对比（Qwen3.5 397B-A17B, d=4096）：

| 项 | 容量 | 128K 时 |
|----|------|---------|
| KV Cache | L × d | 128K × 4096 = 5 亿 |
| GDN 的 S | d² | 4096² = 1600 万 |
| **差距** | | **约 30 倍** |

为什么压缩"够用"？因为 attention 真正需要的是"谁跟谁相关"，**不是"每个 token 的完整坐标"**。

### 5.3 视角 3：α 和 β 是"记多久、记多深"的可学习开关

| 项 | 作用 |
|---|---|
| `α·S_{t-1}` | 旧关联保留比例（α 接近 1 = 长期记忆，接近 0 = 短时记忆）|
| `β·(k_t ⊗ v_t)` | 新关联注入强度（β 接近 1 = 完全进 S，接近 0 = 完全不进）|

**α 小的时候 β 大，反之亦然**——模型通过训练学到"记多久 vs 记多深"的协同。

### 5.4 一句话总结（实战公式版）

> **S 是 d×d 的"压缩笔记"**（存关联模式），**Q 是"查询问题"**（这次要哪些关联），**Q·S → O** 是"按需查表拿到答案"。

回答常见疑问："S 只有 d×d，怎么装得下 L 个 token 的信息？" → **S 装的是聚合关联，不是完整记录**。d² 是付得起的常数，L 是付不起的变量——GDN 选择保留前者。

---

## 第 6 章：Qwen3.5 中的 GDN

### 6.1 397B-A17B 的 GDN 配置

| 项 | 值 | config 字段 |
|---|----|----|
| Hidden dim | 4096 | `hidden_size` |
| GDN V 头数 | 64 | `linear_num_value_heads` |
| GDN QK 头数 | 16 | `linear_num_key_heads` |
| GDN 头维 | 128 | `linear_value_head_dim` |
| S 矩阵大小 | 64 × 128² = **8 MB/层** | d×d per head |
| 全部 S（60 层）| **480 MB** | 60 × 8 MB |

### 6.2 3:1 混合架构

Qwen3.5 397B-A17B 总共 **60 层** = **15 组 × 4 层**。每组结构：

```
[GDN, GDN, GDN, GA]   ← 3:1 比例，循环 15 次
```

- **75% 层是 GDN**（45 层），处理大部分 token
- **25% 层是 GA**（15 层），抓 GDN 处理不好的复杂依赖

最后 1 层（layer 59）总是 GA，意味着 **GA 是"收尾层"**——保证最深层有最强的全局注意力。

### 6.3 性能数据（论文实测）

| 模型 | 注意力机制 | 32K 速度 | 256K 速度 |
|------|----------|---------|----------|
| Qwen3-Max | 标准 Attention | 1x | 1x |
| Qwen3.5 | GDN + GA (3:1) | **8.6x** | **19x** |
| Llama 3.1 405B | Full MHA | 较慢 | 不支持 |

数据来源：[Qwen3.5 发布博客](https://qwen.ai/blog?id=qwen3.5) / [GDN 论文](https://arxiv.org/abs/2412.06464)

### 6.4 为什么 3:1 比例

Qwen Team 调出来的**效率-精度 sweet spot**：
- 100% GDN：精度不够（关联记忆可能丢失）
- 100% GA：长上下文慢（L² 爆炸）
- 3:1 混合：平衡效率和精度

### 6.5 GDN 单独运行的成本

| 操作 | 显存 / 算力 |
|------|------------|
| 训练时 S | 60 层 × 8 MB = **480 MB**（小！）|
| 推理时 S | 同上，作为"超级 KV cache" |
| 增量更新 | 每次 1 token 算 d² = 16M FLOPs（常数）|
| 查表 | 每次 1 token 算 L·d² = 512M FLOPs（线性）|

### 6.6 非对称 QKV 设计：借鉴 GQA 但更激进

#### 6.6.1 数字对比

Qwen3.5 GDN 的实际配置（来自 [Qwen3_5GatedDeltaNet](file:///c:/Users/Administrator/Desktop/code/0611/MindSpeed-MM/mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py#L519-L529)）：

```python
linear_num_value_heads = 64    # V 头：64 个（满）
linear_num_key_heads   = 16    # K 头：16 个（压缩 4×）
head_k_dim = 128
head_v_dim = 128

key_dim   = 16 × 128 = 2048
value_dim = 64 × 128 = 8192
conv_dim  = 2*K + V = 12288   # Conv1d 输入维度
```

Q 与 K 共用 `key_dim`，所以 **Q : K : V = 1 : 1 : 4**。

| 模型 | Q 头 | K 头 | V 头 | 比值 |
|---|---|---|---|---|
| **MHA**（标准 Attention） | 64 | 64 | 64 | 1:1:1 |
| **GQA**（Attention） | 64 | 4 | 4 | 16:1:1 |
| **GDN**（Qwen3.5） | 16 | 16 | 64 | **1:1:4** |

**GQA 只压缩 KV、保留 Q**；**GDN 反过来——保留 V、压缩 QK**。

#### 6.6.2 为什么这样设计（4 大动机）

**动机 1：减小 in_proj_qkv 的权重**

```python
self.in_proj_qkv = nn.Linear(hidden_size, 2*K + V)
# Qwen3.5:  Linear(5120, 12288)  ← 当前
# 对称设计:  Linear(5120, 24576)  ← 2 倍
```

非对称设计让 QKV 投影的权重参数和 FLOPs **减半**。

**动机 2：减小 Conv1d 的计算量**

```python
self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=4, groups=conv_dim)
```

`groups=conv_dim` 是 depthwise conv，每个通道独立：
- 对称：24576 通道 × 4 kernel = 98304 次乘
- 非对称：**12288 通道 × 4 kernel = 49152 次乘**（**减半**）

**动机 3：减小 Recurrent State 显存**

GDN 的核心状态 S 形状 `[v_heads, head_k_dim, head_v_dim]`：
- 64 层 × 64 V 头 × 128 × 128 × 2B = **128 MB / 每 sequence**
- 推理时这个 state 必须常驻每层
- 压缩 K 头 → K⊗V 的外积被 4 个 V 头共享，**state 更新计算量降为 1/4**

**动机 4：保持 V 的表达能力（最关键）**

```
V 头 = "内容载体"，每个头负责一类语义信息
压缩 V → 损害模型能力
压缩 K → K 只参与"查表"（外积键），影响小
压缩 Q → Q 只参与"读状态"（query），影响小
```

V 必须满血，因为：
- 状态 S 的"列"维度 = head_v_dim，**V 决定状态能存什么**
- 输出 `o = q · S` 矩阵乘，V 维度直接决定输出丰富度
- 16 个 V 头 vs 64 个 V 头，模型质量肉眼可见下降

#### 6.6.3 压缩 QK 的实际影响

**正向收益**：
- ✅ in_proj_qkv 权重减半（5120×12288 vs 5120×24576）
- ✅ Conv1d 计算减半
- ✅ K^T·V 矩阵乘的 K 维度从 8192→2048（chunk 内矩阵乘块计算量降为 1/4）
- ✅ Recurrent state 更新量减少

**潜在代价**：
- ⚠️ 同一组的 4 个 V 头共享 K，**键空间冲突**（不同内容可能算出相似的 K）
- ⚠️ 这就是为什么 Q 也不全压（保留 16 个独立 K），而是用 4:1 平衡

#### 6.6.4 与 Attention GQA 的关键差异

```
GQA 设计哲学（Attention）：
  "Q 头各管各的，K/V 共享节省 KV cache"
  → 节省的是推理时的 KV cache（cache 大小 ∝ seq_len）

GDN 非对称设计哲学（Linear Attention）：
  "V 头各管各的，Q/K 共享节省权重和计算"
  → 节省的是权重参数量和 forward FLOPs（不随 seq_len 变）
```

- **Attention 的痛点**是 KV cache 随 seq_len 线性增长（生成 128K token 时 KV cache 巨大）
- **Linear Attention 的痛点**是 in_proj_qkv 权重 + state 更新（不随 seq_len 变，但权重常驻）

所以 GDN 把优化点放在了**权重侧**而非 cache 侧。

#### 6.6.5 与其他 Linear Attention 模型对比

| 模型 | Q 头 | K 头 | V 头 | 非对称? |
|---|---|---|---|---|
| **Mamba 2** | 80 | 80 | 80 | 对称 |
| **Gated DeltaNet（基线）** | 16 | 16 | 32 | 2:1 |
| **Qwen3.5 GDN** | 16 | 16 | 64 | **4:1（最激进）** |
| **Jamba / RWKV** | — | — | — | 不同方案 |

Qwen3.5 的 4:1 在同类 linear attention 中是**最激进的非对称设计**。

#### 6.6.6 一句话总结

> Qwen3.5 GDN 的"4:1 非对称 QKV"是 **GQA 思想在 Linear Attention 上的变体**——保留 V 头（内容表达）而压缩 Q/K 头（键空间共享），省的是 in_proj_qkv 权重、Conv1d 计算和 state 更新代价，本质是用"V 满血"换"K 共享"的内存-表达权衡。

---

## 第 7 章：FLA 优化

### 7.1 朴素 GDN 的问题

朴素 GDN 算 d² 是**逐 token 串行**（每步 1 个 [d, d] 小矩阵加法），GPU 利用率 < 20%。

### 7.2 FLA 怎么解决

**FLA**（Flash Linear Attention）通过 **chunking** 把串行递推改为块状并行：

```
朴素：64 次串行 [d, d] 小矩阵加法
FLA：1 次 [chunk, d] · [d, chunk] 大矩阵乘
     + 1 次 [d, chunk] · [chunk, d] 大矩阵乘
     → 1 次算出 S 增量
```

**chunking 步骤**：
1. 把 L 个 token 切成多个 chunk（典型大小 64）
2. chunk 内用矩阵乘并行（**走 tensor core**）
3. chunk 间保留串行依赖（因果性）

GPU 利用率从 < 20% 提升到 > 80%，训练速度 5-10x。

### 7.3 FLA 实现

[FLA 官方库](https://github.com/sustcsonglin/flash-linear-attention)：纯 Python + Triton 实现，跨平台（CUDA、Ascend NPU、AMD GPU）。

### 7.4 Qwen3.5 8.6x / 19x 提速来源

| 优化 | 贡献 |
|------|------|
| **GDN 替代大部分 GA** | 32K 时 5x，256K 时 10x |
| **FLA chunking 并行** | 训练速度额外 5-10x |
| **混合 3:1** | GA 层保证精度 |
| **累计** | 32K 时 **8.6x**，256K 时 **19x** |

**为什么 256K 加速更明显**：因为 GDN 的 O(Ld²) 相比 Attention 的 O(L²d) 在 L 越大时优势越明显。

---

## 总结

### 一句话总结

> **GDN = Mamba 2 门控（α 控制历史遗忘速度）+ Delta Rule（β 用错误修正关联记忆）**：
> - Mamba 2 给 GDN **时间维度的遗忘**（α 让旧关联自然衰减）
> - Delta Rule 给 GDN **关联维度的修正**（β 让 S 学会"什么该记、什么该改"）
> - 两者结合 = "**既跟时间走（α），又跟错误走（β）**"——这是 GDN 比纯 Mamba 2 / 纯 Delta Rule 都强的原因

### 推荐阅读

- 想理解 SSD 理论：Mamba 2 论文 [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- 想理解 Delta Rule：DeltaNet 论文 [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- 想看工程实现：FLA 库 [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)
- 想看 Qwen3.5 全貌：博客《Qwen3.5 的创新和网络结构》+《Qwen3.5 其余创新》
