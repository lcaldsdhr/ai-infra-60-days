# Day 9-10 笔记：性能分析与显存优化

> **目标**：掌握单卡性能分析的 Roofline 模型、torch.profiler 使用、Flash Attention 思想；能手撕朴素 SDPA 并与 Flash 对比。

---

## 一、Roofline 模型

**核心问题**：一个算子跑得慢，是因为**算力不够**（compute-bound）还是**带宽不够**（memory-bound）？

```
                    算力 Roof（H100 ~1000 TFLOPS BF16）
                    ┌────────────────────────────────┐
     性能 (FLOP/s)   │     🟦  compute-bound          │
                    │         (大矩阵乘)              │
                    └────────────────────────────────┘
                        ↗ 拐点 = 算力 / 带宽
                           ≈ 1000 / 3.35 ≈ 300 FLOP/Byte
                      ─ ─ ─ ─ ─ ─ ─ ─
                    带宽 Roof（H100 ~3.35 TB/s HBM3）
                    🟩 memory-bound（逐元素、layernorm）
```

**计算公式**：计算强度 (Arithmetic Intensity) = FLOPs / Bytes

| 算子 | FLOPs | Bytes | 计算强度 | 类型 |
|---|---|---|---|---|
| LayerNorm (H=4096) | 3×4096 ≈ 12K | 2×4×4096 ≈ 32KB | 0.37 | **memory-bound** |
| 矩阵乘 [4096×4096] | 2×4096³ ≈ 137G | 4×4096²×2 ≈ 134MB | 1024 | **compute-bound** |

**实际应用**：
- compute-bound 算子的优化方向：**更快的算力**（Tensor Core、编译器优化）
- memory-bound 算子的优化方向：**更少的访存**（算子融合、kernel 合并）

---

## 二、torch.profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
    profile_memory=True,
    record_shapes=True,
) as prof:
    with record_function("forward"):
        out = model(x)
    with record_function("backward"):
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")  # 火焰图
```

**关注 3 个指标**：
- `CUDA total`：GPU 耗时（长 = 瓶颈）
- `FLOPs`：计算量（大 = compute-bound）
- `Self CPU %` vs `CUDA total`：差异大说明 kernel launch 开销高

---

## 三、Flash Attention 思想

### 标准 Attention 的痛

```python
S = Q @ K.T               # [B, H, N, N] ← 大！写回 HBM
P = softmax(S / √d)       # [B, H, N, N] ← 又在 HBM
O = P @ V                 # [B, H, N, d]
```

中间 N×N 矩阵频繁读写 HBM，这是瓶颈——**不是 FLOPs，是 HBM 访存**。

### Flash 的 3 个改进

| 改进 | 作用 |
|---|---|
| **Tiling（分块）** | 把 N×N 切成小块，每块在 SRAM（20MB）算完再写回 HBM |
| **Online Softmax** | 分块计算时不需全局 max/sum，动态更新 |
| **不存中间矩阵** | N×N 注意力矩阵只在 SRAM 中存，不写 HBM |

### HBM 访问量对比

| 实现 | HBM 访问 | N=128K, d=128 |
|---|---|---|
| 标准 Attention | O(N² + Nd) | 128K² × 4B ≈ 64 GB |
| Flash Attention | O(Nd) | 128K × 128 × 16B ≈ 256 MB |
| **节省** | **N/d 倍** | **约 256 倍** |

---

## 四、手撕 SDPA

### 基础版（4 行代码）

```python
def naive_sdpa(Q, K, V):
    d = Q.shape[-1]
    scale = d ** -0.5
    attn = Q @ K.transpose(-2, -1) * scale    # 注意力分数
    attn = F.softmax(attn, dim=-1)             # 归一化
    O = attn @ V                                # 加权求和
    return O
```

### 带 Causal Mask

```python
def naive_sdpa_causal(Q, K, V):
    d = Q.shape[-1]
    scale = d ** -0.5
    attn = Q @ K.transpose(-2, -1) * scale
    N = Q.shape[-2]
    mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
    attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    O = attn @ V
    return O
```

### 核心认知

> 朴素 SDPA 和 Flash Attention 都是 O(N²d) FLOPs，Flash 快是因为 **HBM 访存从 O(N² + Nd) 降到 O(Nd)**。对长序列（N=128K），HBM 访问是瓶颈，Flash 收益 100-1000 倍。

---

## 五、面试标准答案

| 题 | 一句话答 |
|---|---|
| **Roofline 模型** | 计算强度 = FLOPs/Bytes，大于拐点（算力/带宽）是 compute-bound，小于是 memory-bound |
| **torch.profiler** | 3 个活动（CPU/CUDA）、2 个表格（by time / by FLOPs）、1 个 trace（Chrome 火焰图）|
| **Flash Attention** | tiling + online softmax + 不写 N×N 到 HBM，常数倍加速来自 HBM 访问量降 N/d 倍 |
| **手撕 SDPA** | 4 行：@ → scale → softmax → @。加 causal mask 就多 2 行 masked_fill |
| **朴素 Flash 区别** | O(N²d) 一样，常数倍不同。Flash 是 IO-aware（访存优化非 FLOPs 优化）|
