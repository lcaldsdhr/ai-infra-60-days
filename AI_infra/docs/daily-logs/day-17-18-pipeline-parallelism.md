# Day 17-18 笔记：流水并行（Pipeline Parallelism, PP）

> **目标**：能用 `send/recv` 实现两 stage 的 PP；讲清 microbatch、GPipe、1F1B、bubble、重计算与 VPP 的工程权衡。

## 1. 最小可运行实验

代码：[two_stage_1f1b_cpu.py](../../code/day17-pp/two_stage_1f1b_cpu.py)。它用 Gloo 在 CPU 启动两个进程，不依赖 GPU 或 NCCL：

```powershell
python code/day17-pp/two_stage_1f1b_cpu.py
```

默认配置是 `local_batch_size=16`、`num_microbatches=2`，所以每份 `microbatch_size=8`。在单个 data-parallel rank 内，数据使用方式为：

```text
batch [0..15]
├── mb0: [0..7]   -> Stage 0 -> send activation -> Stage 1 -> send activation grad -> Stage 0
└── mb1: [8..15]  -> Stage 0 -> send activation -> Stage 1 -> send activation grad -> Stage 0
```

每个 microbatch 的 loss 都除以 `num_microbatches`；两个 microbatch 的梯度累积完后，每个 stage 各执行一次 `optimizer.step()`。因此它在梯度语义上等价于以 batch size 16 对完整模型做一次训练更新（未考虑随机层和数值舍入）。

若 `DP=4`，上面的 16 是**每个 DP rank** 的 local batch，则 global batch 为：

```text
global_batch = micro_batch_size × num_microbatches × DP × grad_accumulation_steps
             = 8 × 2 × 4 × 1 = 64
```

## 2. PP 究竟传什么

模型按连续层分到 stage。前向时前一个 stage 发送 activation，反向时后一个 stage 回传该 activation 的梯度：

```text
rank 0 / Stage 0                         rank 1 / Stage 1
x -> f0 -> a0 ---- send(a0) ----------> a0 -> f1 -> loss
          <--- send(∂loss/∂a0) --------             backward
          backward(grad_a0)
```

注意边界处的 `activation`：发送端必须保留其 autograd 图，直到收到 `grad_a0` 后调用 `activation.backward(grad_a0)`；接收端拿到的是脱离原图的 tensor，需要 `requires_grad_(True)`，以获得要传回去的 `activation.grad`。这是手写 PP 最容易遗漏的点。

## 3. 2-stage 的 1F1B 时间线

`F(i)` 表示第 i 个 microbatch 的前向，`B(i)` 表示反向。两 stage、两份 microbatch 的核心顺序是：

```text
时间 ->       1       2       3       4
Stage 0       F(0)    F(1)    B(0)    B(1)
Stage 1               F/B(0)  F/B(1)
```

Stage 0 在算 `F(1)` 时，Stage 1 可以计算 `F/B(0)`，这就是流水线重叠。更深的流水线会分成：

1. **warmup**：前向先填满各 stage；
2. **steady state**：每个 stage 尽量交替一个 F 和一个 B，即 1F1B；
3. **cooldown**：不再注入新 microbatch，排空剩余反向。

这个示例为了突出通信顺序，使用阻塞式 `send/recv`；生产系统会采用异步 P2P 通信和 CUDA stream，让通信、前向、反向进一步重叠。

## 4. GPipe、1F1B 与显存

| 调度 | 运行方式 | Activation 显存 | 适用性 |
|---|---|---:|---|
| GPipe | 所有 microbatch 先 F，之后统一 B | 高，需要留住多个 F 的 activation | 教学简单、实现直观 |
| 1F1B | warmup 后交替 F / B | 更低，旧 activation 更早释放 | 训练系统常用 |
| Interleaved 1F1B / VPP | 一张卡承载多个虚拟 model chunks，交错调度 | 取决于 chunk 和重计算策略 | 大规模训练，减少 bubble |

流水线 bubble 的近似比例为：

```text
bubble ≈ (PP - 1) / (num_microbatches + PP - 1)
```

`PP=8, M=8` 时约为 `7/15 ≈ 47%`；将 `M` 提升到 32 后约为 `7/39 ≈ 18%`。但 `M` 不是越大越好：每份太小会让矩阵乘变小、kernel 效率降低，并增加调度与通信次数。

## 5. 工程落地检查表

- **模型切分**：优先按 Transformer block 均匀切；不能只按层数，要按参数量、FLOPs、activation 大小实测平衡。
- **通信**：同节点 stage 尽量放在 NVLink/NVSwitch 拓扑内；跨节点 PP 需要经过 IB/RoCE，通常比 TP 通信更容易承受，因为 PP 只与相邻 stage 通信。
- **批次定义**：明确 `micro_batch_size`、`num_microbatches`、`DP`、梯度累积次数；不要把 local batch 和 global batch 混用。
- **loss 缩放**：如果一个 update 消费 M 个 microbatch，常见平均 loss 要除以 M；AMP 下还要与 loss scaler 配合。
- **activation 内存**：首先采用 1F1B；仍不足时，对 Transformer block 开 activation recomputation。它以额外前向计算换内存。
- **不均衡排查**：观察各 stage 的 compute、P2P 时间和 idle 时间；最慢的 stage 决定吞吐。必要时调整切分边界、使用 VPP 或把 embedding / LM head 单独处理。
- **正确性**：固定随机数，关闭 dropout；比较无 PP 单卡基线与 PP 的 loss、参数梯度、一次更新后的参数。允许 BF16/FP16 下的小数值误差。
- **容错**：一次 pipeline flush 内任意 rank 失败都会使相邻 rank 阻塞；生产框架需配合 elastic restart、checkpoint 和通信超时配置。

## 6. 高频面试题

### Q1：PP 和 TP 的本质区别？

PP 按**层/模型深度**切分，使用相邻 stage 的 P2P 通信，并以 bubble 为主要额外损失；TP 在**同一层内部**切张量，常有 all-reduce / all-gather，通信频率更高且更依赖高速互联。PP 主要解决模型层数方向的参数和 activation 容量，TP 主要解决单层矩阵过大或提升单层计算吞吐。

### Q2：为什么需要 microbatch？

没有 microbatch 时，每个 stage 只能等前一 stage 完全结束才工作，绝大多数设备闲置。microbatch 将一个 batch 变成流水，让不同 stage 同时服务不同数据；其代价是 bubble、更多 P2P 次数，以及过小时的计算效率下降。

### Q3：1F1B 为什么比 GPipe 省 activation？

GPipe 要到所有前向完成后才开始反向，早期 microbatch 的 activation 长时间不能释放。1F1B 在 warmup 后让某个 microbatch 尽快反向，其 activation 完成反向就释放，因此峰值需要同时保存的 activation 更少。

### Q4：PP 的全局 batch 怎么计算？

`global_batch = micro_batch_size × num_microbatches × DP × grad_accumulation_steps`。PP 本身不复制样本，只是顺序处理同一个 DP rank 的多个 microbatch；全局 batch 乘 DP，是因为各 DP rank 处理不同样本。

### Q5：VPP 为什么可以减小 bubble？

它把每个物理 stage 再拆为多个虚拟 chunks 并交错执行。流水线的有效深度被更细粒度地填充，某个 GPU 不必长时间等完整物理 stage 链路流过，因此空闲区变短；代价是更多 P2P、更多调度状态，以及可能更高的 activation 压力。

### Q6：手写 PP 时为什么不能直接对收到的 activation 调 `backward()`？

收到的 activation 默认没有连接到发送 rank 的 autograd 图。接收端先令其 `requires_grad=True`，对本 stage 的 loss 反向，取得 `activation.grad` 并发回；发送端再以该梯度作为向量-Jacobian 积的输入：`saved_activation.backward(received_grad)`。

## 7. 自测

1. `batch_size=16, num_microbatches=2, DP=8, gradient_accumulation_steps=4` 时，global batch 是多少？
2. 为什么 `num_microbatches` 增大能减少 bubble，却不一定提高吞吐？
3. 若 Stage 2 比其他 stage 慢 30%，吞吐由谁决定？如何定位并处理？
4. 试着将实验改为 `--num-microbatches 4`，并解释时间线中新增事件的意义。

答案：1) `16 × 8 × 4 = 512`；2) 每份过小会降低 GEMM 效率并增大通信/调度开销；3) 最慢的 Stage 2，使用 profiler 看 compute、P2P、idle 后重新切层或使用 VPP/重计算策略；4) 每份大小变为 4，更多 microbatch 被注入，流水线更饱满。
