# 大模型训练与通用工程技术

这一专题关注算法之外的训练基础设施：怎样把模型和 Batch 放入有限显存，怎样提高有效 Token 比例，怎样控制数值稳定性，以及故障后怎样准确恢复。

## 文章

- [显存与吞吐：把训练放得下、跑得满](memory-and-throughput.md)
  - 梯度累积
  - 混合精度与 Loss Scaling
  - Activation Checkpointing
  - FSDP / ZeRO
  - Sequence Packing
- [训练稳定性与故障恢复](stability-and-recovery.md)
  - Gradient Clipping
  - Learning Rate Warmup / Decay
  - 完整训练 Checkpoint

## 先判断瓶颈再选技术

| 现象 | 首要技术 | 用什么验证 |
| --- | --- | --- |
| 单个 Batch OOM | 梯度累积 | 峰值显存、global batch 是否一致 |
| 长序列 activation OOM | Activation Checkpointing | 激活显存与额外计算时间 |
| 参数/优化器状态放不下 | FSDP / ZeRO | 单卡状态显存、通信占比 |
| Padding 比例高 | Sequence Packing | 有效 token 比例、mask 正确性 |
| 吞吐受精度/带宽限制 | BF16/FP16 混合精度 | tokens/s、NaN/Inf、质量 |
| 梯度偶发爆炸 | Gradient Clipping | 裁剪前后 norm、异常 batch |
| 训练初期震荡 | LR Warmup | early loss、峰值 LR 与步数 |
| 故障后无法续跑 | 完整 Checkpoint | 实际恢复演练与曲线连续性 |
