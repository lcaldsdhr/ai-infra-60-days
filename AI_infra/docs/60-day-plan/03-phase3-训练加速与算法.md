# Phase 3：训练加速与算法（Day 26-40）

> **学习目标**：掌握优化器、注意力机制、归一化、量化、PEFT 等训练核心算法；具备手写这些算法的能力。

## Day 26-27：优化器（Adam/AdamW/Lion/Muon）

### 理论
- SGD → Momentum → Adam → AdamW → Lion → Muon 的演化
- 权重衰减、bias correction、eps 选择
- 分布式优化器状态分片（ZeRO-1 + bf16 master weight）

### 实践
- **手写 AdamW**（30 行 Python，要包含 bias correction + decoupled weight decay）
- 读 PyTorch [`AdamW`](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py) 源码
- 读 [Muon 论文](https://arxiv.org/abs/2402.18463) 并实现简化版

### 面试手写
- **手写 Adam**（15 行）
- 手写 Lion（sign-based）

---

## Day 28-29：LR Scheduler + 梯度裁剪

### 理论
- warmup + cosine / linear / step / inv_sqrt
- 梯度裁剪：`clip_grad_norm` vs `clip_grad_value_`
- 梯度累积（gradient accumulation）

### 实践
- 手写一个 cosine warmup scheduler（带 min_lr）
- 写一个完整的训练 step（forward / loss / backward / grad clip / step / zero_grad）

### 面试题
- 为什么大模型训练一定要 warmup？
- LAMB / LARS 与 Adam 的差异？

---

## Day 30-31：Flash Attention 深度

### 理论
- IO-aware 思想、tiling、online softmax
- Flash Attention 1 / 2 / 3 的差异
- Flash Decoding、Flash Decoding++

### 实践
- 读 [flash-attention 源码](https://github.com/Dao-AILab/flash-attention)（重点 FA2 forward kernel）
- **手写一个 naive SDPA** 并对比 `F.scaled_dot_product_attention`
- 用 `triton` 写一个 2-3 行的简化 Flash kernel（理解 tiling）

### 面试手写
- **手写 online softmax**（解决分块 softmax 的归一化问题）

---

## Day 32-33：Linear Attention 系列

### 理论
- Linear Attention 起源（Performer / Linformer）
- Mamba（S4 / S6 / Selective SSM）
- RWKV、RetNet、Gated Delta Net（Qwen3 / Qwen3.5 用）
- Mamba-2 与 Transformer 的对偶关系

### 实践
- 读 Mamba 论文与官方实现 [state-spaces/mamba](https://github.com/state-spaces/mamba)
- 用 PyTorch 写一个简化版的 Gated Delta Net（前向 + 反向）

### 面试题
- Linear Attention 与 Softmax Attention 的本质差异？为什么 Linear Attention 推理 KV cache 占用小？
- Mamba 的选择性机制（input-dependent B/C/Δ）解决了什么？

---

## Day 34-35：激活重计算 / Offload / 显存优化

### 理论
- Full Recompute vs Selective Recompute vs Chunk Recompute
- ZeRO-Offload（CPU 卸载 optimizer + grad）
- ZeRO-Infinity（NVMe 卸载）
- Activation Offload（H2D/D2H 异步）

### 实践
- 用 `torch.utils.checkpoint` 验证重计算对显存的影响
- 读 DeepSpeed [`ZeRO-Offload`](https://www.deepspeed.ai/tutorials/zero-offload/) 论文与代码

### 面试手写
- 写一个重计算装饰器，包装任意 `nn.Module.forward`
- 解释 CPU offload 中"分阶段换入换出"的设计

---

## Day 36-37：归一化与位置编码

### 理论
- LayerNorm / RMSNorm 的差异
- Pre-Norm vs Post-Norm
- RoPE / ALiBi / YaRN / NTK-aware 插值

### 实践
- **手写 RMSNorm**（20 行）
- 读 HuggingFace `transformers/modeling_rope_utils.py`
- 实现一个支持 long-context 插值的 RoPE

### 面试手写
- 手写 RMSNorm（含 fp32 计算 + 残差连接）
- 手写 RoPE 的位置旋转

---

## Day 38-39：量化（QAT / PTQ / FP8）

### 理论
- PTQ（训练后量化）：对称 / 非对称 / 通道级
- QAT（训练感知量化）：LSQ、PACT
- INT8 / INT4 / FP8（E4M3 / E5M2）
- GPTQ、AWQ、AutoAWQ、BitNet、KV Cache 量化

### 实践
- 用 `bitsandbytes` 跑 4-bit 加载（`load_in_4bit=True`）
- 读 [GPTQ 论文](https://arxiv.org/abs/2210.17323) + 跑通 GPTQ quantize 脚本
- 在 PyTorch 中实现 per-channel 对称量化

### 面试手写
- **手写 per-channel 对称量化算子**（`quantize` / `dequantize`）

---

## Day 40：模型压缩与稀疏化

### 理论
- 知识蒸馏（KD）、Feature Distillation、DISTILL
- LoRA / QLoRA / DoRA / AdaLoRA
- 剪枝（结构化 / 非结构化 / 2:4 sparsity）

### 实践
- 用 `peft` 跑 LoRA 微调
- **手写 LoRA**（继承 `nn.Module`，冻结 base，加 `lora_A` / `lora_B`）
- 读 QLoRA 论文

### 面试手写
- 手写 LoRA 层（带 `merge_and_unload`）
- 解释 LoRA 相比 Adapter 的优势

---

## Phase 3 自检清单

- [ ] 能手写 AdamW、Adam（含 bias correction）
- [ ] 能手写 cosine warmup scheduler
- [ ] 能讲清 Flash Attention 的 IO-aware 思想
- [ ] 能手写 naive SDPA + online softmax
- [ ] 能讲清 Linear Attention 与 Softmax Attention 的本质差异
- [ ] 能手写 RMSNorm、LoRA、per-channel 量化
- [ ] 能讲清 Pre-Norm vs Post-Norm 的差异
