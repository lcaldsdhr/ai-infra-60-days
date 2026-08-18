# Phase 1：基础与单卡深度（Day 01-10）

> **学习目标**：建立 AI Infra 整体认知；深入 PyTorch 内部机制；掌握混合精度训练与单卡性能分析。

## Day 01-02：AI Infra 全景与生态

### 理论
- 读 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（重温 Transformer 细节）
- 读 [Andrej Karpathy "Zero to Hero"](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) 系列（BPE、GPT、minbpe 等）
- 读 [Lilian Weng 技术博客](https://lilianweng.com/)（"Prompt Engineering"、"LLM-powered Autonomous Agents"、"Why GPT-4 fakes reasoning" 等经典长文）
- 读 [Sebastian Raschka "Build an LLM (from scratch)"](https://github.com/rasbt/LLMs-from-scratch) 的 Chapter 1-2（理解 LLM 的整体结构与训练流程）
- 整理"训练 vs 推理"、"数据并行 vs 模型并行"、"延迟 vs 吞吐" 等基本概念对照表

### 实践
- 用 `transformers` + `datasets` 跑通一个 BERT-base 的 SFT，写好训练脚本（含 EMA、Eval 钩子）

### 面试卡片
- 训练大模型的核心难点有哪些？分别用什么方法解决？
- 为什么需要分布式训练？单卡极限在哪里？

> **AI Infra 全景阅读建议（替代原"白皮书"）**：
> 1. **入门**：Karpathy 视频（直觉）+ The Illustrated Transformer（图解）
> 2. **系统认知**：Lilian Weng 博客（深度长文，覆盖 LLM 训练/推理/评估全栈）
> 3. **动手前置**：Sebastian Raschka "Build an LLM (from scratch)" Chapter 1-2（数据 + 模型的整体结构）
> 4. **进阶前置**：[CMU 11-667 Large Language Model Systems](https://www.cs.cmu.edu/~katef/courses/llmsys-2024/index.html) 与 [Stanford CS336](https://stanford-cs336.github.io/spring2024/) 课程主页（看 syllabus，建立全局认知）

---

## Day 03-04：PyTorch 深度

### 理论
- `torch.autograd` 源码（[gen_engine](https://github.com/pytorch/pytorch/tree/main/torch/csrc/autograd) + Python 侧 `autograd.py`）
- `nn.Module` / `Parameter` / `Buffer` 区别
- `torch.compile` 的基本机制（FX graph + inductor）

### 实践
- 手写一个 `Linear` 层（继承 `nn.Module`），用 `torch.autograd.Function` 自定义前反向
- 用 `torch.profiler` profile 一个 Transformer block 的耗时分布，画出 breakdown

### 面试题
- `model.eval()` 与 `torch.no_grad()` 区别？
- PyTorch 的 DDP 与 DP（`DataParallel`）区别？为什么 DP 慢？
- `torch.zeros(3, requires_grad=True)` 与 `nn.Parameter` 区别？

---

## Day 05-06：自动微分与计算图

### 理论
- 动态图 vs 静态图（PyTorch vs TensorFlow）
- 反向传播链式法则、`backward()` 实现
- 显存组成：模型参数 + 梯度 + 优化器状态 + 激活值 + 临时 buffer

### 实践
- 写一个 5 行代码计算 7B 模型的"理论显存占用"（参数 fp16 + 优化器 fp32 + 梯度 fp16）
- 写一个简单的 `recompute` 装饰器（用 `torch.utils.checkpoint`）

### 面试手写
- 写一段代码估算"训练 7B 模型最少需要多少显存"

---

## Day 07-08：混合精度训练（AMP）

### 理论
- FP32 / FP16 / BF16 / TF32 的位级结构与数值范围
- 损失缩放（loss scaling）
- AMP 的两类实现：`torch.cuda.amp`（autocast + GradScaler）vs Apex O2

### 实践
- 用 `torch.cuda.amp.autocast` + `GradScaler` 训练一个 GPT-2 small，对比 FP32 / FP16 / BF16 收敛曲线
- 测不同 dtype 下的吞吐与显存

### 面试题
- BF16 相比 FP16 有什么优势？为什么 BF16 不需要 loss scaling？
- FP16 训练中 `inf` / `NaN` 出现时如何排查？

---

## Day 09-10：性能分析与显存优化（单卡）

### 理论
- Roofline 模型（计算受限 vs 带宽受限）
- `torch.profiler` / Nsight Systems / Kineto 输出解读
- Flash Attention 思想（tiling + online softmax）

### 实践
- 写一个 PyTorch `scaled_dot_product_attention` 的等价实现（手撕 SDPA），与 `F.scaled_dot_product_attention` 对比
- 用 Profiler 找到 ResNet-50 的瓶颈层

### 面试手写
- **手写 SDPA**（无 Flash，朴素 O(N²)），要求能讲清为何 Flash 是 IO-aware

---

## Phase 1 自检清单

- [ ] 能讲清"训练一个 7B 模型最少需要多少显存"（参数 / 梯度 / 优化器 / 激活 / 临时）
- [ ] 能手写一个带 AMP 的训练 step（autocast + GradScaler）
- [ ] 能用 `torch.profiler` 定位 ResNet-50 / Transformer block 的瓶颈
- [ ] 能区分 `nn.Parameter` / `Buffer` / `requires_grad` / `register_buffer`
- [ ] 能讲清 BF16 vs FP16 vs TF32 vs FP32 的位级结构
