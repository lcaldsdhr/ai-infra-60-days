# Phase 4：推理与服务化（Day 41-50）

> **学习目标**：掌握 LLM 推理的 KV Cache 优化、调度算法、Speculative Decoding；理解 vLLM / TensorRT-LLM / SGLang 等主流推理引擎的设计与源码。

## Day 41-42：KV Cache 优化

### 理论
- KV Cache 原理与显存占用计算
- Multi-Query Attention (MQA) / Grouped-Query Attention (GQA) / MLA（DeepSeek-V2）
- Paged Attention（vLLM 的核心）
- KV Cache 压缩：Quantization、Sliding Window、Heavy-Hitter、StreamingLLM

### 实践
- 手写一个支持 KV Cache 的 incremental decoding
- 读 vLLM [`block_manager.py`](https://github.com/vllm-project/vllm)
- **手写一个简化版 Paged Attention**（每页 16 token）

### 面试手写
- 手写 KV Cache 增量推理（含 GQA）
- 解释 Paged Attention 解决了传统 KV Cache 的什么问题

---

## Day 43-44：Continuous Batching 与调度

### 理论
- Static Batching vs Dynamic Batching vs Continuous Batching
- vLLM 的 iteration-level scheduling
- 抢占（preemption）与重计算
- Sarathi-Serve、BitsAndBytes 调度

### 实践
- 读 vLLM [`scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py)
- 写一个简化版 Continuous Batching 调度器（Python 伪代码）

### 面试题
- 什么是 token-level 与 request-level 调度？vLLM 选哪种？
- 抢占 + 重计算 vs 抢占 + 交换的取舍？

---

## Day 45-46：Speculative Decoding

### 理论
- Speculative Decoding（Leviathan et al. 2023）
- Self-Speculative Decoding、Medusa、EAGLE、EAGLE-2
- Lookahead Decoding、Prompt Lookup

### 实践
- 实现一个最简版 Speculative Decoding（小模型 draft + 大模型 verify）
- 读 Medusa 论文与代码

### 面试手写
- 写一段 speculative decoding 的 verify 逻辑（接受/拒绝 token）

---

## Day 47-48：vLLM 源码深度

### 理论
- vLLM 架构：`EngineCore` / `AsyncLLM` / `Scheduler` / `ModelRunner` / `BlockManager`
- `torch.compile` + CUDA Graph 在 vLLM 中的应用
- Chunked Prefill、Prefix Caching、Disaggregated Prefill/Decode

### 实践
- 部署 vLLM（`python -m vllm.entrypoints.openai.api_server`），压测 `QPS` / `TPS` / `TTFT`
- 读 vLLM 关键文件：`scheduler.py`、`block_manager.py`、`model_executor.py`

### 面试题
- vLLM 如何解决 prefix 共享？
- Disaggregated Prefill/Decode 解决了什么痛点？

---

## Day 49-50：TensorRT-LLM 与其他推理引擎

### 理论
- TensorRT-LLM 架构：In-flight Batching、Kernel Fusion、Quantization
- SGLang（RadixAttention、structured output）
- LMDeploy（Turbomind）、MLC-LLM（端侧）

### 实践
- 用 TensorRT-LLM 构建一个 7B 模型的 engine，对比 vLLM 的吞吐
- 读 SGLang [`RadixAttention`](https://github.com/sgl-project/sglang) 源码

### 面试题
- TensorRT-LLM 与 vLLM 的核心差异？
- 端侧推理（手机/车机）的核心优化点？

---

## Phase 4 自检清单

- [ ] 能讲清 KV Cache 的显存占用公式（batch × seq × heads × head_dim × 2 × 2）
- [ ] 能手写 KV Cache 增量解码（含 GQA）
- [ ] 能讲清 Paged Attention 解决传统 KV Cache 什么痛点（碎片化、共享、抢占）
- [ ] 能讲清 Continuous Batching vs Static Batching 的延迟/吞吐差异
- [ ] 能讲清 Speculative Decoding 的接受/拒绝逻辑与分布约束
- [ ] 能列出 vLLM 5 个关键文件及其作用
