# Phase 2：分布式训练核心（Day 11-25）

> **学习目标**：掌握数据并行 / 张量并行 / 流水并行 / 序列并行 / 专家并行的原理与实现；具备 5D 混合并行编排能力。

## Day 11-12：数据并行 DP / DDP

### 理论
- PyTorch DDP 原理：Ring-AllReduce、gradient bucketing、overlap
- `DistributedSampler` 与 `DataLoader` 配合
- DDP 启动方式：`torchrun` / `multiprocessing.spawn` / `elastic launcher`

### 实践
- 用 DDP 训练一个 ResNet-50（CIFAR-10），2 机 8 卡，测加速比
- 读 DDP 源码 [torch/nn/parallel/distributed.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py)，重点看 `Reducer` 和 `_ddp_init_helper`

### 面试手写
- **手写 Ring-AllReduce**（无 NCCL，纯 PyTorch），要求 all-reduce 一个 8-rank 的 float 列表
- 写一段 DDP 启动代码（含 `init_process_group` + `DistributedSampler`）

---

## Day 13-14：ZeRO 系列 / FSDP

### 理论
- ZeRO-1/2/3 阶段划分（Pos/g/p 三类状态分片）
- FSDP vs FSDP2 区别：flat 参数 + `summon_full_params`
- PyTorch FSDP 配置：`sharding_strategy` / `backward_prefetch` / `forward_prefetch` / `limit_all_gathers`

### 实践
- 用 `torch.distributed.fsdp.fully_shard`（FSDP2 API）包装一个 1.3B GPT-2 跑通训练
- 对比 ZeRO-1/2/3 在显存上的差异（用 `torch.cuda.memory_summary`）

### 面试手写
- **手写 ZeRO-1 优化器**（把 optimizer state 按 rank 分片，每个 rank 只负责更新自己分到的参数）
- 写代码用 FSDP2 包装一个 4 层 Transformer

---

## Day 15-16：张量并行 TP

### 理论
- Megatron-LM TP：Column-Parallel Linear + Row-Parallel Linear
- 通信原语：`all-reduce` vs `all-gather` vs `reduce-scatter`
- Sequence Parallel (SP) 在 LayerNorm/Dropout 上的应用

### 实践
- 手写一个 1 层 Transformer 的 TP 版本（用 PyTorch 原生 `torch.distributed` 模拟）
- 用 `torch.distributed.tensor`（DTensor）跑一个 TP=2 的 MLP

### 面试手写
- **手写 ColumnParallelLinear**（`forward` / `backward` 通信原语要写对）
- 画 TP=2 时一个 MLP 的数据流图

---

## Day 17-18：流水并行 PP

### 理论
- GPipe（同步微批）+ PipeDream（1F1B 异步）
- 1F1B / Interleaved 1F1B / Zero-Bubble
- PP 通信原语：`P2P Send/Recv`
- Activation recomputation 与 PP 配合

### 实践
- 用 `torchgpipe` 或自己实现一个 4-stage PP，跑 BERT-base 微调
- 画 1F1B 时间线

### 面试手写
- 写伪代码 1F1B 调度器（`steady_phase` + `cooldown_phase`）
- 解释 VPP（Virtual Pipeline Parallel）相对 PP 的优势

---

## Day 19-20：序列并行 SP + 上下文并行 CP

### 理论
- SP：把 LayerNorm/Dropout 在序列维切分，配合 TP 减少 activation 重复
- CP：Ulysses CP（all-to-all 重排 head 维）+ Ring Attention（KV 切分循环）
- 长序列场景（>32K）的 CP 通信与显存折中

### 实践
- 读 Megatron-Core CP 源码（[sequence_parallel.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py)）
- 用 Ring Attention 跑一个 128K 序列的小模型

### 面试手写
- 写一段 Ulysses CP 中 head 切分 + all-to-all 的 PyTorch 伪代码
- 比较 Ulysses CP 与 Ring Attention 的通信开销

---

## Day 21-22：专家并行 EP（MoE）

### 理论
- Mixture-of-Experts (MoE)：Top-K 路由、专家容量、负载均衡损失
- EP 的 all-to-all 通信
- DeepSeek-V2/V3 的 DeepSeekMoE（细粒度专家 + 共享专家）

### 实践
- 用 `transformers` + 一个简化 MoE 实现，验证 all-to-all
- 读 [DeepSeek-V3 论文](https://arxiv.org/abs/2412.19437) 的 MoE 部分

### 面试题
- MoE 训练中如何保证不同 rank 看到的 expert 数量一致？
- Load balance loss 的设计动机是什么？

---

## Day 23-24：混合并行 + ZeRO++

### 理论
- 4D/5D 混合并行：TP × CP × PP × EP × DP
- ZeRO++（DeepSpeed）：通信-计算重叠优化
- Auto-Parallel（MindSpore / Alpa）

### 实践
- 读 [Megatron-LM GPT-3 训练脚本](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/run_gpt3.sh)，理解并行维度编排
- 在小模型（GPT-2）上同时开 TP=2 + PP=2 + DP=2，验证正确性

### 面试手写
- 给定 64 卡、4 个 8 卡节点，画出"TP=8 / PP=2 / DP=4"的并行拓扑，并解释 inter-node 通信策略

---

## Day 25：分布式集合通信基础

### 理论
- NCCL 算法树（Ring / Tree）
- 通信原语：`all-reduce` / `all-gather` / `reduce-scatter` / `broadcast` / `gather` / `scatter` / `barrier` / `send` / `recv`
- 网络拓扑感知（NVLink / IB / RoCE）

### 实践
- 用 `nccl-tests` 跑一遍 all-reduce bench
- 用 `torch.distributed` 写一个集合通信测试程序

### 面试手写
- **手写 AllGather**（基于 Ring）
- 解释 `all-reduce` = `reduce-scatter` + `all-gather` 的拆分

---

## Phase 2 自检清单

- [ ] 能讲清 5D 并行（TP/PP/SP/CP/EP）的通信模式与适用场景
- [ ] 能手写 Ring-AllReduce、AllGather、ReduceScatter
- [ ] 能手写 ZeRO-1 优化器、ColumnParallelLinear
- [ ] 能讲清 DDP 启动到运行的完整流程
- [ ] 能解释 Megatron TP 中为什么 Column-Parallel + Row-Parallel 之后只需 2 次 all-reduce
- [ ] 能讲清 Ulysses CP 与 Ring Attention 的差异
