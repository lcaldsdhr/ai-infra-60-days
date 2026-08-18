# Phase 5：主流框架源码（Day 51-57）

> **学习目标**：能讲清"主流框架是如何把上面这些算法落地到工程代码里的"——以 DDP/DeepSpeed/Megatron/vLLM 四大框架为对象，看懂关键路径、核心类、调用关系。

## Day 51-52：PyTorch DDP 源码

### 重点文件
- [`torch/nn/parallel/distributed.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py) — `DistributedDataParallel`
- [`torch/distributed/optim/zero_redundancy_optimizer.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/optim/zero_redundancy_optimizer.py) — ZeRO-1
- [`torch/distributed/fsdp/`](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp) — FSDP

### 重点问题
- DDP `Reducer` 的 bucketing 策略
- DDP hook 的 `register_forward_pre_hook` / `register_full_backward_hook`
- FSDP1 vs FSDP2 的 `flat_parameter` 区别
- DDP `no_sync()` 在梯度累积时的作用

### 必读代码段
- `DistributedDataParallel._ddp_init_helper` — 初始化流程
- `Reducer._push_all_reduce_params` — bucketing 触发
- `FSDP._post_init_method` — 参数分片
- `fully_shard` (FSDP2) — 状态机与通信

---

## Day 53-54：DeepSpeed 源码

### 重点文件
- [`deepspeed/runtime/engine.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/engine.py)
- [`deepspeed/runtime/zero/stage_1_and_2.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py) — ZeRO-1/2
- [`deepspeed/runtime/zero/stage3.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py) — ZeRO-3
- [`deepspeed/runtime/zero/partition_parameters.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py) — Init 阶段参数分片
- [`deepspeed/runtime/zero/offload_states.py`](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/offload_states.py) — Offload

### 重点问题
- ZeRO-3 的 `_post_init_method` 在 `init` 时如何分片参数
- ZeRO-Offload 的 `offload_optimizer` 实现
- DeepSpeed 与 Megatron 的集成点（`deepspeed.runtime.megatron`）
- ZeRO-Infinity 的 NVMe 卸载流水线

### 必读代码段
- `ZeROStage3._post_init_method` — Init 阶段完成参数 gather → 分片
- `ZeROStage3.optimizer_step` — 分片 optimizer 步进
- `PartitionedParameterCoordinator` — 预取协调器

---

## Day 55-56：Megatron-LM 源码

### 重点文件
- [`megatron/core/tensor_parallel/layers.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py) — TP 层
- [`megatron/core/tensor_parallel/mappings.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py) — 通信原语封装
- [`megatron/core/pipeline_parallel/schedules.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py) — 1F1B / Interleaved
- [`megatron/core/transformer/transformer_block.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py) — 模型主体
- [`megatron/training/arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py) — 并行参数
- [`megatron/core/optimizer/__init__.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/__init__.py) — 分布式优化器

### 重点问题
- Column/Row Parallel Linear 的前向/反向通信
- VPP 怎么把一个 TransformerBlock 切成多个 model chunk
- Megatron 与 FSDP2 的融合（`--use-torch-fsdp2`）
- `get_forward_backward_func()` 如何根据 PP size 选择 1F1B / Interleaved
- Sequence Parallel 在 LayerNorm / Dropout 上的 reduction

### 必读代码段
- `ColumnParallelLinear.forward` / `backward`
- `RowParallelLinear.forward` / `backward`
- `forward_backward_pipelining_without_interleaving`（1F1B）
- `megatron.core.distributed.distributed_data_parallel`（DistributedOptimizer）

---

## Day 57：vLLM 关键路径

### 重点文件
- [`vllm/engine/llm_engine.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py) — 引擎主循环
- [`vllm/engine/async_llm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm.py) — 异步引擎
- [`vllm/core/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py) — 调度器
- [`vllm/core/block_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py) — 块管理
- [`vllm/worker/model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py) — 模型执行
- [`vllm/attention/backends/`](https://github.com/vllm-project/vllm/tree/main/vllm/attention/backends) — Attention 后端
- [`vllm/entrypoints/openai/api_server.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py) — API Server

### 重点问题
- PagedAttention 的 KV block 如何分配/释放
- Continuous Batching 的 iteration-level 调度实现
- Chunked Prefill 怎么把长 prompt 切成小块
- Prefix Caching 的 prefix tree 实现
- vLLM 怎么支持 LoRA 热加载

### 必读代码段
- `Scheduler.schedule()` — 主调度入口
- `BlockManager.allocate_slots` / `free` — 块分配/释放
- `ModelRunner.execute_model` — 单 step 执行
- `LLMEngine.step` — 引擎主循环

---

## 框架对比速查表

| 维度 | PyTorch DDP | DeepSpeed | Megatron-LM | vLLM |
| --- | --- | --- | --- | --- |
| 主要场景 | 数据并行 | 数据并行 + Offload | 模型并行 | 推理服务 |
| 并行 | DP | DP + ZeRO | TP / PP / SP / CP / EP | TP / PP（推理） |
| 显存优化 | DDP 自身 | ZeRO + Offload + Infinity | Recompute + Optimizer Shard | PagedAttention + Continuous Batching |
| Checkpoint | torch.save | DeepSpeed ckpt | Megatron ckpt | HF safetensors |
| 启动器 | torchrun | deepspeed launcher | torchrun + 自定义 | 独立服务 |
| 通信库 | NCCL | NCCL | NCCL | NCCL |
| 核心文件 | `distributed.py` | `engine.py` / `zero/` | `tensor_parallel/` / `pipeline_parallel/` | `scheduler.py` / `block_manager.py` |

---

## Phase 5 自检清单

- [ ] 能在 30 分钟内画出 DDP / DeepSpeed / Megatron / vLLM 四大框架的核心类图
- [ ] 能讲清 DDP bucketing 策略的动机（为什么能 overlap）
- [ ] 能讲清 ZeRO-3 的 `_post_init_method` 做了什么
- [ ] 能讲清 Megatron TP 的两次 all-reduce 通信时机
- [ ] 能讲清 vLLM scheduler 的一次完整 step 流程
- [ ] 能在 PyTorch / DeepSpeed / Megatron / vLLM 中找到对应的通信原语位置
