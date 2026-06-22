# Phase 6：系统设计与模拟面试（Day 58-60）

> **学习目标**：完成 4 道经典系统设计 + 12 轮模拟面试（5 概念 + 5 手写 + 2 系统设计），做到能从架构、选型、trade-off 三维度答出深度。

## Day 58：系统设计题

### 题目清单（每题 1~2 小时）

#### 1. 设计一个分布式训练框架
**问题描述**：从零设计一个能训练千亿参数模型的分布式训练框架。覆盖：硬件抽象层、通信原语、并行策略（DP/TP/PP/EP）、优化器、Checkpoint、推理导出。

**要求回答**：
- 整体架构图（画出核心模块与调用关系）
- 关键模块选型理由（NCCL / Gloo / 共享内存）
- 模型并行维度如何编排（5D 并行的拓扑）
- Checkpoint 格式选择（DCP / 自有格式）
- 推理导出路径（HF 兼容 vs 自有）
- 异常处理（节点故障恢复、弹性训练）

#### 2. 设计一个 LLM 推理服务
**问题描述**：设计一个能支撑 1000 QPS、平均延迟 < 200ms 的 LLM 在线推理服务。

**要求回答**：
- 整体架构图（API Server / 调度 / 模型执行 / KV Cache）
- 调度策略（Static / Dynamic / Continuous）
- KV Cache 管理（PagedAttention、Prefix Caching、抢占）
- 性能优化（Speculative、Continuous Batching、量化）
- 监控指标（QPS / TPS / TTFT / TBT / 显存）
- 弹性扩缩容（HPA / 冷启动 / 模型加载）

#### 3. 设计一个 KV Cache 共享系统
**问题描述**：设计一个跨请求共享 KV Cache 的系统，支持 prefix 复用、跨实例路由、容量管理。

**要求回答**：
- Prefix Tree / Radix Tree 数据结构
- Cache Key 设计（token 序列 hash + 长度）
- 跨实例路由（一致性 hash / 广播）
- 容量管理（LRU / LFU / cost-based eviction）
- 一致性保证（写入原子性、版本控制）

#### 4. 设计一个百万 QPS 的 Embedding 服务
**问题描述**：设计一个能支撑百万 QPS 的文本 Embedding 服务，要求 P99 延迟 < 50ms。

**要求回答**：
- 模型服务化（TensorRT / ONNX Runtime / vLLM）
- 特征缓存（向量级 / token 级 / Embedding 级）
- 批处理策略（Dynamic Batching）
- 向量检索召回（HNSW / IVF / ScaNN）
- 监控与降级

---

## Day 59：模拟面试（上午场）

### 概念题（5 轮，每轮 15 分钟）

#### 轮 1
- DDP 与 FSDP 区别？
- FSDP1 vs FSDP2？
- ZeRO-1/2/3 分别做了什么？
- 答：数据并行是同一份参数每卡一份，全量通信梯度；FSDP 把参数/梯度/优化器状态分片。FSDP1 用 `flat_parameter` + Bucket，FSDP2 用 per-parameter 分片 + `fully_shard`。ZeRO-1 只分优化器状态；ZeRO-2 加梯度；ZeRO-3 加参数。

#### 轮 2
- 解释 TP / PP / SP / CP / EP 各解决什么问题？
- 答：TP（张量并行）解决单层参数过大；PP（流水并行）解决层数过多 + activation 过大；SP（序列并行）解决序列维 activation；CP（上下文并行）解决长序列；EP（专家并行）解决 MoE 路由。

#### 轮 3
- Flash Attention 相比朴素 Attention 快在哪？
- 答：IO-aware，tiling + online softmax，把 attention 中间结果放 SRAM 而不是 HBM，减少 IO 次数。复杂度仍 O(N²)，但常数项小很多（GPU 内存带宽决定）。

#### 轮 4
- 解释 vLLM 的 Paged Attention 与 Continuous Batching。
- 答：PagedAttention 把 KV cache 分页（block）管理，类似 OS 虚拟内存，消除碎片化，支持 prefix 共享与抢占。Continuous Batching 在 token-level 调度，每生成一个 token 就能塞入新请求，提升吞吐。

#### 轮 5
- 解释 MoE 的 all-to-all 通信与负载均衡。
- 答：每个 token 按路由器分配到 top-K 专家，不同 token 可能去不同 rank，需要 all-to-all 把 token 重排到目标 rank 的专家。负载均衡通过 aux loss（专家容量均匀）实现，避免某些专家过载。

---

## Day 59：模拟面试（下午场）

### 手写代码题（5 轮，每轮 20 分钟）

#### 轮 1：手写 Ring-AllReduce
- 题目：用 PyTorch 写一个 8-rank Ring AllReduce
- 要点：`send` / `recv` 的 chunk-by-chunk 传递；两次环（reduce-scatter + all-gather）
- 见 [07-手写代码清单 - Q1](./07-手写代码清单.md)

#### 轮 2：手写 AdamW 优化器
- 题目：30 行 Python 实现 AdamW
- 要点：bias correction + decoupled weight decay
- 见 [07-手写代码清单 - Q5](./07-手写代码清单.md)

#### 轮 3：手写 ColumnParallelLinear
- 题目：实现 TP 中的 Column-Parallel Linear
- 要点：`forward` 切分输出维度；`backward` 用 all-reduce 聚合梯度
- 见 [07-手写代码清单 - Q11](./07-手写代码清单.md)

#### 轮 4：手写 RMSNorm
- 题目：20 行实现 RMSNorm（含 fp32 计算 + 残差）
- 要点：`x / sqrt(mean(x²) + eps) * weight`
- 见 [07-手写代码清单 - Q9](./07-手写代码清单.md)

#### 轮 5：手写 KV Cache 增量推理
- 题目：实现 incremental decoding 的 KV Cache
- 要点：每步拼接新 K/V 增量；attention 用历史 K/V 拼接
- 见 [07-手写代码清单 - Q19](./07-手写代码清单.md)

---

## Day 60：系统设计 + 复盘

### 系统设计题（2 轮，每轮 30 分钟）

#### 轮 1：设计一个分布式训练框架
- 复用 Day 58 的题 1，但要求 30 分钟内口述
- 重点考察：模块化能力、trade-off 取舍

#### 轮 2：设计一个 LLM 推理服务
- 复用 Day 58 的题 2，但要求 30 分钟内口述
- 重点考察：性能优化能力、监控设计

### 全程复盘

- 60 天知识点覆盖自查（对照 [08-资源与里程碑 - 关键里程碑](./08-资源与里程碑.md)）
- 25 道手写代码题全部完成情况
- 4 道系统设计题答题完整度
- 准备 1 份 1 页简历 + 1 份 5 页作品集（GitHub README + 关键项目）

---

## 面试题库（持续补充）

### 概念题（持续更新）
- ZeRO-Infinity 怎么用 NVMe 卸载？延迟与吞吐 trade-off？
- DeepSpeed 与 Megatron 的集成点？分别在哪个模块？
- 为什么 1F1B 能让 PP 显存占用降到 O(1/stage)？
- VPP（Virtual Pipeline Parallel）为什么比 1F1B 吞吐高？
- Ulysses CP 与 Ring Attention 哪个更适合 64K+ 序列？
- LoRA 的 `merge_and_unload` 何时调用？是否可逆？
- AdamW 的 weight decay 与 L2 正则等价吗？为什么？
- BF16 的 8 位 exponent 相比 FP16 的 5 位有什么好处？
- Flash Attention 1 vs 2 vs 3 各自的关键改进？
- 推理时 KV Cache 量化（INT4/INT8）会降低生成质量吗？
- vLLM 怎么支持 LoRA 热加载？
- 抢占 + 重计算 vs 抢占 + 交换的 trade-off？
- Disaggregated Prefill/Decode 解决了什么核心问题？
- 端侧 LLM 推理（手机）的核心瓶颈与优化方向？
- TensorRT-LLM 与 vLLM 在生产环境的选型依据？

### 系统设计题（持续更新）
- 设计一个支持 100 万 QPS 的 LLM 在线服务
- 设计一个支持 100B 参数的分布式训练平台
- 设计一个 1 万亿参数的 MoE 训练框架
- 设计一个 64K 长文档的 RAG 系统
- 设计一个支持多模型多租户的 LLM 网关

---

## Phase 6 自检清单

- [ ] 4 道系统设计题能在 30 分钟内完整口述
- [ ] 12 轮模拟面试全部完成（5 概念 + 5 手写 + 2 系统设计）
- [ ] 1 页简历 + 5 页作品集准备就绪
- [ ] 60 天知识点覆盖自查 100% 完成
