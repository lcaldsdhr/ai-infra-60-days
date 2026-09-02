# LLM 推理容量规划与可观测性

## 1. 不要只看 GPU 利用率

推理服务至少有五种不同的“慢”：排队慢、Prefill 慢、KV 传输慢、Decode 慢、客户端消费慢。只看端到端延迟无法定位，只看 GPU 利用率更无法区分健康吞吐与过载。

```text
arrival
  → queue_ms
  → prefill_ms
  → kv_transfer_ms（若 PD 分离）
  → first_token_ts
  → inter_token_latency[]
  → finish_ts
```

每条请求应携带稳定 `request_id`，并记录模型/Tokenizer/策略版本、输入输出 token 数、路由节点、终止原因和错误分类。

## 2. 指标分层

| 层级 | 核心指标 | 回答的问题 |
| --- | --- | --- |
| 用户体验 | TTFT、TPOT/ITL、E2E、P50/P95/P99 | 用户看到首字和后续输出是否稳定 |
| 调度 | queue time、running/waiting、preemption、batch tokens | 慢在排队还是执行，是否存在饥饿 |
| Cache | KV bytes、blocks 使用率、prefix hit、eviction | 并发受容量、碎片还是命中率限制 |
| 计算/通信 | prefill/decode tokens/s、kernel time、collective、KV transfer | GPU、显存带宽还是网络在限制吞吐 |
| 可靠性 | OOM、timeout、拒绝、取消、重试 | 过载策略和故障域是否有效 |
| 质量 | 准确率、困惑度、拒答率、格式通过率 | 量化/采样/加速是否改变输出语义 |

## 3. KV 容量的第一性估算

对于 GQA/MQA 模型，每个 token 的 KV 字节数近似为：

```text
2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

总容量还要乘所有活跃请求中已缓存的 token 数。它不包含权重、activation、allocator 元数据和 CUDA workspace，因此只能作为容量规划起点。

运行本仓库计算器：

```powershell
python RL_go/code/inference_kv_capacity/demo.py `
  --layers 28 --kv-heads 4 --head-dim 128 `
  --context 8192 --concurrency 64 --dtype-bytes 2
```

## 4. 压测矩阵

不要只跑固定 128/128 token。至少覆盖：

| 负载 | 输入 | 输出 | 目的 |
| --- | ---: | ---: | --- |
| 短问答 | 128 | 128 | 基础 TPOT 与调度开销 |
| RAG | 4096 | 256 | Prefill、Prefix Cache、TTFT |
| 长生成 | 512 | 2048 | Decode 与 KV 增长 |
| 长上下文 | 16K+ | 256 | KV 容量、分页和通信 |

每类再按并发递增，直到出现 SLO 违约、OOM 或拒绝率上升，得到“安全工作区”而非单个峰值。

## 5. SLO 与过载保护

容量不足时，请求无限进入只会同时放大 queue、KV 占用和 P99。服务需要并发上限、队列上限、超时、取消传播与准入控制；PD 系统还需要 P→D 背压。报告 **goodput**（满足 TTFT/TPOT SLO 的有效吞吐）比只报告总吞吐更接近真实服务价值。

## 6. 验收清单

- 模型、精度、运行时、GPU/网络拓扑和采样参数可复现；
- 请求级 trace 能分解 queue/prefill/transfer/decode；
- 同时有 P50 与 P99，不用平均值掩盖长尾；
- 加速前后质量与错误率无不可接受回退；
- 过载时系统会限流/拒绝，而不是进入不可恢复抖动。

参考：[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)、[Mooncake](https://arxiv.org/abs/2407.00079) 与[完整一手资料索引](../../docs/research/learning-curriculum-primary-sources.md)。
