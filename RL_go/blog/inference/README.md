# LLM 推理专题

这一专题讨论模型在“已经训练好”之后，怎样以更低的延迟、更高的吞吐稳定地服务请求。重点不在某一个推理框架的 API，而在理解资源瓶颈、请求生命周期与调度取舍。

## 文章

- [LLM 推理系统全景：从请求到性能治理](serving-system-guide.md)：建立请求生命周期、指标与优化手段的总地图。
- [高吞吐推理：Batch、Cache 与解码加速](batching-cache-and-acceleration.md)：连续批处理、Paged KV、Prefix Cache、量化、投机解码与并行。
- [Prefill–Decode（PD）分离：从一次请求到集群调度](pd-disaggregation.md)：理解 Prefill/Decode 的不同瓶颈、不分离/静态分离/动态分离的边界，以及 KV Cache 传输、背压和一致性要求。
- [容量规划与可观测性](observability-and-capacity.md)：把 TTFT、TPOT、吞吐、P99、KV 显存和 SLO 串成可诊断闭环。

## 建议阅读顺序

1. 先读推理系统全景，知道一个请求经过哪些阶段、各指标属于哪里。
2. 再读 Batch/Cache/加速专题，理解单机吞吐如何提升。
3. 接着读 PD 分离，理解阶段隔离、KV 传输和集群调度。
4. 最后用容量规划与可观测性把优化效果量化，而不是只看单一吞吐数字。

## 本专题的指标语言

| 指标 | 含义 | 它通常暴露的问题 |
| --- | --- | --- |
| TTFT | Time To First Token，首 token 延迟 | 排队、Prefill 拥塞或长 Prompt 干扰 |
| TPOT | Time Per Output Token，后续 token 间隔 | Decode 访存压力、批处理和调度效率 |
| P99 | 99 分位延迟 | 热点、资源争用、KV 传输抖动或背压失控 |
| 吞吐 | 单位时间完成的 token / 请求 | 资源利用率与批处理策略 |

完整论文、官方文档与固定版本源码入口见[一手资料索引](../../docs/research/learning-curriculum-primary-sources.md)。
