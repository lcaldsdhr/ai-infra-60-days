# LLM 推理专题

这一专题讨论模型在“已经训练好”之后，怎样以更低的延迟、更高的吞吐稳定地服务请求。重点不在某一个推理框架的 API，而在理解资源瓶颈、请求生命周期与调度取舍。

## 文章

- [Prefill–Decode（PD）分离：从一次请求到集群调度](pd-disaggregation.md)：理解 Prefill/Decode 的不同瓶颈、不分离/静态分离/动态分离的边界，以及 KV Cache 传输、背压和一致性要求。

## 建议阅读顺序

1. 先读 PD 分离，建立“计算密集的 Prefill 与访存密集的 Decode 不是同一类工作”的直觉。
2. 再把连续批处理、KV Cache、请求路由和 P/D 比例放回同一条请求链路理解。
3. 最后结合具体引擎（如 vLLM、SGLang 或厂内服务）验证：指标不只看吞吐，也要看 TTFT、TPOT 和 P99。

## 本专题的指标语言

| 指标 | 含义 | 它通常暴露的问题 |
| --- | --- | --- |
| TTFT | Time To First Token，首 token 延迟 | 排队、Prefill 拥塞或长 Prompt 干扰 |
| TPOT | Time Per Output Token，后续 token 间隔 | Decode 访存压力、批处理和调度效率 |
| P99 | 99 分位延迟 | 热点、资源争用、KV 传输抖动或背压失控 |
| 吞吐 | 单位时间完成的 token / 请求 | 资源利用率与批处理策略 |
