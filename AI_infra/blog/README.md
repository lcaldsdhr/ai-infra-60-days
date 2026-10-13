# AI Infra 模型、架构与工程实践博客

## 报错分析与排查

1. [模型训练与推理常见报错：从现象到根因的排查方法](common-model-errors-and-debugging.md)：内存踩踏、OOM、索引与张量错误、NaN/Inf、autograd、NCCL、环境/数据/checkpoint，以及 verl / Agentic RL 静默错配。

作图研究：[从高引用 arXiv 论文学习技术图绘制风格](../docs/research/arxiv-figure-style-study.md)，包含论文原图位置、视觉分析与后续作图准则。

## Qwen 系列

1. [Qwen3.5 的创新和网络结构](qwen3.5-innovations-and-architecture.md)
2. [GDN 深度：从 Mamba 2 到 GDN](gdn-deep-dive.md)
3. [Qwen3.5 其余创新：MoE、Early Fusion、mRoPE 与 MTP](qwen3.5-remaining-innovations.md)
4. [Qwen3.8 最新进展：站在 Qwen3.5 上看两条升级路线](qwen3.8-latest-from-qwen3.5.md)
5. [Qwen3.8-Flash-Next：架构、训练与部署](qwen3.8-flash-next-deep-dive.md)

建议先掌握 Qwen3.5 的混合架构和 GDN，再阅读 Qwen3.8 的能力旗舰线与 Flash-Next 下一代架构线。
