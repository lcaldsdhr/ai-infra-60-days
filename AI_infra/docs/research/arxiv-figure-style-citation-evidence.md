# arXiv 论文插图参考：引用量核验证据

查询日期：2026-10-13（用户所在时区 Asia/Singapore）。本文只记录论文身份与引用索引证据，不分析图片风格。

## 选择口径

将“高引用参考样本”操作性定义为：本次查询在明确标注的引用索引中至少有 1,000 次引用，且论文标题与 arXiv ID 对应。该门槛用于筛选学习样本，不表示插图质量，也不是统一排名。

## 已核实记录

| 论文标题 | arXiv ID | 索引与工作 ID | 当前索引引用数 | 查询源 |
|---|---|---|---:|---|
| Attention is All you Need | [1706.03762](https://arxiv.org/abs/1706.03762) | Semantic Scholar `204e3073870fae3d05bcbc2f6a8e263d9b72e776` | 196,585 | [Graph API](https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762?fields=title,citationCount,externalIds,url) |
| LoRA: Low-Rank Adaptation of Large Language Models | [2106.09685](https://arxiv.org/abs/2106.09685) | Semantic Scholar `a8ca46b171467ceb2d7652fbfb67fe701ad86092` | 24,226 | [Graph API](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2106.09685?fields=title,citationCount,externalIds,url) |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | [2205.14135](https://arxiv.org/abs/2205.14135) | Semantic Scholar `87c5b281fa43e6f27191b20a8dd694eda1126336` | 5,550 | [Graph API](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2205.14135?fields=title,citationCount,externalIds,url) |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | [2309.06180](https://arxiv.org/abs/2309.06180) | OpenAlex `W4387321091` | 1,702 | [OpenAlex work API](https://api.openalex.org/works/W4387321091) |
| ReAct: Synergizing Reasoning and Acting in Language Models | [2210.03629](https://arxiv.org/abs/2210.03629) | Semantic Scholar `99832586d55f540f603637e458a292406a0ed75d` | 12,411 | [Graph API](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2210.03629?fields=title,citationCount,externalIds,url) |

Semantic Scholar 四项记录均通过 `ARXIV:<ID>` 查询，并同时取得 `title`、`externalIds.ArXiv`、`paperId` 与 `citationCount`。Attention 的标题大小写沿用索引返回值。

PagedAttention 的 OpenAlex 记录标题完全匹配，`locations` 包含 `http://arxiv.org/abs/2309.06180` 和 `https://doi.org/10.48550/arxiv.2309.06180`；其正式发表 DOI 为 `10.1145/3600006.3613165`。本次 Semantic Scholar 请求返回 HTTP 429，因此使用 OpenAlex，不能将两种索引的计数混合为排名。

## 未核实与拒绝使用的记录

| 目标论文 | 状态 | 原因 |
|---|---|---|
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | 引用数未核实，不据此宣称达到门槛 | Semantic Scholar API 连续返回 HTTP 429；OpenAlex 的 DOI 查询返回标题不匹配的记录，予以拒绝 |

DeepSeekMath 的 arXiv 官方页面确认标题、ID `2402.03300` 和 DOI `10.48550/arXiv.2402.03300`。但 [OpenAlex DOI 查询](https://api.openalex.org/works/https://doi.org/10.48550/arxiv.2402.03300?select=id,title,cited_by_count,doi,locations) 返回 `W4391631327`，标题为 “Axiom-World: A Pre-Registered Study of Two-Stage Post-Training for Rule-Grounded Planning in a Verifiable Toy World”，计数为 98。即使 DOI/arXiv location 相同，也因标题不匹配而不能把 98 归于 DeepSeekMath；这里只记录异常以便复核。

## 索引限制

- 引用数属于各索引的实时快照，不是 arXiv 提供的统计，也不等于 Google Scholar 引用数。
- 同一论文的不同版本、正式发表记录与去重方式可能影响统计。仅凭标题搜索结果不能确认身份，必须交叉检查 arXiv ID、标题与工作 ID。
- 本次 FlashAttention 的 [OpenAlex `W7133227460`](https://api.openalex.org/works/W7133227460) 标题和 arXiv location 匹配，计数为 1,001；Semantic Scholar 为 5,550。差异本身说明不同索引不可直接互换。主表采用 Semantic Scholar 已成功返回的记录，不求和、不择最大值构造统一统计。
- Semantic Scholar 未认证 API 本次间歇返回 HTTP 429。未取得有效响应的计数保留为未核实，不用二手文章补齐。
- 查询接口：[Semantic Scholar Graph API 文档](https://api.semanticscholar.org/api-docs/graph)、[OpenAlex works API 文档](https://docs.openalex.org/api-entities/works)。后续复核需重新查询；本文数值不会自动更新。

## 补充参考：Mamba、DPO 与 GPipe

同日（2026-10-13，Asia/Singapore）补充查询三篇论文。Semantic Scholar 对三项 `ARXIV:<ID>` 请求均返回 HTTP 429，未进行长时间重试；下表使用 OpenAlex DOI 直查结果。三个记录的标题、arXiv DOI 与 `locations` 中的 arXiv ID 均匹配。

| 论文标题 | arXiv ID | OpenAlex 工作 ID | `cited_by_count` | 本次 ≥1,000 门槛 |
|---|---|---|---:|---|
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | [2312.00752](https://arxiv.org/abs/2312.00752) | [W4389326242](https://api.openalex.org/works/W4389326242) | 1,057 | 达到 |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | [2305.18290](https://arxiv.org/abs/2305.18290) | [W4378771755](https://api.openalex.org/works/W4378771755) | 293 | 该索引记录未达到 |
| GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | [1811.06965](https://arxiv.org/abs/1811.06965) | [W2901299405](https://api.openalex.org/works/W2901299405) | 255 | 该索引记录未达到 |

可复核的原始请求为 [Mamba DOI 查询](https://api.openalex.org/works/https://doi.org/10.48550/arxiv.2312.00752?select=id,title,cited_by_count,doi,locations)、[DPO DOI 查询](https://api.openalex.org/works/https://doi.org/10.48550/arxiv.2305.18290?select=id,title,cited_by_count,doi,locations) 和 [GPipe DOI 查询](https://api.openalex.org/works/https://doi.org/10.48550/arxiv.1811.06965?select=id,title,cited_by_count,doi,locations)。

这些计数只描述查询到的 OpenAlex 工作记录。DPO、GPipe 可作为机制图设计的补充参考，但本次证据不足以将它们按既定门槛列为“已核实高引用样本”；不能把索引记录的较低数值解释为论文总体影响力较低，也不应以个人印象补写更高引用数。
