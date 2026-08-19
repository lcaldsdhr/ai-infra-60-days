# Agentic RL：让语言模型在多步环境中学习行动

## 1. 概念边界：它不是单一算法

**Agentic RL（智能体强化学习）是问题范式/系统设计，不是一条与 GRPO、DAPO 并列的唯一 loss。**本文把它定义为：LLM 以 agent 身份在工具、代码执行器、浏览器、检索系统或模拟环境中多步行动，从环境反馈得到回报，再用 RL 更新可训练策略。2025 年的综述将其与常见单轮 LLM-RL 区分为时间延展的、部分可观测的 POMDP，并列举 planning、tool use、memory、reasoning 等能力维度。[*The Landscape of Agentic Reinforcement Learning for LLMs*](https://arxiv.org/abs/2509.02547)

为避免概念只停留在口号，本文以 Microsoft 的 **Agent Lightning** 为具体工程锚点：*Agent Lightning: Train ANY AI Agents with Reinforcement Learning*，Luo 等，[arXiv:2508.03680](https://arxiv.org/abs/2508.03680)，[官方代码](https://github.com/microsoft/agent-lightning)。它把 agent 执行建模为 MDP，用统一数据接口和 LightningRL 的 credit-assignment 模块将复杂 agent 轨迹拆为可训练 transition。[论文摘要与 §1](https://arxiv.org/abs/2508.03680)

## 2. 为什么单轮 RL 不够

单轮数学题 RL 通常可写为：`prompt → completion → scalar reward`。Agentic RL 则是：

```text
observation ot ─> policy πθ ─> action at (文本 / tool call / code)
       ^                              │
       └─ environment/tool feedback ──┘  重复至 success / timeout / budget exhaustion
                                      ↓
                         trajectory τ = (o0,a0,...,oT,aT), R(τ)
                                      ↓
                   credit assignment + policy optimization
```

状态常被上下文窗口近似，实际环境状态不可完全见；动作可能包含 JSON 参数、工具选择、检索 query、代码编辑或最终回答。因而训练不仅要处理 token 概率，还要定义终止条件、上下文截断、工具错误、延迟、成本和哪一步应为最终回报负责。综述将这种差别正式表述为从单步 MDP 到时间延展 POMDP 的变化。[综述摘要](https://arxiv.org/abs/2509.02547)

## 3. 核心目标与训练流程

给一条轨迹的回报可写为 `R(τ)=R_task + R_format - λ_cost·cost - λ_invalid·invalid`。并不是所有任务都需要逐步奖励：例如代码测试通过、SQL 执行正确或工具任务完成都可给终局 reward；但此时必须解决长轨迹中的 credit assignment。

一个可复现的在线 loop：

1. **定义环境契约**：观测、合法动作/工具 schema、停止条件、可自动验证的 reward、超时和资源预算。
2. **rollout**：冻结或版本化的策略与真实/隔离环境交互，记录每一步的输入 token、输出 token、tool request/response、状态摘要、耗时、错误和终局回报。
3. **轨迹校验与奖励**：过滤无效轨迹，执行测试/SQL/检索评测等 verifier；保留 reward 分解，不只存一个总分。
4. **信用分配**：把轨迹回报映射到 token 或 step 的 advantage；可使用 trajectory-level 回报、规则/模型 step reward，或由算法显式分段。Agent Lightning 的 LightningRL 是一例：其论文提出分层 RL 和 credit-assignment 模块，将任意 agent 的轨迹分解为 transition。[论文](https://arxiv.org/abs/2508.03680)
5. **策略更新与评测**：使用 PPO、GRPO 等 policy optimization；固定 benchmark、seed、工具版本和预算，随后仅把通过的 policy artifact 灰度进入下一轮 rollout。

Agent Lightning 的工程设计是 Training–Agent Disaggregation：agent 侧继续运行原有框架，观测/trace 进入统一 store，算法侧读取 trace 并写回策略权重或其他资源。它声称可接入 LangChain、OpenAI Agents SDK、AutoGen 或自建 agent；这是**该项目的能力声明**，不能替代你自己的兼容性测试。[论文](https://arxiv.org/abs/2508.03680)；[官方仓库架构说明](https://github.com/microsoft/agent-lightning)

## 4. 关键工程机制与超参数

| 机制 | 必须锁定/监控的项目 | 失控后果 |
| --- | --- | --- |
| 环境可重放 | 容器镜像、工具/API 版本、seed、网络策略、数据快照 | 同一策略得到不可比较的 reward |
| 轨迹语义 | `episode_id`、step 顺序、token 边界、tool 输入输出、终止原因 | 无法对齐 action 与 advantage，训练数据损坏 |
| 奖励 | 终局/过程 reward、格式惩罚、超时/费用惩罚、verifier 版本 | reward hacking：学会刷分而非完成任务 |
| 信用分配 | discount `γ`、GAE `λ`（若使用）、step/trajectory aggregation、优势归一化 | 长链任务梯度高方差，或把功劳归给错误步骤 |
| 探索与预算 | `temperature`、`top_p`、最大 step、最大 token、并发、每 episode 工具预算 | rollout 成本失控或探索不足 |
| 稳定更新 | learning rate、PPO clip / KL 系数、batch、更新轮数、reference policy | 策略漂移、工具格式退化、训练-推理 tokenization skew |

这些不是可照抄的全局默认值：与模型、任务稀疏度、工具延迟及并发系统耦合。特别是调用 OpenAI-compatible serving 时，训练器若重分词而不是使用 rollout 的原始 token IDs，可能引入 tokenization drift；应把 token IDs、chat template 与 server/model revision 纳入同一份 artifact 契约。[Agent Lightning 官方仓库收录的 vLLM 技术说明](https://github.com/microsoft/agent-lightning)

## 5. 优点、局限与失败模式

**适合的场景**：任务存在可自动检查的外部结果，且多步决策确实影响结果，如 Text-to-SQL 执行、RAG 检索路径、数学工具调用、代码测试。Agent Lightning 论文报告了 text-to-SQL、RAG、数学工具使用上的实验；这说明可行性，不等同于所有 agent 都会提升。[论文摘要](https://arxiv.org/abs/2508.03680)

**典型失败模式**：

- **reward hacking**：模型利用评分漏洞、伪造格式或调用捷径；应做对抗测试、隐藏测试与过程日志抽检。
- **稀疏且延迟的 reward**：长链任务大量全零轨迹，优势估计噪声很大；先缩短任务、加入可靠过程验证或课程式难度。
- **环境非平稳**：搜索索引、网页、API、工具版本变化使 reward 不可比；冻结快照或把环境版本写入轨迹。
- **off-policy / 异步陈旧性**：rollout policy 与更新后的 actor 相差过远会偏置更新；记录 policy version，限制 lag，必要时做重要性修正或丢弃陈旧轨迹。
- **安全与成本**：代码/浏览器/数据库工具必须隔离、限权、限时、限额；不要把生产凭证或真实写操作当作训练环境。
- **可观测性不足**：只存最终文本与总 reward，便无法诊断哪次 tool call、哪段 context 或哪条规则导致退化。

## 6. 与 verl 的学习路径

本仓库的 `verl` 是策略优化和高吞吐 rollout 基础设施；Agentic RL 则在其外部增加“多轮 agent loop + 环境 + 轨迹语义 + credit assignment”。建议按下列顺序阅读：

1. 先对照 [GRPO](grpo.md) 理解 group reward、advantage 与 policy update；
2. 阅读本地 verl 的 [Agentic RL 起步文档](../verl/docs/start/agentic_rl.rst)、[agent loop](../verl/docs/advance/agent_loop.rst) 与 [rollout trace](../verl/docs/advance/rollout_trace.rst)；
3. 再查看 [多轮 SGLang 文档](../verl/docs/sglang_multiturn/multiturn.rst) 和 [search-tool 示例](../verl/docs/sglang_multiturn/search_tool_example.rst)；
4. 最后把一个**无副作用、可 deterministic verifier 的小环境**接到 GRPO/PPO loop，先验证轨迹、reward、token 对齐，再扩展并发与真实工具。

固定源码版本由 [RL_go/docs/README.md](README.md) 记录的 `verl@commit` 决定；路径若随上游改动，应以该 gitlink 检出的版本为准。

## 7. 阅读与实验入口

- 概念地图：[Agentic RL 综述](https://arxiv.org/abs/2509.02547)。
- 一套可研究的架构/算法例子：[Agent Lightning 论文](https://arxiv.org/abs/2508.03680)、[官方代码与文档](https://github.com/microsoft/agent-lightning)。
- 本仓库源码对照：[verl `start/agentic_rl.rst`](../verl/docs/start/agentic_rl.rst)、[`agent_loop.rst`](../verl/docs/advance/agent_loop.rst)、[`search_tool_example.rst`](../verl/docs/sglang_multiturn/search_tool_example.rst)。

首个实验的验收不是“reward 上升”这一项：至少要能按 `episode_id` 重放一条轨迹，验证每次 tool call 合法、终局 reward 可重算、训练/rollout 模型与 tokenizer 版本一致，并在隐藏任务和固定预算下报告成功率、平均步数、平均 token、工具错误率与每成功一次的成本。
