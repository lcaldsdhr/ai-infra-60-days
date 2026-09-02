# LLM 强化学习训练栈

## 1. 把经典 RL 映射到语言模型

| RL 概念 | 单轮 LLM RL | Agentic RL |
| --- | --- | --- |
| state | Prompt + 已生成 token 前缀 | 对话、工具结果与环境观测 |
| action | 下一个 token | token、tool call 或语义动作 |
| transition | 追加 token | 工具/网页/代码环境返回新观测 |
| reward | 最终回答分数或 token reward | 成功、过程验证、成本与安全约束 |
| episode | 一条 completion | 到成功、失败、超时或预算耗尽的轨迹 |

## 2. 一次 LLM RL 迭代

```text
Prompt batch
  → rollout engine 每题采样一个或多个 response
  → reward model / rule verifier 评分
  → old log-prob、reference log-prob、Value（可选）
  → GAE / GRPO / RLOO 等 advantage estimator
  → PPO-style actor update
  → 同步新权重到 rollout engine
  → 下一轮
```

actor、rollout、reference、critic、reward model 是**逻辑角色**，不一定是五套独立硬件。verl/HybridFlow 的关键正是把逻辑数据流与每个角色内部的分布式执行组合起来。

## 3. RLHF、DPO、GRPO 不要混为一类

| 方法 | 数据从哪里来 | 是否在线 rollout | 是否显式 reward/critic |
| --- | --- | --- | --- |
| PPO-RLHF | 当前策略生成 + reward model | 是 | reward model；通常 critic |
| DPO | 已有 chosen/rejected 偏好对 | 否 | 无显式 reward model/critic |
| GRPO | 同一 Prompt 多条当前策略回答 + reward | 是 | reward/verifier；无独立 critic |
| SPIN | 人类/SFT 回答 vs 上轮策略生成 | 分轮生成 | 偏好式目标，无标准 critic |
| Agentic RL | 策略与工具/环境的多步轨迹 | 通常是 | verifier/环境 reward；算法可多样 |

反馈来源、优势估计器和优化目标是三个维度。例如 RLAIF 只说明偏好来自 AI，不自动等于 DPO 或 PPO。

## 4. Reward 是训练规格，不是一行函数

一个可审计 reward 应记录：输入字段、解析规则、verifier 版本、成功/失败/格式/超时分解、异常行为和测试用例。最低限度覆盖：正确答案、错误答案、空输出、格式异常、截断、恶意字符串与 verifier 超时。

```python
def reward(sample):
    # 不要只返回一个来源不明的标量；保留分解便于诊断。
    return {
        "task": check_answer(sample),
        "format": check_format(sample),
        "overlong": length_penalty(sample),
    }
```

高 reward 只说明模型更会满足当前评分器。隐藏测试、人评与对抗样例仍不可省略。

## 5. Token 级数据契约

训练 batch 至少要明确：`prompts`、`responses`、`input_ids`、`attention_mask`、`position_ids`、`response_mask`、`old_log_probs`、`rewards`、`advantages`，以及策略/Tokenizer 版本。工具观测、system/user tokens 与 padding 通常不应计入 policy loss。

沿固定 `verl@c4b389ad` 阅读：

1. [`main_ppo.py`](../../verl/verl/trainer/main_ppo.py)：配置、数据、worker 与 trainer 入口；
2. [`ray_trainer.py`](../../verl/verl/trainer/ppo/ray_trainer.py)：rollout → reward → advantage → update 编排；
3. [`core_algos.py`](../../verl/verl/trainer/ppo/core_algos.py)：GAE、GRPO、policy/value loss 与 KL；
4. [`vllm_rollout.py`](../../verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py)：推理、采样与权重同步。

更完整的数据流见 [Verl 训练生命周期](../verl-training-lifecycle.md)。

## 6. 最小验收

- 手算一个小 batch 的 reward、advantage、ratio 与 loss，并与实现一致；
- padding、Prompt 和环境 observation 不进入错误的 policy loss；
- reward 可单元测试、可重算并带版本；
- rollout policy version 可追踪，新权重同步后才进入下一批；
- 日志同时有 reward、KL、entropy、clip fraction、长度与外部验证质量。

一手资料：[InstructGPT](https://arxiv.org/abs/2203.02155)、[DPO](https://arxiv.org/abs/2305.18290)、[GRPO](https://arxiv.org/abs/2402.03300)、[HybridFlow/verl](https://arxiv.org/abs/2409.19256) 与[完整索引](../../docs/research/learning-curriculum-primary-sources.md)。
