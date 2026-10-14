# Agent 动作 mask：CPU 教学实验

课程：[数据与 mask 正确性](../../blog/verl-data-and-masks.md)。参考 verl 固定快照 `c4b389adadc58ce51cb2b63e70df497ca166d77f` 的 AgentLoop 与 policy loss 路径。

环境：Python 3.10+ 标准库，无 torch、Ray、GPU、模型或数据集依赖。

```powershell
python RL_go/code/agent_masks_demo/demo.py
```

预期输出：

```text
response_mask=[1, 1, 0, 0, 1, 1, 0]; valid actions=4
correct action mean=2.5; padding-only mean=35.0
changed tool-position loss; action mean unchanged
empty action mask: rejected (no valid action tokens)
PASS: context/action distinction and valid-token reduction
```

2026-10-14 本地 Python 3.13 运行通过。限制：只验证给定 mask 下的标量归约，不计算 attention、token logits、真实 PPO loss 或梯度。工具位置的 100、padding 的 999 是教学值。实际后端可能在计算无效位置 loss 前就跳过这些位置，也须处理非有限值，不能认为 `0 * NaN` 自动安全。全零 mask 的错误策略是教学选择，不宣称所有框架都以异常退出。
