# GRPO 组相对优势：最小可运行实验

对应固定源码：[`verl@c4b389ad`](https://github.com/volcengine/verl/tree/c4b389adadc58ce51cb2b63e70df497ca166d77f) 中的 [`compute_grpo_outcome_advantage`](../../verl/verl/trainer/ppo/core_algos.py)。

本实验不加载模型、不使用 GPU，也不替代 Verl trainer。它只验证三个学习不变量：

1. 同一 prompt 的多条 completion 按组计算相对优势；
2. 固定版本按样本标准差（`torch.std` 默认行为）标准化；
3. 正优势 completion 在一次简化 policy-gradient 更新后概率上升，负优势 completion 概率下降。

## 运行

```powershell
python RL_go/code/grpo_advantage_demo/demo.py
```

## 预期结果

```text
== 1. One prompt, four rollouts ==
rewards:       [0.0, 1.0, 1.0, 0.0]
group mean:    0.500000
sample std:    0.577350
advantages:    [-0.866024, +0.866024, +0.866024, -0.866024]

== 2. One simplified policy-gradient ascent step ==
probability before: [+0.250000, +0.250000, +0.250000, +0.250000]
probability after:  [+0.223042, +0.276958, +0.276958, +0.223042]
```

数值末位可能因 `epsilon` 产生极小差异。真实 GRPO 还包含 token-level log-prob、response mask、PPO clip、可选 KL、minibatch/epoch 与分布式权重同步；这些被刻意排除，以便单独看清优势和更新方向。
