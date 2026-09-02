# PPO Clip 最小实验

这个标准库实验直接计算：

```text
ratio = exp(current_log_prob - old_log_prob)
objective = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
```

## 运行

```powershell
python RL_go/code/ppo_clip_demo/demo.py
```

预期关键输出：

```text
positive / too high      1.50   1.00     1.50     1.20       1.20
negative / too low       0.50  -1.00    -0.50    -0.80      -0.80
```

这说明正优势动作的概率抬得过高后不再获得额外目标收益；负优势动作的概率压得过低时，也会按裁剪边界处理。实验末尾有断言验证两个边界。

## 限制

它只隔离 clipped surrogate，不包含 entropy bonus、Value loss、KL penalty、mask、minibatch 或 optimizer；真实 LLM PPO/GRPO 还必须验证这些部分及训推 log-prob 对齐。
