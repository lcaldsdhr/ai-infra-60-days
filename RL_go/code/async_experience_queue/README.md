# 异步经验队列 CPU 实验

对应[异步队列源码导读](../../blog/verl-async-queue-code-walkthrough.md)。参考 `verl@c4b389adadc58ce51cb2b63e70df497ca166d77f` 的实验路径 `MessageQueue`，本代码独立实现教学子集，不导入上游代码。

依赖：Python 3.10+ 标准库，无 GPU、torch、Ray、模型或数据集要求。

```powershell
python RL_go/code/async_experience_queue/demo.py
```

预期输出：

```text
rollout now v18; consumed A from v17
full queue put C returned False; retained [B, C]
waiting consumer resumed with D; remaining=0
PASS: provenance, oldest eviction, and asynchronous waiting
```

2026-10-14：本地 Python 3.13 运行通过。限制：只验证队列等待、淘汰与来源不变，不模拟 token 生成、RPC、关闭协议、参数传输、真实 Trainer 或训练质量。异步不自动等于 GPU 计算重叠；本例也没有通过 sleep 人为测量性能。
