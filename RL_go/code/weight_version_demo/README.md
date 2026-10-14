# 权重版本与同步：CPU 教学实验

对应课程：[第 1 课：权重版本与训推同步](../../blog/verl-weight-version-and-sync.md)。

## 环境与运行

- Python 3.10+，仅标准库；无 GPU、torch、Ray 或模型权重依赖。
- 参考 verl 快照：`c4b389adadc58ce51cb2b63e70df497ca166d77f`。
- 从仓库根目录执行：

```powershell
python RL_go/code/weight_version_demo/demo.py
```

## 预期输出

```text
history: version=17, weights=(0.6, 0.4)
actor updated, rollout stale: rejected (version mismatch)
only changed labels: rejected (same label, different weights)
only synced replica 0: rejected (same label, different weights)
new request: version=18, weights=(0.75, 0.25)
PASS: all replicas synced; historical provenance unchanged
```

## 已知限制

这是教学状态模型，不执行 token 采样、梯度训练、网络传输或分布式故障恢复。两项浮点值仅模拟参数，不是大模型；精确 tuple 比较只用于 CPU 示例。真实后端需要核对所有 worker 的传输/应用完成、layout、dtype、cache 和请求生命周期。示例的暂停开关不模拟清空已在途请求，更不证明跨 replica 的全局原子切换。

## 验证记录

2026-10-14：在本地 Python 3.13 运行，以上故障分支均被拒绝，最终 PASS。未运行真实 verl/GPU 实验。
