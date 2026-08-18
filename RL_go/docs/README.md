# verl / RL 学习线

本目录承载以 **verl** 为主线的强化学习训练基础设施学习记录，重点覆盖数据流、rollout、奖励、训练器、分布式执行与可观测性。

## 固定源码版本

| 项目 | 上游仓库 | 固定提交 | 本地路径 |
| --- | --- | --- | --- |
| verl | https://github.com/volcengine/verl | `c4b389adadc58ce51cb2b63e70df497ca166d77f` | `RL_go/verl` |

该路径是 Git submodule。首次克隆本仓库后执行：

```bash
git submodule update --init RL_go/verl
```

若仅更新上游源码而未更新父仓库的 gitlink，不应把它视为本课程可复现的版本。

## 建议学习顺序

1. 建立 PPO、GRPO 等 RL 训练循环的输入/输出认知。
2. 从 `RL_go/verl` 的训练入口追踪配置加载、worker 初始化和资源分配。
3. 阅读 rollout、奖励计算和优势估计的数据流。
4. 对照分布式训练、checkpoint、容错和指标记录实现。
5. 为每个主题按 [课程笔记模板](../../templates/learning-and-agent/lesson-template.md) 记录可复现命令和结论。

## 版本更新流程

1. 记录要升级到的 `verl@commit` 与原因。
2. 更新 submodule gitlink。
3. 复查课程笔记中的源码路径、配置示例和兼容性说明。
4. 提交时同时更新本表和验证结果。
