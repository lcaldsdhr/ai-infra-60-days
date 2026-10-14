# RL / verl 解析文章

将面向理解的文章放在此目录，例如：

- PPO、GRPO、优势估计和奖励建模的机制解析；
- verl 的训练入口、rollout、worker 和资源调度源码导读；
- 分布式训练、checkpoint、容错与可观测性的工程复盘。

文章应在开头记录适用的 `verl@commit` 与相关实验或源码链接。

## 已收录

### RL 系统基础课程

- [第 1 课：权重版本与训推同步](verl-weight-version-and-sync.md)：图像、固定版本源码、CPU 教学实验与课后题；本课问答完成，真实 GPU 同步未验证。
- [异步延伸：经验队列源码导读](verl-async-queue-code-walkthrough.md)：区分待生成任务和已完成经验，阅读生产/消费、满队列淘汰与版本字段；课后题待回答。
- [第 2 课：数据与 mask 正确性](verl-data-and-masks.md)：Agent 工具轨迹、attention 与动作 mask、真实源码和 CPU 归约实验；课后题待回答。
- 后续学习：checkpoint 与恢复能力 → 按瓶颈选择异步、共置与 packing。

### 文章与复习

- [Verl 训练生命周期：从 DataProto 到 rollout 与 actor 更新](verl-training-lifecycle.md)
- [训练生命周期 Q&A：样本对齐、训练循环、策略概率与异步训推](verl-lifecycle-qa.md)
- [LLM 推理专题](inference/README.md)
- [强化学习专题](reinforcement-learning/README.md)
- [大模型训练与通用技术专题](training/README.md)
- [特性对比图谱：Why / Without / With / Trade-off](feature-comparison-atlas.md)
