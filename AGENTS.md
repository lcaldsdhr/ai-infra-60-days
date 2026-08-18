# AI Infra 与 RL 学习仓库协作说明

## 目标与结构

本仓库用于沉淀可运行的 AI Infra、RL 与 verl 学习材料，并按学习线隔离存储。

- `AI_infra/`：既有 AI Infra 学习计划、博客、文档、图表和代码。
- `RL_go/`：RL / verl 学习线；`blog/`、`docs/`、`code/` 分别存放解析文章、学习笔记和实验。
- `RL_go/verl`：固定版本的 verl 外部源码，仅用于阅读、检索与对照。
- `templates/learning-and-agent/`：跨学习线复用的课程和 Agent 协作模板。

## 新增一节课程

1. 基于 `templates/learning-and-agent/lesson-template.md` 创建笔记。
2. AI Infra 代码放入 `AI_infra/code/`；RL / verl 实验放入 `RL_go/code/`，并在笔记中链接它。
3. 记录版本、硬件前提、运行命令、预期输出和已知限制。
4. 在相关索引或 README 中增加入口。

## verl 与外部源码

- `RL_go/verl` 是 `volcengine/verl` 的 Git submodule，版本以父仓库记录的 gitlink 为准。
- 不直接修改 submodule 中的源码；需要更新时，明确说明目标上游提交、兼容性影响与验证结果。
- 不提交模型权重、数据集、检查点、API 密钥或本地实验产物。

## 验证与提交

- 文档改动至少执行 `git diff --check` 并检查 Markdown 链接。
- 代码改动运行最小相关示例或测试；无法运行时说明原因。
- 保持提交聚焦，避免混入本地未跟踪实验目录或无关格式化。
