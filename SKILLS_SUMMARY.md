# Codex Skills 使用速览

本文记录当前已安装的常用 Skill 及其适用场景。

## 编码通用纪律

| Skill | 适用场景 |
| --- | --- |
| `karpathy-guidelines-for-codex` | 编码、修复、重构和评审时约束工作方式：先判断，再做最小改动，并定义可验证的成功标准。 |

## 工程开发工作流

| Skill | 适用场景 |
| --- | --- |
| `implement` | 按既定规格完成工程实现。 |
| `tdd` | 希望以测试驱动方式实现功能或修复缺陷。 |
| `prototype` | 快速制作一次性原型，用于验证交互、状态模型或技术方案。 |
| `to-spec` | 将想法或问题收敛成清晰、可执行的规格。 |
| `to-tickets` | 将规格拆解为可跟踪的任务项。 |
| `triage` | 对任务或缺陷分类、判断优先级。 |
| `wayfinder` | 在复杂工程任务中确定下一步与执行路径。 |
| `wizard` | 生成需要人类完成的交互式配置/迁移操作向导。 |

## 诊断与质量

| Skill | 适用场景 |
| --- | --- |
| `diagnosing-bugs` | 定位报错、功能异常、测试失败和性能回退；强调复现、证据与根因验证。 |
| `code-review` | 评审分支、PR 或工作区改动，检查规范与需求符合度。 |
| `grill-with-docs` | 依据项目文档审视方案或实现，发现不一致、遗漏与风险。 |
| `resolving-merge-conflicts` | 处理 Git merge 或 rebase 过程中的冲突。 |

## 架构与知识沉淀

| Skill | 适用场景 |
| --- | --- |
| `codebase-design` | 设计或优化模块边界、接口深度与可测试性。 |
| `domain-modeling` | 梳理业务术语、领域模型，并记录 Context 或 ADR。 |
| `research` | 基于高可信一手资料研究主题，并将结论写入仓库 Markdown。 |
| `handoff` | 整理任务交接：进度、修改、验证方法、风险与下一步。 |
| `ask-matt` | 按 Matt Pocock 的工程方法选择合适的工作流或提出问题。 |

## 文档与办公产物

| Skill | 适用场景 |
| --- | --- |
| `docx` | 创建、读取和编辑 Word 文档或模板。 |
| `pptx` | 创建、读取和编辑原生 PowerPoint 演示文稿。 |
| `xlsx` | 创建、清洗、分析和编辑 Excel、CSV、TSV 等表格。 |
| `pdf` | 读取、生成、审查 PDF，尤其是需要关注版式时。 |

## 视觉表达

| Skill | 适用场景 |
| --- | --- |
| `guizang-ppt-skill` | 生成单文件 HTML 横向翻页网页 PPT；适合杂志风叙事、瑞士风数据/产品汇报、社媒封面。 |
| `imagegen` | 生成或编辑位图视觉资产，例如配图、封面和产品图。 |
| `sora` | 使用 Sora 生成、改编或下载视频。 |
| `figma` | 从 Figma 获取设计上下文、资产和变量，并转化为实现。 |

## 常用组合

- 报错或性能问题：`diagnosing-bugs`
- 按最佳实践完成开发并补测试：`karpathy-guidelines-for-codex` + `implement` + `tdd`
- 评审当前改动：`code-review`
- 把文章制作成瑞士风网页演示：`guizang-ppt-skill`
- 生成可编辑 PowerPoint：`pptx`
- 暂停任务并让其他人或 Agent 继续：`handoff`

## 说明

`guizang-ppt-skill` 输出网页演示稿（HTML），而 `pptx` 面向原生 PowerPoint 文件；根据交付格式选择其一。
