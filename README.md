# AI Infra & RL 学习仓库

本仓库将 AI Infra 与 RL / verl 两条学习线分开管理，同时保留可复用的课程与 Agent 协作模板。

## 目录

```text
.
├── AI_infra/                         # 原有 AI Infra 学习计划、博客、文档与代码
├── RL_go/                            # RL / verl 学习线
│   ├── verl/                         # 固定 commit 的 verl Git submodule
│   ├── blog/                         # 对算法、源码和工程设计的解析文章
│   ├── docs/                         # 学习笔记、源码导读与实验记录
│   └── code/                         # 可运行的 RL / verl 实验
├── templates/learning-and-agent/     # 可复用课程与 Agent 协作模板
├── AGENTS.md                         # 仓库协作规范
└── SKILLS_SUMMARY.md                 # 已安装 Skill 的使用速览
```

## 学习入口

- [AI Infra 60 天计划](AI_infra/README.md)
- [RL / verl 学习线](RL_go/README.md)
- [课程与 Agent 模板](templates/learning-and-agent/README.md)

## 初始化 verl 源码

`RL_go/verl` 固定到父仓库记录的上游提交。首次克隆后执行：

```bash
git submodule update --init RL_go/verl
```

具体版本与更新流程见 [RL_go/docs](RL_go/docs/README.md)。
