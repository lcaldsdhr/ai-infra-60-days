# AI Infra 60 天学习计划（框架 + 算法）

> **目标人群**：有一定 PyTorch 基础、希望在 60 天内系统掌握 AI Infra（分布式训练框架 + 训练/推理算法）80%~100% 知识图谱，并具备跳槽面试工程实现与手写代码能力的学习者。
>
> **范围界定**：本计划**不深入单算子**（不要求从零写 CUDA/Triton kernel），而是**框架（PyTorch DDP / FSDP / DeepSpeed / Megatron）+ 训练/推理算法**。
>
> **学习形式**：理论（论文 + 文档）→ 工程实践（基于 PyTorch 写小 demo）→ 源码解读（关键路径）→ 面试手写代码（白板级实现）。

## 一、文档结构

```text
docs/zh/learning-plan/
├── README.md                    ← 本文件（总索引 + 总体节奏 + 资源 + 检验标准）
├── 01-phase1-基础与单卡深度.md   ← Day 01-10
├── 02-phase2-分布式训练核心.md   ← Day 11-25
├── 03-phase3-训练加速与算法.md   ← Day 26-40
├── 04-phase4-推理与服务化.md     ← Day 41-50
├── 05-phase5-主流框架源码.md     ← Day 51-57
├── 06-phase6-系统设计与面试.md   ← Day 58-60
├── 07-手写代码清单.md            ← 25 道面试手写题（含要点）
└── 08-资源与里程碑.md            ← 论文/课程/项目 + 阶段验收标准
```

> **使用方式**：从 README 开始 → 按 Phase 顺序学习 → 每周翻 07-手写代码清单 巩固 → 每个 Phase 结束对照 08-资源与里程碑 自检。

## 二、总体节奏（60 天 / 6 阶段）

```text
Phase 1:  Day 01-10   基础与单卡深度（地基）
Phase 2:  Day 11-25   分布式训练核心（数据并行/张量并行/流水并行/序列并行）
Phase 3:  Day 26-40   训练加速与算法（优化器/注意力/量化/重计算）
Phase 4:  Day 41-50   推理与服务化（KV Cache/Continuous Batching/vLLM/TensorRT-LLM）
Phase 5:  Day 51-57   主流框架源码（DeepSpeed/Megatron-LM/FSDP/vLLM）
Phase 6:  Day 58-60   系统设计 + 模拟面试
```

## 三、每日固定作息建议（在职党）

| 时段 | 时长 | 任务 |
| --- | --- | --- |
| 早 / 通勤 | 30 min | 论文或博客（理论） |
| 午休 | 30 min | 刷面试题卡片 |
| 下班后 | 2.5 ~ 3 h | 视频课 + 写代码 + 调实验 |
| 周末 | 4 ~ 6 h | 集中做 mini-project 或源码解读 |

## 四、知识地图（鸟瞰）

```text
AI Infra 知识图谱
├── 基础层
│   ├── PyTorch 内部（autograd、nn.Module、torch.compile）
│   ├── AMP（FP16 / BF16 / TF32 / Loss Scaling）
│   └── Profiling（torch.profiler、Roofline）
├── 分布式训练
│   ├── 集合通信（NCCL、Ring/Tree AllReduce、AllGather、ReduceScatter）
│   ├── 数据并行（DP / DDP / FSDP1 / FSDP2 / ZeRO-1/2/3）
│   ├── 模型并行（TP / PP / SP / CP / EP）
│   ├── 混合并行（4D / 5D）
│   └── 显存优化（Recompute / Offload / ZeRO-Infinity）
├── 训练算法
│   ├── 优化器（SGD / Adam / AdamW / Lion / Muon）
│   ├── LR 调度（warmup / cosine / linear）
│   ├── 注意力（SDPA / Flash / Linear / Gated Delta Net / Mamba）
│   ├── 归一化（LayerNorm / RMSNorm）
│   ├── 位置编码（RoPE / ALiBi / YaRN）
│   ├── 量化（INT8 / INT4 / FP8 / GPTQ / AWQ）
│   └── PEFT（LoRA / QLoRA / DoRA / AdaLoRA）
├── 推理与服务化
│   ├── KV Cache（MQA / GQA / MLA / PagedAttention）
│   ├── 调度（Static / Dynamic / Continuous Batching）
│   ├── 加速（Speculative / Medusa / EAGLE）
│   └── 引擎（vLLM / TensorRT-LLM / SGLang / LMDeploy）
├── 框架源码
│   ├── PyTorch DDP / FSDP
│   ├── DeepSpeed（ZeRO-1/2/3 / Offload）
│   ├── Megatron-LM（TP / PP / VPP）
│   └── vLLM（PagedAttention / Scheduler / ModelRunner）
└── 面试能力
    ├── 概念题（5D 并行 / Flash / MoE / vLLM）
    ├── 手写代码（25 题）
    └── 系统设计（4 道经典题）
```

## 五、关键里程碑 & 检验标准

| 阶段 | 通过标准 |
| --- | --- |
| Phase 1 结束 | 能用 PyTorch 从零写一个 AMP 训练循环；解释清楚 FP16/BF16 区别与显存组成。 |
| Phase 2 结束 | 能讲清 5D 并行（TP/PP/SP/CP/EP）的通信模式与适用场景；能手写 ZeRO-1。 |
| Phase 3 结束 | 能从零写一个带重计算 + AdamW + cosine warmup 的训练 step；能讲清 Flash Attention 思想。 |
| Phase 4 结束 | 能讲清 vLLM 的 PagedAttention + Continuous Batching；能手写 KV Cache 增量推理。 |
| Phase 5 结束 | 能在 30 分钟内画出 DDP/DeepSpeed/Megatron/vLLM 四大框架的核心类图与调用关系。 |
| Phase 6 结束 | 通过模拟面试 12 轮（概念题 5 + 手写题 5 + 系统设计 2）。 |

## 六、推荐每日复盘模板

```markdown
## Day 03 - PyTorch 深度
- 学到的概念 1: ...
- 学到的概念 2: ...
- 写了什么代码: ... (贴 GitHub 链接)
- 面试题: ... (写下自己的答案)
- 明天要做的: ...
```

> **坚持 60 天**，掌握 80%~100% 的 AI Infra 知识图谱（框架 + 算法），并具备面试工程实现与手写代码能力。

## 七、Q&A

**Q1：60 天够吗？**
A：紧但可行（按"理论 30% + 实践 50% + 源码 20%"分配）。如果时间宽裕，建议 90 天；60 天版适合在职党集中冲刺。

**Q2：要不要学 CUDA / 算子？**
A：本计划**明确不深入算子**。但建议用 `triton` 写 1~2 个简化 kernel（如 SDPA、LayerNorm）帮助理解 IO-aware 思想，对面试加分明显。

**Q3：先看论文还是先写代码？**
A：**先写代码再读论文**。先动手实现一个朴素版本（哪怕错的），再去看论文为何这样优化，理解会更深。

**Q4：手写代码题要不要背？**
A：**不要背，要理解**。面试官会让你现场改（如加一个 `dtype` 参数、换通信策略），背下来的代码无法应对变体。

**Q5：要不要刷 LeetCode？**
A：AI Infra 面试**比算法岗少刷 1/3 难度**（medium 为主），但**手写代码题比算法题更吃工程经验**。建议每天 1 题 LeetCode + 1 题 AI Infra 手写题，交替进行。

## 八、跳转链接

- [01-Phase 1：基础与单卡深度（Day 01-10）](./01-phase1-基础与单卡深度.md)
- [02-Phase 2：分布式训练核心（Day 11-25）](./02-phase2-分布式训练核心.md)
- [03-Phase 3：训练加速与算法（Day 26-40）](./03-phase3-训练加速与算法.md)
- [04-Phase 4：推理与服务化（Day 41-50）](./04-phase4-推理与服务化.md)
- [05-Phase 5：主流框架源码（Day 51-57）](./05-phase5-主流框架源码.md)
- [06-Phase 6：系统设计与模拟面试（Day 58-60）](./06-phase6-系统设计与模拟面试.md)
- [07-手写代码清单（25 题）](./07-手写代码清单.md)
- [08-资源与里程碑](./08-资源与里程碑.md)
