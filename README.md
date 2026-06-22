# AI Infra 60 天学习计划 🚀

> 从 PyTorch 单卡训练到分布式大模型训练 / 推理 / 服务化全栈 —— 60 天掌握 AI Infra（框架 + 算法）80-90% 知识

[![Phase](https://img.shields.io/badge/进度-Day%201--2%2F60-blueviolet)](#-进度看板)
[![Status](https://img.shields.io/badge/状态-进行中-success)](#-进度看板)
[![Focus](https://img.shields.io/badge/方向-AI%20Infra-orange)](#-学习范围)
[![Scope](https://img.shields.io/badge/范围-框架%20%2B%20算法-lightgrey)](#-学习范围)

---

## 📖 项目简介

本仓库记录一份 **60 天 AI Infra（框架 + 算法）系统学习计划** 的全过程：

- 📚 **理论常识**：AI Infra 全景、5D 并行、混合精度、KV Cache、量化、PEFT、推理引擎…
- 🛠 **工程实践**：基于 PyTorch / DeepSpeed / Megatron / vLLM 等真实代码的实操
- ✍️ **手写代码**：25 道高频面试手写题（Ring-AllReduce、AdamW、ColumnParallelLinear、RMSNorm…）
- 🎯 **面试导向**：覆盖跳槽中常见的工程实现题与系统设计题

**学习范围**：AI Infra 全栈（**不含算子开发**，算子有独立学习线）

---

## 📊 进度看板

| Phase | 主题 | 天数 | 状态 | 关键产出 |
|------|------|------|------|---------|
| **Phase 1** | 基础与单卡深度 | Day 1-10 | 🟡 Day 1-2 进行中 | AI Infra 全景图 / 单卡训练代码 |
| **Phase 2** | 分布式训练核心 | Day 11-20 | ⬜ 待开始 | 5D 并行实现 / 手写 Ring-AllReduce |
| **Phase 3** | 训练加速与算法 | Day 21-30 | ⬜ 待开始 | FlashAttention 复现 / 混合精度实战 |
| **Phase 4** | 推理与服务化 | Day 31-42 | ⬜ 待开始 | vLLM / KV Cache / Continuous Batching |
| **Phase 5** | 主流框架源码 | Day 43-52 | ⬜ 待开始 | DeepSpeed / Megatron / FSDP2 源码精读 |
| **Phase 6** | 系统设计与模拟面试 | Day 53-60 | ⬜ 待开始 | 8 道系统设计题 + 25 道手写代码题 |

> 图例：🟢 已完成 ｜ 🟡 进行中 ｜ ⬜ 待开始

---

## 🗂 目录结构

```
ai-infra-60-days/
├── README.md                                  ← 你正在看的文件（项目门面）
├── LICENSE                                    ← MIT License
├── .gitignore                                 ← Python 标准忽略规则
│
├── docs/
│   ├── 60-day-plan/                           ← 60 天完整学习计划
│   │   ├── README.md
│   │   ├── 01-phase1-基础与单卡深度.md
│   │   ├── 02-phase2-分布式训练核心.md
│   │   ├── 03-phase3-训练加速与算法.md
│   │   ├── 04-phase4-推理与服务化.md
│   │   ├── 05-phase5-主流框架源码.md
│   │   ├── 06-phase6-系统设计与模拟面试.md
│   │   ├── 07-手写代码清单.md                ← 25 道手写代码题
│   │   └── 08-资源与里程碑.md
│   │
│   └── daily-logs/                            ← 每日学习日志（Day 1-60）
│       ├── day-01-02-ai-infra-overview.md     ← ✅ Day 1-2 已完成
│       ├── day-03-04-xxx.md
│       └── ...
│
└── code/                                      ← 学习过程中的所有代码
    ├── README.md
    ├── phase1/                                ← Phase 1 单卡训练代码
    ├── phase2/                                ← Phase 2 分布式实现
    ├── phase3/
    ├── phase4/
    └── phase5/
```

---

## 🎯 学习范围

### ✅ 包含

| 模块 | 内容 |
|------|------|
| **训练框架** | PyTorch DDP / FSDP2 / DeepSpeed / Megatron-LM |
| **并行策略** | DP / TP / PP / SP / CP / EP（5D 混合并行） |
| **精度与显存** | Mixed Precision (AMP / BF16) / Gradient Accumulation / ZeRO |
| **注意力加速** | FlashAttention / FlashAttention-2 / PagedAttention |
| **推理引擎** | vLLM / TGI / TensorRT-LLM / llama.cpp |
| **KV Cache** | Paged KV Cache / Continuous Batching / Speculative Decoding |
| **量化** | INT8 / INT4 / GPTQ / AWQ / SmoothQuant |
| **PEFT** | LoRA / QLoRA / Adapter / Prefix-Tuning |
| **服务化** | Triton Inference Server / gRPC / Streaming Response |
| **手写代码** | Ring-AllReduce / AdamW / ColumnParallelLinear / RMSNorm / KV Cache 增量解码 … |

### ❌ 不包含

- **算子开发**（CUDA Kernel / Triton Kernel 底层优化）—— 独立学习线
- **ML 基础理论**（反向传播推导 / 优化器收敛性证明）—— 假设已有基础
- **数据 / 特征工程**（ETL / Feature Store）—— 偏 ML Ops，本计划不覆盖
- **业务应用层**（RAG / Agent / Prompt Engineering）—— 偏应用层

---

## 📈 阶段产出

每个 Phase 结束时应交付：

- [ ] 至少 1 个可运行的代码示例（`code/phaseN/` 下）
- [ ] 1 篇每日学习日志（`docs/daily-logs/` 下）
- [ ] 至少 3 道手写代码题（`code/phaseN/handwritten/` 下）
- [ ] 1 张知识图谱（可选，Mermaid / XMind 均可）
- [ ] 1 份自测清单（覆盖 Phase 全部关键点）

---

## 📚 知识图谱（顶层视图）

```
                     AI Infra 60 天
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    【训练】           【推理】           【服务化】
        │                 │                 │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │         │       │         │       │         │
 单卡深度  分布式   注意力加速  量化   推理引擎   在线服务
   │         │       │         │       │         │
 DDP      TP/PP    FlashAttn  GPTQ    vLLM     Triton
 AMP      CP/EP    PagedAttn  AWQ     TGI      gRPC
 ZeRO     ZeRO-3   SparseAttn SmoothQ TensorRT Streaming
 FSDP2    Hybrid   LinearAttn          llama.cpp
```

详细知识图谱见各 Phase 文档。

---

## 🚀 如何使用本仓库

### 1. 按 Phase 顺序学习

```bash
# 克隆到本地
git clone https://github.com/<your-name>/ai-infra-60-days.git
cd ai-infra-60-days

# 从 Phase 1 开始
cat docs/60-day-plan/01-phase1-基础与单卡深度.md
```

### 2. 每天的节奏（建议）

| 时段 | 任务 | 时长 |
|------|------|------|
| 🌅 早上 | 阅读当天 Phase 文档 + 知识图谱 | 30 min |
| 🌞 上午 | 跑通示例代码 / 复现论文 | 2-3 h |
| 🌆 下午 | 手写代码（每天 1 道） | 1-2 h |
| 🌙 晚上 | 写每日学习日志 | 30 min |

### 3. 提交规范

```bash
git add .
git commit -m "Day N: <phase> <topic>"
git push
```

---

## 📝 学习日志索引

| Day | 主题 | 状态 | 日志 |
|-----|------|------|------|
| Day 1-2 | AI Infra 全景 + 单卡训练 | 🟡 进行中 | [day-01-02-ai-infra-overview.md](docs/daily-logs/day-01-02-ai-infra-overview.md) |
| Day 3-10 | Phase 1 剩余 | ⬜ 待开始 | — |

---

## 🛠 环境要求

- **Python**：3.10+
- **PyTorch**：2.1+（含 FSDP2）
- **DeepSpeed**：0.14+
- **硬件建议**：
  - 单卡阶段：8GB+ 显存即可
  - 分布式阶段：2-4 张 A100/H100 / 昇腾 NPU
  - 无 GPU 环境：可使用 Colab / 阿里云 PAI / 华为云 ModelArts

---

## 📖 关联项目

本计划配套参考：

- [MindSpeed-MM](https://github.com/mindspore-lab/MindSpeed-MM) — 华为昇腾多模态训练套件
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [vLLM](https://github.com/vllm-project/vllm)

---

## 📜 License

[MIT License](LICENSE) — 自由使用、修改、分发

---

> 💡 **学习箴言**：AI Infra 不靠背，靠"跑代码 + 改源码 + 画图 + 讲给别人听"。
> 每天 4 小时 × 60 天 = 240 小时，足以把一个工程师从应用层推进到 Infra 核心层。
