# 2026-06-22 进度快照

> **本文件用途**：记录"两条学习路径并行"启动时的状态快照。

---

## 📊 当前状态总览

| 维度 | 状态 |
|------|------|
| **日期** | 2026-06-22（周日） |
| **路径 1** | AI Infra 60 天学习计划 — Day 1 上午完成 |
| **路径 2** 🆕 | Qwen3.5 + MindSpeed-MM 深度研究 — **启动中** |
| **GitHub 仓库** | https://github.com/lcaldsdhr/ai-infra-60-days |

---

## ✅ 路径 1：AI Infra 60 天（Day 1 上午完成）

### 已完成（7 项）
- [x] 读 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [x] 整理核心概念笔记：Q/K/V / √d_k / Multi-Head / Mask / RoPE
- [x] 5 个核心点笔记质量评估：5/5 / 5/5 / 5/5 / 4/5（缺 padding mask）/ 5/5
- [x] 补漏：Mask 完整解析（Causal + Padding 两类 + 训练-推理一致性）
- [x] 答 3 道面试验证题 + 满分答案 + 变形题
- [x] nanoGPT CausalSelfAttention 精讲（4 维 reshape / 双路径 mask / QKV 一次性算）
- [x] 补 `~pad_mask` 和 `torch.matmul` vs `@` 细节

### 关键笔记
1. **Q/K/V 角色**：Q=搜索词 / K=标签 / V=内容
2. **√d_k 原因**：Q·K^T 方差 d_k，不除以会导致 softmax 极端、梯度消失
3. **Multi-Head**：多个 head 学不同语义子空间（句法 / 指代 / 远程依赖）
4. **Mask 两类**：Padding mask（屏蔽 <pad>）+ Causal mask（屏蔽未来）
5. **RoPE**：旋转 Q/K 让内积天然只依赖相对位置差 (m-n)，强外推性
6. **Decoder-Only 主流**：BERT/T5 vs GPT/Llama/Qwen，causal mask + 训练-推理一致

### 暂停点
- 用户切换到路径 2，路径 1 挂起

### 待继续（路径 1 剩余）
- [ ] ⏸ 手写概念对照表（训练 vs 推理 / DP vs MP / 延迟 vs 吞吐）
- [ ] ⏸ 面试卡 1：训练大模型的 4 个难点 + 4 类方案
- [ ] ⏸ 面试卡 2：为什么需要分布式训练
- [ ] ⏸ Day 1 复盘
- [ ] ⏸ Day 2 实践：手写 BERT-base SFT 训练脚本（含 EMA + Eval 钩子）

---

## 🆕 路径 2：Qwen3.5 + MindSpeed-MM 深度研究（启动）

### 研究目标
1. **Qwen3.5 架构深度理解**
   - Hybrid Attention（线性注意力 GDN + 完整注意力交替）
   - MoE 架构（35B-A3B 等）
   - RoPE / NTK-aware / YaRN
   - DeepStack / DeepFusion（多模态融合机制）
   - Qwen3-VL 的视觉编码器
2. **MindSpeed-MM 中 Qwen3.5 的实现**
   - 配置文件解析
   - 模型代码组织
   - 训练流程
   - 权重转换（DCP 格式 ↔ HF 格式）
3. **创新点分析**
   - 相对标准 Transformer 的创新
   - 相对其他大模型（Llama / DeepSeek / Mistral）的差异化
   - 在昇腾 NPU 上的工程创新
4. **主要算子梳理**
   - 自定义 kernel（FlashAttention 变种 / GDN 算子 / fused operator）
   - NPU 适配算子
   - 性能瓶颈与优化

### 初步时间规划（4 周）
| 周次 | 主题 | 产出 |
|------|------|------|
| 第 1 周 | Qwen3.5 架构调研 | 架构图 + 关键论文/技术报告笔记 |
| 第 2 周 | MindSpeed-MM 仓库结构 + Qwen3.5 配置文件 | 仓库结构文档 + 配置解析 |
| 第 3 周 | Qwen3.5 核心组件源码精读 | 代码流程图 + 关键模块笔记 |
| 第 4 周 | 创新点 + 算子梳理 | 创新点清单 + 算子表 |

### 文档产出
- 1 个总览文档（README.md）
- 4 个专题文档（架构 / 实现 / 创新 / 算子）
- N 个工作日志（持续记录）
- 1 张 Qwen3.5 架构 ↔ MindSpeed-MM 实现映射图

### 起点
- 探索 MindSpeed-MM 仓库结构
- 找到 Qwen3.5 相关配置 / 代码
- 读 Qwen3.5 技术报告（Qwen Team 官方）
- 启动文档框架

---

## 🎯 下次继续

### 立即开始
- **路径 2**：探索 MindSpeed-MM 仓库结构 + 定位 Qwen3.5 实现
- **目标**：1 小时内输出仓库结构图 + Qwen3.5 核心文件清单

### 路径 1（择机继续）
- Day 1 剩余任务（手写概念对照表 + 面试卡 + 复盘）
- 建议每天 30 min 推进，避免遗忘

---

## 📅 时间分配建议

| 任务 | 时长 | 频率 |
|------|------|------|
| 路径 1（AI Infra） | 2h / 天 | 周末集中推进 |
| 路径 2（Qwen3.5 + MindSpeed-MM） | 2-3h / 天 | 工作日主线 |
| 复盘 + 提交 | 30 min / 天 | 每日 |

---

> 💡 **核心策略**：路径 1 慢学慢练（建立深度），路径 2 集中突击（产出文档）。
> 两条路径通过"分布式训练"等主题可以相互促进。
