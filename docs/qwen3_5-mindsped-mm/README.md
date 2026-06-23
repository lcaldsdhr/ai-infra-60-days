# Qwen3.5 + MindSpeed-MM 深度研究

> **路径 2 文档集**：分析 Qwen3.5 架构 + MindSpeed-MM 中的实现 + 创新点 + 主要算子

---

## 📚 文档索引

| 序号 | 文档 | 主题 | 状态 |
|------|------|------|------|
| **01** | [架构总览](./01-qwen3.5-architecture-overview.md) | Qwen3.5 整体架构（4 大创新、397B-A17B 详细参数） | ✅ |
| 02 | GDN 详解（待写） | Gated DeltaNet 数学 + PyTorch 实现 | ⬜ |
| 03 | Gated Attention 源码（待写） | QK Norm + Output Gate 完整实现 | ⬜ |
| 04 | mRoPE 详解（待写） | 多模态位置编码原理 + 实现 | ⬜ |
| 05 | 极致稀疏 MoE 工程（待写） | AllToAll + Token 调度 | ⬜ |
| 06 | MTP 多 Token 预测（待写） | 训练时多预测几个 token | ⬜ |
| 07 | MindSpeed-MM 实现（待写） | NPU 适配 + 性能优化 | ⬜ |
| 08 | Qwen3.5 vs Qwen3.6 演进对比（待写） | 版本差异分析 | ⬜ |

---

## 🎯 研究目标

1. **Qwen3.5 架构深度理解**
   - Hybrid Attention（线性注意力 GDN + 完整注意力交替）
   - MoE 架构（397B-A17B 极致稀疏）
   - RoPE / mRoPE / YaRN
   - Gated Attention 升级（QK Norm + Output Gate）
2. **MindSpeed-MM 中 Qwen3.5 的实现**
   - 配置文件解析
   - 模型代码组织
   - 训练流程
   - 权重转换（DCP 格式 ↔ HF 格式）
3. **创新点分析**
   - 相对标准 Transformer 的创新
   - 相对其他大模型（Llama / DeepSeek / Mistral / Kimi）的差异化
   - 在昇腾 NPU 上的工程创新
4. **主要算子梳理**
   - 自定义 kernel（FlashAttention 变种 / GDN 算子 / FLA / fused operator）
   - NPU 适配算子（AscendC / Triton）
   - 性能瓶颈与优化

---

## 📐 4 周时间规划

| 周次 | 主题 | 产出 |
|------|------|------|
| 第 1 周 | Qwen3.5 架构调研 | 架构图 + 关键论文笔记（✅ 已完成 01-架构总览） |
| 第 2 周 | MindSpeed-MM 仓库 + Qwen3.5 配置 | 仓库结构 + 配置文件解析 |
| 第 3 周 | Qwen3.5 核心组件源码精读 | 代码流程图 + 关键模块笔记 |
| 第 4 周 | 创新点 + 算子梳理 | 创新点清单 + 算子表 |

---

## 🔗 相关资源

### 官方资源

- [Qwen 官方 GitHub](https://github.com/QwenLM/Qwen3.5)
- [HF 模型集合 Qwen3.5](https://huggingface.co/collections/Qwen/qwen35)

### 关键论文

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

### 参考仓库

- MindSpeed-MM（本地）: `C:\Users\Administrator\Desktop\code\0611\MindSpeed-MM\`
- HF Transformers: `transformers/models/qwen3_5/`
