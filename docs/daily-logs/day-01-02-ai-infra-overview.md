# Day 1-2 工作日志：AI Infra 全景与生态

> **本文件用途**：每天复盘模板，记录读完/写完/答完的内容；完成后打勾。
>
> **学习节奏**：在职党每天 3h（理论 1h + 实践 2h），周末可拉长到 4-6h。

---

## 一、本日目标

完成"AI Infra 全景"地图的搭建，建立对**训练 vs 推理、数据并行 vs 模型并行、延迟 vs 吞吐**等核心概念的心智模型。

- [ ] **理论**：读完 3 篇核心材料 + 整理概念对照表
- [ ] **实践**：跑通 BERT-base SFT 训练脚本（含 EMA + Eval 钩子）
- [ ] **面试**：能口头答出 2 道概念卡（5 分钟内）

---

## 二、理论（Day 1 上午 / 1.5h）

### 2.1 必读材料

| # | 材料 | 时长 | 重点章节 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 30 min | 全文 | ☐ |
| 2 | [Andrej Karpathy "Zero to Hero" - Intro 视频 + "Let's build GPT"](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | 45 min | Intro + 前 2 集 | ☐ |
| 3 | [Lilian Weng 博客](https://lilianweng.com/) 主页浏览 | 15 min | 看文章列表，**先不读细节**，标记 3-5 篇感兴趣的长文 | ☐ |

### 2.2 概念对照表（自己手写一遍，不要看现成答案）

```markdown
## 训练 vs 推理
| 维度 | 训练 | 推理 |
| --- | --- | --- |
| 计算量 | 大（前向 + 反向 + 优化器步进） | 小（仅前向） |
| 显存占用 | 模型 + 梯度 + 优化器 + 激活 + 临时 | 模型 + KV Cache（生成式） |
| 通信 | 数据并行 all-reduce / TP all-reduce / PP send-recv | 极小（除了 TP） |
| 延迟 | 不敏感（要吞吐） | 敏感（用户体感） |
| 吞吐 | 优先 | 次之 |
| 异常容忍 | 低（一个 NaN 全部废） | 高（单请求失败不致命） |

## 数据并行 vs 模型并行
| 维度 | 数据并行 (DP/DDP) | 模型并行 (TP/PP) |
| --- | --- | --- |
| 切分对象 | 样本 | 模型权重 / 激活 |
| 通信频率 | 每 step 一次 all-reduce | 每层都通信 |
| 显存节省 | 不省参数 | 显著省 |
| 扩展性 | 受 batch 大小限制 | 受模型规模限制 |
| 适用 | 中小模型 / 数据量大 | 大模型 |

## 延迟 vs 吞吐
| 维度 | 延迟 (Latency) | 吞吐 (Throughput) |
| --- | --- | --- |
| 含义 | 单请求耗时 | 单位时间处理量（QPS / TPS） |
| 单位 | ms | req/s / token/s |
| 优化方向 | 减关键路径、KV Cache、Speculative | 增 batch、FSDP、Continuous Batching |
| 场景 | 实时对话 | 离线批处理 |
```

- [ ] 把上面这张表**用 30 分钟自己手写一遍**（关掉参考答案，闭卷）
- [ ] 然后对比原文，补齐遗漏项

### 2.3 读 Karpathy "Let's build GPT" 的小作业

看完视频后，用**不超过 20 行 PyTorch** 写一个 nanoGPT 风格的 decoder block，验证你真的看懂了：

```python
# 参考骨架（不要直接抄，自己写）
class NanoBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x])[0]
        x = x + self.mlp(self.ln2(x))
        return x
```

---

## 三、实践（Day 2 上午 / 2h）

### 3.1 环境准备（首次需要 30 min）

```bash
# 1) 创建虚拟环境（PowerShell）
python -m venv .venv-ai-infra
.venv-ai-infra\Scripts\Activate.ps1

# 2) 安装依赖
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install evaluate scikit-learn pandas tensorboard

# 3) 验证 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

> **注意**：如果只有 NPU / 集成显卡，本步骤可改为 `pip install transformers datasets accelerate` 跳过 CUDA 验证；后续会切到 NPU 训练。

### 3.2 数据准备（30 min）

```python
# 1) 用 HuggingFace datasets 加载 GLUE/SST-2（小数据集，CPU 即可下载）
from datasets import load_dataset
raw = load_dataset("glue", "sst2")
print(raw)
# 输出：
# DatasetDict({
#     train: Dataset({...features: ['sentence', 'label', 'idx']...})
#     validation: Dataset({...})
#     test: Dataset({...})
# })
```

> 数据集选 **SST-2** 的原因：5 万条样本、二分类、token 数 < 50，单卡几分钟就能跑完一个 epoch。

### 3.3 训练脚本（90 min，自己手写）

> **目标**：实现"含 EMA + Eval 钩子"的 SFT 训练脚本。**不要直接抄 run_glue.py，自己手写一遍。**

```python
# 作业版训练脚本（你需要自己补全）
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset

# ===== 1) 数据 =====
raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def collate(batch):
    texts = [b["sentence"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    return {**enc, "labels": labels}

train_loader = DataLoader(raw["train"], batch_size=32, shuffle=True, collate_fn=collate)
eval_loader  = DataLoader(raw["validation"], batch_size=64, collate_fn=collate)

# ===== 2) 模型 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.999))

# ===== 3) 优化器 + 调度器 =====
optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * 2  # 2 epoch
sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

# ===== 4) Eval 钩子 =====
@torch.no_grad()
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        pred = out.logits.argmax(-1)
        correct += (pred == batch["labels"]).sum().item()
        total += len(batch["labels"])
    model.train()
    return correct / total

# ===== 5) 训练循环 =====
model.train()
global_step = 0
for epoch in range(2):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ★ 梯度裁剪
        optim.step()
        sched.step()
        optim.zero_grad()
        ema_model.update_parameters(model)  # ★ EMA 更新

        if global_step % 100 == 0:
            acc_raw = evaluate(model)
            acc_ema = evaluate(ema_model)
            print(f"step {global_step}: loss={loss.item():.4f} acc={acc_raw:.4f} ema_acc={acc_ema:.4f}")
        global_step += 1
```

### 3.4 验收标准

- [ ] 脚本能跑完至少 100 step
- [ ] 训练 loss 持续下降
- [ ] 评估 accuracy > 0.85（基线 ~0.92）
- [ ] 关键代码（forward / backward / clip / EMA）都是自己手写而非复制
- [ ] 能讲清 `evaluate()` 函数为什么用 `@torch.no_grad()` 装饰

---

## 四、面试卡片（Day 1 晚上 / 30 min）

### 卡片 1：训练大模型的核心难点有哪些？分别用什么方法解决？

**要点（4 个难点 → 4 类方案）**：

```text
1. 显存不足（参数 + 梯度 + 优化器 + 激活）
   ├── 优化器分片：ZeRO-1 / FSDP1
   ├── 优化器 + 梯度分片：ZeRO-2 / FSDP2
   ├── 优化器 + 梯度 + 参数分片：ZeRO-3 / FSDP2
   ├── 激活重计算：gradient checkpointing
   └── 混合精度：BF16 / FP16 + loss scaling

2. 计算太慢（单卡 FLOPS 不足）
   ├── 张量并行（TP）：切单层矩阵乘
   ├── 流水并行（PP）：切层数 + micro-batch
   └── 5D 混合并行：TP × PP × SP × CP × EP

3. 通信瓶颈（all-reduce / all-to-all 慢）
   ├── 重叠通信与计算：fused kernel、CUDA Graph
   ├── 压缩通信：ZeRO++（DeepSpeed）
   └── 高效拓扑感知：NCCL 选 Ring / Tree

4. 数据太大（单机装不下）
   ├── 数据并行（DDP / FSDP）
   ├── 流式数据 + IterableDataset
   └── 异步预取 + 数据重排
```

### 卡片 2：为什么需要分布式训练？单卡极限在哪里？

**要点**：

```text
单卡瓶颈：
- 显存：80GB H100 也装不下 70B 模型的 BF16 权重（140GB）
- 算力：单卡 FLOPS 即使是 H100（约 1000 TFLOPS BF16），
        训练 70B 模型在万亿 token 上仍需 1+ 月
- 通信：单卡 = 0 通信但 0 扩展

分布式收益：
- 显存线性扩展：8 卡 ≈ 8 倍显存
- 算力线性扩展：8 卡 ≈ 7~8 倍训练速度（考虑 10-15% 通信损耗）
- 模型可扩展到 100B+ 参数

代价：
- 通信开销：all-reduce / all-to-all
- 同步开销：慢卡拖整队
- 复杂度：多维并行编排、Checkpoint、容错
```

### 自检

- [ ] 卡片 1：能 5 分钟内背出"4 难点 → 4 方案"映射
- [ ] 卡片 2：能 3 分钟内讲清"显存 + 算力 + 通信"三个角度
- [ ] 能用 1 分钟说出"为什么 BF16 训练比 FP32 快"（硬件 Tensor Core 加速 + 显存减半）

---

## 五、复盘（Day 2 晚上 / 20 min）

```markdown
## Day 1-2 复盘

### 学到的概念（用 3 句话总结）
1. _____________________________________________
2. _____________________________________________
3. _____________________________________________

### 写了什么代码（贴 GitHub 链接）
- _____________________________________________

### 跑通了什么实验
- _____________________________________________

### 面试卡答题情况
- [ ] 卡片 1：_____ / 5 分
- [ ] 卡片 2：_____ / 5 分

### 哪些点还没理解（明天 / Phase 1 后半段要补）
- _____________________________________________

### 明天 (Day 03) 要做的
- _____________________________________________
```

---

## 六、推荐时间分配

| 时段 | 时长 | 任务 |
| --- | --- | --- |
| **Day 1 上午** | 1.5h | 理论：读 3 篇核心材料 + 整理概念对照表 |
| **Day 1 晚上** | 30 min | 面试卡 1 + 2 背诵 + 看 Karpathy 视频 |
| **Day 2 上午** | 2h | 实践：环境 + 数据 + 训练脚本 |
| **Day 2 晚上** | 20 min | 复盘 + 答面试卡 + 准备 Day 3 |

> **本阶段总投入**：约 4.5h
