"""
Day 2: BERT-base SFT 训练脚本
目标：手写完整训练循环（含 EMA + Eval 钩子），跑通 SST-2 二分类

验收标准：
- 跑通 100 step
- loss 持续下降
- eval accuracy > 0.85（基线 ~0.92）
"""

import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset


# ============================================================
# 第 1 步：加载数据（15 min）
# ============================================================

raw = load_dataset("glue", "sst2")

print(raw)
# 预期输出：
# DatasetDict({
#     train: Dataset({ features: ['sentence', 'label', 'idx'], num_rows: 67349 })
#     validation: Dataset({ features: ['sentence', 'label', 'idx'], num_rows: 872 })
#     test: Dataset({ features: ['sentence', 'label', 'idx'], num_rows: 1821 })
# })


# ============================================================
# 第 2 步：加载 tokenizer + 模型（15 min）
# ============================================================

# 加载 bert-base-uncased 的 tokenizer 和模型
# 模型用 AutoModelForSequenceClassification，num_labels=2
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)


# ============================================================
# 第 3 步：写 collate 函数（15 min）
# ============================================================

def collate(batch):
    """
    输入：list of dict，每个 dict 有 'sentence' 和 'label'
    输出：dict，含 input_ids, attention_mask, token_type_ids, labels
    """
    texts = [b["sentence"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])

    # enc 需要：padding=True, truncation=True, max_length=128, return_tensors="pt"
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    return {**enc, "labels": labels}


# ============================================================
# 第 4 步：创建 DataLoader（10 min）
# ============================================================

train_loader = DataLoader(
    raw["train"], batch_size=32, shuffle=True, collate_fn=collate
)
eval_loader = DataLoader(
    raw["validation"], batch_size=64, collate_fn=collate
)


# ============================================================
# 第 5 步：设备 + EMA（10 min）
# ============================================================

# 选设备（cuda 或 cpu）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# EMA：指数移动平均，让推理用"更平滑"的权重
# 原理：ema_weight = decay × ema_weight + (1-decay) × model_weight
# 每个 step 都更新 EMA 影子权重，但只在 eval 时用
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.999))


# ============================================================
# 第 6 步：优化器 + 调度器（15 min）
# ============================================================

# AdamW，lr=2e-5，weight_decay=0.01
optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# 训练 2 个 epoch
num_epochs = 2
total_steps = len(train_loader) * num_epochs

# linear schedule with warmup，warmup 占 10%
sched = get_linear_schedule_with_warmup(
    optim,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)


# ============================================================
# 第 7 步：写 Eval 函数（15 min）★ 必须手写
# ============================================================

@torch.no_grad()
def evaluate(model_to_eval):
    """
    eval 函数：计算验证集 accuracy
    @torch.no_grad() 的 3 个原因：
    1. 不构建计算图 → 省显存（不用存中间激活）
    2. 不需要梯度 → 省计算（跳过反向图构建）
    3. 防止误改模型 → Dropout/BN 在 eval() 下已有不同行为
    """
    model_to_eval.eval()
    correct, total = 0, 0

    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model_to_eval(**batch)
        pred = out.logits.argmax(dim=-1)
        correct += (pred == batch["labels"]).sum().item()
        total += len(batch["labels"])

    model_to_eval.train()
    return correct / total


# ============================================================
# 第 8 步：训练循环（30 min）★ 必须手写
# ============================================================

model.train()
global_step = 0

for epoch in range(num_epochs):
    for batch in train_loader:
        # --- 8.1 数据搬到 device ---
        batch = {k: v.to(device) for k, v in batch.items()}

        # --- 8.2 清零梯度（必须在 backward 之前）---
        optim.zero_grad()

        # --- 8.3 前向 ---
        out = model(**batch)
        loss = out.loss

        # --- 8.4 反向 ---
        loss.backward()

        # --- 8.5 梯度裁剪 ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # --- 8.6 优化器步进 ---
        optim.step()

        # --- 8.7 调度器步进 ---
        sched.step()

        # --- 8.8 EMA 更新 ---
        ema_model.update_parameters(model)

        # --- 8.9 日志 + Eval ---
        if global_step % 100 == 0:
            acc_raw = evaluate(model)
            acc_ema = evaluate(ema_model)
            print(f"step {global_step:5d} | "
                  f"loss={loss.item():.4f} | "
                  f"acc={acc_raw:.4f} | "
                  f"ema_acc={acc_ema:.4f}")

        global_step += 1

print("训练完成！")
