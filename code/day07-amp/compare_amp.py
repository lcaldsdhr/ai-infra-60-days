"""
Day 7: AMP 混合精度训练
改动：把 Day 2 的 BF16 训练改成 AMP 对比实验

跑 3 轮（各 200 step），对比 loss 曲线和显存：
  ① FP32 基线
  ② AMP+FP16（GradScaler）
  ③ AMP+BF16（无 scaler）
"""

import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ===== 数据（同 Day 2）=====
raw = load_dataset("nyu-mll/glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def collate(batch):
    texts = [b["sentence"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return {**enc, "labels": labels}

train_loader = DataLoader(raw["train"], batch_size=32, shuffle=True, collate_fn=collate)
eval_loader = DataLoader(raw["validation"], batch_size=64, collate_fn=collate)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ===== 对比实验 =====
# FP16 需要 GradScaler，BF16 不需要
# FP32 不需要 autocast，也不需要 scaler

def train_one_config(mode, use_scaler=False):
    print(f"\n{'='*50}")
    print(f"Mode: {mode}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.999))
    
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 2
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    
    scaler = GradScaler() if use_scaler else None
    max_steps = 200
    
    model.train()
    global_step = 0
    
    for epoch in range(2):
        for batch in train_loader:
            if global_step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optim.zero_grad()
            
            # ─── AMP 的核心差异在这里 ───
            if mode == "fp32":
                # FP32：正常计算，无 autocast
                out = model(**batch)
                loss = out.loss
            elif mode == "amp_fp16":
                # FP16：autocast 自动选 FP16 算子 + GradScaler
                with autocast(dtype=torch.float16):
                    out = model(**batch)
                    loss = out.loss
            elif mode == "amp_bf16":
                # BF16：autocast 自动选 BF16 算子，不需要 scaler
                with autocast(dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss
            # ────────────────────────────
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            
            sched.step()
            ema_model.update_parameters(model)
            
            if global_step % 50 == 0:
                print(f"  step {global_step:4d} | loss={loss.item():.4f}")
            
            global_step += 1
    
    print(f"{mode} done ({max_steps} steps)")
    return model


if __name__ == "__main__":
    for mode, use_scaler in [("fp32", False), ("amp_fp16", True), ("amp_bf16", False)]:
        train_one_config(mode, use_scaler)
        torch.cuda.empty_cache()
