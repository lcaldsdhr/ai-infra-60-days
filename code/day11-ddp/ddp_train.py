"""
Day 11: DDP 数据并行训练骨架
用 DDP 训练 BERT-base（SST-2）——单卡代码加 4 行就变多卡

启动方式（2 卡）：
  torchrun --nproc_per_node=2 day11-ddp/ddp_train.py

单卡用途（验证）：
  python day11-ddp/ddp_train.py
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def train(rank=None, world_size=None):
    # ===== 1. 初始化分布式（单卡/多卡兼容）=====
    if rank is not None:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = rank
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 2. 数据 =====
    raw = load_dataset("nyu-mll/glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def collate(batch):
        texts = [b["sentence"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch])
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        return {**enc, "labels": labels}

    sampler = None
    shuffle = True
    if is_distributed():
        sampler = DistributedSampler(raw["train"], shuffle=True)
        shuffle = False  # sampler 自带 shuffle

    train_loader = DataLoader(
        raw["train"], batch_size=32, sampler=sampler, shuffle=shuffle, collate_fn=collate
    )

    # ===== 3. 模型（单卡 rank=0 才打印信息）=====
    if not is_distributed() or rank == 0:
        print(f"Device: {device}, World: {world_size or 1}")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[rank])

    # ===== 4. 优化器 =====
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 2
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # ===== 5. 训练循环 =====
    model.train()
    global_step = 0

    for epoch in range(2):
        if sampler is not None:
            sampler.set_epoch(epoch)  # 不同 epoch 不同 shuffle

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optim.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            if global_step % 100 == 0:
                print(f"[rank={dist.get_rank() if is_distributed() else 0}] "
                      f"step {global_step}: loss={loss.item():.4f}")

            global_step += 1

    # DDP 保存权重（只有 rank 0 保存即可）
    if is_distributed():
        if rank == 0:
            torch.save(model.module.state_dict(), "bert_sst2_ddp.pt")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), "bert_sst2_single.pt")

    print(f"{'DDP' if is_distributed() else 'Single'} 训练完成！")


if __name__ == "__main__":
    # torchrun 方式：自动注入环境变量
    if "WORLD_SIZE" in torch.cuda.get_device_properties.__code__.co_varnames:  # fallback
        pass

    # 检查是否用 torchrun 启动
    if dist.is_torchelastic_launched():
        train()
    else:
        # 单卡模式
        train()
