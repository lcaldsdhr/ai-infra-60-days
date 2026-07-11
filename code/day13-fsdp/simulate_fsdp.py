"""
Day 13: FSDP2 包装演示——用 PyTorch 原生 API 模拟 FSDP 包装
（纯演示代码，真正多卡需 torchrun 启动）
"""

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    """4 层小 Transformer，用于演示 FSDP 包装"""
    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1000)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


def simulate_fsdp_sharding(model, world_size=8):
    """
    FSDP 的核心逻辑（纯模拟，不依赖 torch.distributed）
    演示每层参数如何分片到 N 卡
    """
    print(f"{'Layer':30s} {'Total params':15s} {'Per-rank params':15s} {'Saved%':>8s}")
    print("-" * 70)

    for name, param in model.named_parameters():
        total = param.numel()
        per_rank = (total + world_size - 1) // world_size   # 向上取整
        saved = (1 - per_rank / total) * 100
        print(f"{name:30s} {total:>10d} {per_rank:>12d} {saved:>7.1f}%")

    # 总参数量
    total_all = sum(p.numel() for p in model.parameters())
    per_rank_all = sum((p.numel() + world_size - 1) // world_size for p in model.parameters())
    print("-" * 70)
    print(f"{'总计':30s} {total_all:>10d} {per_rank_all:>12d} "
          f"{(1 - per_rank_all / total_all) * 100:>7.1f}%")


if __name__ == "__main__":
    model = TinyTransformer()
    print(f"模型总参数量（7 层）: {sum(p.numel() for p in model.parameters()):,}")
    print(f"BF16 权重 = {sum(p.numel() for p in model.parameters()) * 2 / 1e6:.1f} MB")
    print()

    simulate_fsdp_sharding(model, world_size=8)
