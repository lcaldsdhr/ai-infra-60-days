"""
Day 9: 手撕 SDPA + Profiler 对比
对比 naive_sdpa vs F.scaled_dot_product_attention（Flash Attention）的速度
"""

import torch
import torch.nn.functional as F
import time


def naive_sdpa(Q, K, V):
    """朴素 SDPA：O(N²) 中间矩阵"""
    d = Q.shape[-1]
    scale = d ** -0.5
    attn = Q @ K.transpose(-2, -1) * scale
    attn = F.softmax(attn, dim=-1)
    O = attn @ V
    return O


def naive_sdpa_causal(Q, K, V):
    """带 causal mask 的 SDPA"""
    d = Q.shape[-1]
    scale = d ** -0.5
    attn = Q @ K.transpose(-2, -1) * scale
    N = Q.shape[-2]
    mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
    attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    O = attn @ V
    return O


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 验证一致性
    B, H, N, d = 2, 8, 64, 128
    Q = torch.randn(B, H, N, d, device=device)
    K = torch.randn(B, H, N, d, device=device)
    V = torch.randn(B, H, N, d, device=device)

    O_naive = naive_sdpa(Q, K, V)
    O_fa = F.scaled_dot_product_attention(Q, K, V)
    print(f"无 mask 一致性: {torch.allclose(O_naive, O_fa, atol=1e-5)}")

    O_causal = naive_sdpa_causal(Q, K, V)
    O_fa_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print(f"causal mask 一致性: {torch.allclose(O_causal, O_fa_causal, atol=1e-5)}")

    # 速度对比
    for N in [128, 512, 2048, 8192]:
        Q = torch.randn(1, 8, N, 128, device=device)
        K = torch.randn(1, 8, N, 128, device=device)
        V = torch.randn(1, 8, N, 128, device=device)

        # warmup
        for _ in range(5):
            naive_sdpa(Q, K, V)
            F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # 计时（10 次平均）
        t0 = time.time()
        for _ in range(10):
            naive_sdpa(Q, K, V)
        t_naive = (time.time() - t0) / 10

        t0 = time.time()
        for _ in range(10):
            F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        t_fa = (time.time() - t0) / 10

        print(f"N={N:5d} | naive={t_naive*1000:.2f}ms | flash={t_fa*1000:.2f}ms | "
              f"加速={t_naive/t_fa:.1f}x")
