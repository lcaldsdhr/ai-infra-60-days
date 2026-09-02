"""Estimate decoder KV-cache capacity with only the Python standard library."""

from __future__ import annotations

import argparse


def kv_bytes_per_token(
    *, layers: int, kv_heads: int, head_dim: int, dtype_bytes: int
) -> int:
    """Return KV bytes for one cached token across all transformer layers."""
    values = (layers, kv_heads, head_dim, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError("layers, kv_heads, head_dim and dtype_bytes must be positive")
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.3f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument(
        "--kv-budget-gib",
        type=float,
        default=40.0,
        help="Memory budget reserved for KV cache, not total GPU memory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.context <= 0 or args.concurrency <= 0 or args.kv_budget_gib <= 0:
        raise ValueError("context, concurrency and kv-budget-gib must be positive")

    per_token = kv_bytes_per_token(
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype_bytes=args.dtype_bytes,
    )
    per_sequence = per_token * args.context
    total = per_sequence * args.concurrency
    budget = int(args.kv_budget_gib * 1024**3)
    max_full_sequences = budget // per_sequence

    print("== KV Cache Capacity Estimate ==")
    print(
        "formula: 2(K,V) x layers x kv_heads x head_dim x dtype_bytes"
    )
    print(f"per cached token:       {format_bytes(per_token)}")
    print(f"per {args.context}-token sequence: {format_bytes(per_sequence)}")
    print(f"{args.concurrency} full sequences:    {format_bytes(total)}")
    print(
        f"max full sequences in {args.kv_budget_gib:.1f} GiB KV budget: "
        f"{max_full_sequences}"
    )
    print("\nNotes:")
    print("- This is decoder KV only; weights, activations and runtime workspaces are excluded.")
    print("- Real requests have different lengths, so paged allocation changes utilization.")
    print("- Reducing kv_heads (GQA/MQA), context, concurrency or dtype bytes lowers KV use.")

    # Default configuration has a known, easy-to-audit result.
    if (
        args.layers,
        args.kv_heads,
        args.head_dim,
        args.dtype_bytes,
        args.context,
    ) == (28, 4, 128, 2, 8192):
        assert per_token == 57_344
        assert per_sequence == 469_762_048


if __name__ == "__main__":
    main()
