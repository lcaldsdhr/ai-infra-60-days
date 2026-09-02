"""Show how PPO clipping behaves for positive and negative advantages."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    ratio: float
    advantage: float


def ppo_clipped_objective(ratio: float, advantage: float, epsilon: float) -> tuple[float, float, float]:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1")
    clipped_ratio = min(max(ratio, 1.0 - epsilon), 1.0 + epsilon)
    raw = ratio * advantage
    clipped = clipped_ratio * advantage
    return raw, clipped, min(raw, clipped)


def main() -> None:
    epsilon = 0.2
    cases = (
        Case("positive / inside", ratio=1.1, advantage=1.0),
        Case("positive / too high", ratio=1.5, advantage=1.0),
        Case("negative / inside", ratio=0.9, advantage=-1.0),
        Case("negative / too low", ratio=0.5, advantage=-1.0),
    )

    print("== PPO Clipped Surrogate (epsilon=0.2) ==")
    print("case                     ratio      A      raw  clipped  objective")
    for case in cases:
        raw, clipped, objective = ppo_clipped_objective(
            case.ratio, case.advantage, epsilon
        )
        delta_logp = math.log(case.ratio)
        print(
            f"{case.name:<24} {case.ratio:>5.2f} "
            f"{case.advantage:>6.2f} {raw:>8.2f} {clipped:>8.2f} {objective:>10.2f}"
            f"  (delta_logp={delta_logp:+.3f})"
        )

    _, _, positive_limited = ppo_clipped_objective(1.5, 1.0, epsilon)
    _, _, negative_limited = ppo_clipped_objective(0.5, -1.0, epsilon)
    assert positive_limited == 1.2
    assert negative_limited == -0.8

    print("\nInterpretation:")
    print("- A > 0: raising probability past ratio 1.2 gives no extra objective gain.")
    print("- A < 0: lowering probability past ratio 0.8 is treated as -0.8, not -0.5.")
    print("- Clip limits update incentives; it does not guarantee a hard KL bound.")


if __name__ == "__main__":
    main()
