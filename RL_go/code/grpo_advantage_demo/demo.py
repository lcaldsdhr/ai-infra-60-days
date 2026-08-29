"""A dependency-free GRPO outcome-advantage and policy-update toy experiment.

This mirrors the fixed verl implementation's teaching-relevant behavior:
- one outcome reward per completion;
- group by prompt id;
- use the sample standard deviation (torch.std default) when normalizing;
- broadcast each scalar advantage to valid response tokens;
- show one simplified policy-gradient ascent step.

It is intentionally not a trainer, model, or replacement for verl.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, stdev

EPSILON = 1e-6


@dataclass(frozen=True)
class GroupResult:
    rewards: list[float]
    mean: float
    sample_std: float
    advantages: list[float]


def grpo_advantages(rewards: list[float], normalize_by_std: bool = True) -> GroupResult:
    """Return GRPO outcome advantages for one prompt's rollout group.

    ``statistics.stdev`` uses the n - 1 denominator, matching PyTorch's
    default ``torch.std`` behavior used by this repository's fixed verl code.
    """
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two rollouts in each prompt group")

    mean = fmean(rewards)
    sample_std = stdev(rewards)
    denominator = sample_std + EPSILON if normalize_by_std else 1.0
    advantages = [(reward - mean) / denominator for reward in rewards]
    return GroupResult(rewards, mean, sample_std, advantages)


def broadcast_to_response_tokens(advantage: float, response_mask: list[int]) -> list[float]:
    """Match ``scores.unsqueeze(-1) * response_mask`` conceptually."""
    return [advantage if mask else 0.0 for mask in response_mask]


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_logits)
    return [value / total for value in exp_logits]


def one_policy_gradient_ascent_step(
    logits: list[float], advantages: list[float], learning_rate: float
) -> tuple[list[float], list[float]]:
    """Optimize mean(A_i * log pi_i) for a categorical toy policy.

    In real GRPO, this is token-level and PPO-clipped, may include KL, and is
    applied to model parameters. This compact form isolates update direction.
    """
    if len(logits) != len(advantages):
        raise ValueError("logits and advantages must have the same length")

    probabilities = softmax(logits)
    mean_advantage = sum(advantages) / len(advantages)
    gradients = [
        (advantage - probability * sum(advantages)) / len(advantages)
        for advantage, probability in zip(advantages, probabilities, strict=True)
    ]
    assert abs(mean_advantage) < 1e-9, "group-normalized advantages should sum to zero"
    updated_logits = [logit + learning_rate * grad for logit, grad in zip(logits, gradients, strict=True)]
    return probabilities, softmax(updated_logits)


def fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:+.6f}" for value in values) + "]"


def main() -> None:
    mixed = grpo_advantages([0.0, 1.0, 1.0, 0.0])
    no_signal = grpo_advantages([1.0, 1.0, 1.0, 1.0])

    print("== 1. One prompt, four rollouts ==")
    print(f"rewards:       {mixed.rewards}")
    print(f"group mean:    {mixed.mean:.6f}")
    print(f"sample std:    {mixed.sample_std:.6f}")
    print(f"advantages:    {fmt(mixed.advantages)}")
    print(f"token broadcast (trajectory 1, mask [1, 1, 1, 0]): {fmt(broadcast_to_response_tokens(mixed.advantages[0], [1, 1, 1, 0]))}")

    print("\n== 2. One simplified policy-gradient ascent step ==")
    before, after = one_policy_gradient_ascent_step([0.0, 0.0, 0.0, 0.0], mixed.advantages, learning_rate=0.5)
    print(f"probability before: {fmt(before)}")
    print(f"probability after:  {fmt(after)}")
    print("expected: trajectories 2/3 (positive advantage) increase; 1/4 decrease")

    print("\n== 3. A group without relative signal ==")
    print(f"rewards:       {no_signal.rewards}")
    print(f"advantages:    {fmt(no_signal.advantages)}")
    print("expected: all advantages are zero, so this group supplies no ranking gradient")

    assert after[1] > before[1] and after[2] > before[2]
    assert after[0] < before[0] and after[3] < before[3]
    assert all(abs(value) < 1e-9 for value in no_signal.advantages)


if __name__ == "__main__":
    main()
