"""CPU teaching example: readable tool context is not a sampled policy action."""


def masked_mean(values: list[float], mask: list[int]) -> float:
    if len(values) != len(mask):
        raise ValueError("loss and mask lengths differ")
    if any(bit not in (0, 1) for bit in mask):
        raise ValueError("mask must be binary")
    count = sum(mask)
    if count == 0:
        raise ValueError("no valid action tokens")
    return sum(value for value, bit in zip(values, mask) if bit) / count


def main() -> None:
    tokens = ["Q0", "Q1", "A0", "A1", "T0", "T1", "F0", "F1", "PAD"]
    attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    full_action_mask = [0, 0, 1, 1, 0, 0, 1, 1, 0]
    prompt_length = 2
    response_mask = full_action_mask[prompt_length:]
    response_attention_mask = attention_mask[prompt_length:]
    assert len(response_mask) == len(tokens) - prompt_length == 7
    assert response_mask == [1, 1, 0, 0, 1, 1, 0]
    assert sum(response_mask) == 4
    assert attention_mask[4:6] == [1, 1]
    assert full_action_mask[4:6] == [0, 0]

    # Teaching scalar losses, not actual PPO losses or measured model outputs.
    losses = [1.0, 3.0, 100.0, 100.0, 2.0, 4.0, 999.0]
    correct = masked_mean(losses, response_mask)
    wrong = masked_mean(losses, response_attention_mask)
    assert correct == 2.5 and wrong == 35.0
    print(f"response_mask={response_mask}; valid actions={sum(response_mask)}")
    print(f"correct action mean={correct}; padding-only mean={wrong}")

    # A masked position does not contribute to this reduction.
    changed = losses.copy()
    changed[2] = 10000.0
    assert masked_mean(changed, response_mask) == correct
    print("changed tool-position loss; action mean unchanged")

    try:
        masked_mean(losses, [0] * len(losses))
    except ValueError as error:
        print(f"empty action mask: rejected ({error})")
    else:
        raise AssertionError("all-zero mask unexpectedly accepted")
    print("PASS: context/action distinction and valid-token reduction")


if __name__ == "__main__":
    main()
