"""A dependency-free, single-head attention demo for KV-cache reuse."""

from __future__ import annotations

import math

Vector = list[float]
Matrix = tuple[tuple[float, ...], ...]

W_Q: Matrix = ((0.7, 0.1, 0.2), (0.0, 0.8, 0.2), (0.2, 0.1, 0.7))
W_K: Matrix = ((0.6, 0.2, 0.1), (0.1, 0.7, 0.2), (0.3, 0.0, 0.8))
W_V: Matrix = ((0.8, 0.1, 0.0), (0.2, 0.6, 0.2), (0.0, 0.3, 0.7))


def embedding(token: int) -> Vector:
    """Deterministic toy embedding; real models read a learned embedding table."""
    return [token / 10.0, (token % 3) / 3.0, 1.0]


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


def dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def softmax(values: Vector) -> Vector:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def attention(query: Vector, keys: list[Vector], values: list[Vector]) -> Vector:
    scale = math.sqrt(len(query))
    weights = softmax([dot(query, key) / scale for key in keys])
    return [
        sum(weight * value[index] for weight, value in zip(weights, values))
        for index in range(len(query))
    ]


def full_recompute(prefix: list[int]) -> tuple[Vector, int]:
    """Re-project K/V for every historical token, then query the last token."""
    embeddings = [embedding(token) for token in prefix]
    keys = [matvec(W_K, item) for item in embeddings]
    values = [matvec(W_V, item) for item in embeddings]
    query = matvec(W_Q, embeddings[-1])
    return attention(query, keys, values), 2 * len(prefix)


def cached_decode(prompt: list[int], generated: list[int]) -> tuple[list[Vector], int]:
    """Project prompt K/V once, then append only each new token's K/V."""
    keys = [matvec(W_K, embedding(token)) for token in prompt]
    values = [matvec(W_V, embedding(token)) for token in prompt]
    projection_count = 2 * len(prompt)
    outputs: list[Vector] = []

    for token in generated:
        item = embedding(token)
        keys.append(matvec(W_K, item))
        values.append(matvec(W_V, item))
        projection_count += 2
        query = matvec(W_Q, item)
        outputs.append(attention(query, keys, values))
    return outputs, projection_count


def main() -> None:
    prompt = [1, 2, 3]
    generated = [4, 5, 6]

    full_outputs: list[Vector] = []
    full_projection_count = 0
    prefix = prompt.copy()
    for token in generated:
        prefix.append(token)
        output, projections = full_recompute(prefix)
        full_outputs.append(output)
        full_projection_count += projections

    cached_outputs, cached_projection_count = cached_decode(prompt, generated)

    print("== Toy KV Cache Attention ==")
    for step, (full, cached) in enumerate(zip(full_outputs, cached_outputs), start=1):
        max_error = max(abs(a - b) for a, b in zip(full, cached))
        print(
            f"step {step}: full={['%.6f' % x for x in full]} "
            f"cached={['%.6f' % x for x in cached]} max_error={max_error:.3e}"
        )
        assert max_error < 1e-12

    print(f"\nK/V projections without cache: {full_projection_count}")
    print(f"K/V projections with cache:    {cached_projection_count}")
    print(f"saved projections:              {full_projection_count - cached_projection_count}")
    assert (full_projection_count, cached_projection_count) == (30, 12)

    print("\nConclusion: outputs match; the cache removes repeated historical K/V projections.")
    print("Attention still reads historical keys/values, so long context is not free.")


if __name__ == "__main__":
    main()
