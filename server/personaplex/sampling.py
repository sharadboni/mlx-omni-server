from __future__ import annotations
from typing import List
import mlx.core as mx


def sample_top_k(logits: mx.array, temperature: float, top_k: int) -> mx.array:
    """Sample from [B, vocab] logits with temperature and top-k filtering."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    scaled = logits / temperature

    if top_k > 0 and top_k < logits.shape[-1]:
        sorted_vals = mx.sort(scaled, axis=-1)
        threshold = sorted_vals[:, -top_k : -top_k + 1]
        scaled = mx.where(scaled >= threshold, scaled, mx.full(scaled.shape, -1e9))

    # Gumbel-max trick
    gumbel = -mx.log(-mx.log(mx.random.uniform(shape=scaled.shape)))
    return mx.argmax(scaled + gumbel, axis=-1)


def sample_top_k_with_penalty(
    logits: mx.array,
    temperature: float,
    top_k: int,
    past_tokens: List[int],
    penalty: float,
) -> mx.array:
    """Top-k sampling with repetition penalty."""
    if penalty <= 1.0 or not past_tokens:
        return sample_top_k(logits, temperature, top_k)

    unique_past = list(set(past_tokens))
    vocab_size = logits.shape[-1]
    penalty_mask = [1.0] * vocab_size
    for tok in unique_past:
        if 0 <= tok < vocab_size:
            penalty_mask[tok] = penalty
    p = mx.array(penalty_mask).reshape(1, vocab_size)
    penalized = mx.where(logits > 0, logits / p, logits * p)
    return sample_top_k(penalized, temperature, top_k)
