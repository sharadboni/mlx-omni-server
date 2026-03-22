from __future__ import annotations
import mlx.core as mx


class KVCache:
    """Simple concatenation-based KV cache."""

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def offset(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        return self.keys, self.values

    def reset(self) -> None:
        self.keys = None
        self.values = None
