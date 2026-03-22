from __future__ import annotations
from typing import Callable, List, Optional
import mlx.core as mx
import mlx.nn as nn

from .config import DepformerConfig
from .kv_cache import KVCache


class MultiLinear(nn.Module):
    """
    Stores weights for N steps as a single concatenated tensor and performs
    step-specific matmul. Supports 4-bit quantization in MLX QuantizedLinear format.
    """

    def __init__(
        self,
        num_steps: int,
        in_dim: int,
        out_dim: int,
        bits: int = 16,
        group_size: int = 64,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.out_dim = out_dim
        self.group_size = group_size
        self.bits = bits

        if bits < 16:
            packed_cols = in_dim // (32 // bits)
            num_groups = in_dim // group_size
            self.weight = mx.zeros([num_steps * out_dim, packed_cols], dtype=mx.uint32)
            self.scales = mx.zeros([num_steps * out_dim, num_groups], dtype=mx.float16)
            self.biases = mx.zeros([num_steps * out_dim, num_groups], dtype=mx.float16)
        else:
            self.weight = mx.zeros([num_steps * out_dim, in_dim])
            self.scales = None
            self.biases = None

    def __call__(self, x: mx.array, step: int) -> mx.array:
        s = step * self.out_dim
        e = s + self.out_dim
        w = self.weight[s:e]
        if self.scales is not None:
            ws = self.scales[s:e]
            wb = self.biases[s:e]
            return mx.quantized_matmul(
                x, w, scales=ws, biases=wb,
                transpose=True, group_size=self.group_size, bits=self.bits,
            )
        return x @ w.T


def _causal_mask(T: int, kv_len: int, dtype) -> mx.array:
    i = mx.arange(T).reshape(T, 1)
    j = mx.arange(kv_len).reshape(1, kv_len)
    valid = (j <= i + (kv_len - T)).astype(mx.float32)
    return (valid * 1e9 - 1e9).reshape(1, 1, T, kv_len).astype(dtype)


class DepformerAttention(nn.Module):
    def __init__(self, cfg: DepformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        total_dim = 3 * cfg.dim
        self.in_proj = MultiLinear(cfg.num_steps, cfg.dim, total_dim,
                                   bits=cfg.bits, group_size=cfg.group_size)
        self.out_proj = MultiLinear(cfg.num_steps, cfg.dim, cfg.dim,
                                    bits=cfg.bits, group_size=cfg.group_size)
        self.scale = cfg.head_dim ** -0.5

    def __call__(self, x: mx.array, step: int, cache: KVCache) -> mx.array:
        B, T, _ = x.shape
        H, D = self.cfg.num_heads, self.cfg.head_dim

        qkv = self.in_proj(x, step).reshape(B, T, 3, H, D)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        k, v = cache.update(k, v)

        kv_len = k.shape[2]
        if kv_len > self.cfg.context:
            k = k[:, :, kv_len - self.cfg.context:]
            v = v[:, :, kv_len - self.cfg.context:]

        actual_kv_len = k.shape[2]
        mask = _causal_mask(T, actual_kv_len, q.dtype) if T > 1 else None

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.cfg.dim)
        return self.out_proj(out, step)


class DepformerFFN(nn.Module):
    def __init__(self, cfg: DepformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.linear_in = MultiLinear(cfg.num_steps, cfg.dim, 2 * cfg.dim_feedforward,
                                      bits=cfg.bits, group_size=cfg.group_size)
        self.linear_out = MultiLinear(cfg.num_steps, cfg.dim_feedforward, cfg.dim,
                                       bits=cfg.bits, group_size=cfg.group_size)

    def __call__(self, x: mx.array, step: int) -> mx.array:
        doubled = self.linear_in(x, step)
        ffn_dim = self.cfg.dim_feedforward
        gate = doubled[..., :ffn_dim]
        value = doubled[..., ffn_dim:]
        return self.linear_out(nn.silu(gate) * value, step)


class DepformerLayer(nn.Module):
    def __init__(self, cfg: DepformerConfig) -> None:
        super().__init__()
        from .temporal import RMSNormF32
        self.norm1 = RMSNormF32(cfg.dim, eps=cfg.rms_norm_eps)
        self.norm2 = RMSNormF32(cfg.dim, eps=cfg.rms_norm_eps)
        self.self_attn = DepformerAttention(cfg)
        self.gating = DepformerFFN(cfg)

    def __call__(self, x: mx.array, step: int, cache: KVCache) -> mx.array:
        x = x + self.self_attn(self.norm1(x), step, cache)
        x = x + self.gating(self.norm2(x), step)
        return x


class Depformer(nn.Module):
    def __init__(self, cfg: DepformerConfig, temporal_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = [DepformerLayer(cfg) for _ in range(cfg.num_layers)]
        # Per-step input projections: temporal_dim → depformer_dim
        self.depformer_in = [
            nn.QuantizedLinear(temporal_dim, cfg.dim, bias=False,
                               group_size=cfg.group_size, bits=cfg.bits)
            for _ in range(cfg.num_steps)
        ]
        self.depformer_text_emb = nn.Embedding(cfg.text_card + 1, cfg.dim)
        self.depformer_emb = [
            nn.Embedding(cfg.card + 1, cfg.dim) for _ in range(cfg.num_steps - 1)
        ]
        self.linears = [nn.Linear(cfg.dim, cfg.card, bias=False) for _ in range(cfg.num_steps)]

    def generate(
        self,
        temporal_hidden: mx.array,
        text_token: mx.array,
        provided_tokens: Optional[List[int]],
        sample_fn: Callable[[mx.array, int], mx.array],
    ) -> mx.array:
        """
        Generate all codebook tokens for one timestep.
        temporal_hidden: [B, 1, temporal_dim]
        text_token: [B] (int32)
        provided_tokens: optional list of length num_steps; >=0 means use that token as conditioning
        Returns: [B, num_steps]
        """
        tokens: list[mx.array] = []
        prev_token = text_token  # [B]
        # Fresh KV caches for this temporal step (shared across depformer steps)
        caches = [KVCache() for _ in range(self.cfg.num_layers)]

        for k in range(self.cfg.num_steps):
            # Project temporal hidden to depformer dim
            x = self.depformer_in[k](temporal_hidden)  # [B, 1, depformer_dim]

            # Add previous token embedding
            prev_emb = prev_token[:, None]  # [B, 1]
            if k == 0:
                x = x + self.depformer_text_emb(prev_emb)
            else:
                x = x + self.depformer_emb[k - 1](prev_emb)

            # Pass through layers
            for layer, cache in zip(self.layers, caches):
                x = layer(x, k, cache)

            logits = self.linears[k](x).squeeze(1)  # [B, card]
            token = sample_fn(logits, k)  # [B]
            tokens.append(token)

            # Use provided token as conditioning if available
            if provided_tokens is not None and k < len(provided_tokens) and provided_tokens[k] >= 0:
                prev_token = mx.array([provided_tokens[k]], dtype=mx.int32)
            else:
                prev_token = token

        return mx.stack(tokens, axis=1)  # [B, num_steps]
