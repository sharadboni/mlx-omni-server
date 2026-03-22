from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

from .config import TemporalConfig
from .kv_cache import KVCache


class RMSNormF32(nn.Module):
    """RMSNorm computed in float32 for numerical stability."""

    def __init__(self, dims: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = mx.ones([dims])
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        ms = mx.mean(x32 * x32, axis=-1, keepdims=True)
        normed = x32 * mx.rsqrt(ms + self.eps)
        return (normed * self.weight).astype(x.dtype)


def _causal_mask(T: int, kv_len: int, dtype) -> mx.array:
    """Additive causal mask: 0 for valid positions, -1e9 for masked."""
    i = mx.arange(T).reshape(T, 1)
    j = mx.arange(kv_len).reshape(1, kv_len)
    valid = (j <= i + (kv_len - T)).astype(mx.float32)
    mask = valid * 1e9 - 1e9
    return mask.reshape(1, 1, T, kv_len).astype(dtype)


class TemporalAttention(nn.Module):
    def __init__(self, cfg: TemporalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        total_dim = 3 * cfg.dim
        self.in_proj = nn.QuantizedLinear(cfg.dim, total_dim, bias=False,
                                          group_size=cfg.group_size, bits=cfg.bits)
        self.out_proj = nn.QuantizedLinear(cfg.dim, cfg.dim, bias=False,
                                           group_size=cfg.group_size, bits=cfg.bits)
        self.rope = nn.RoPE(cfg.head_dim, traditional=True, base=float(cfg.max_period))
        self.scale = cfg.head_dim ** -0.5

    def __call__(self, x: mx.array, cache: KVCache, offset: int) -> mx.array:
        B, T, _ = x.shape
        H, D = self.cfg.num_heads, self.cfg.head_dim

        qkv = self.in_proj(x).reshape(B, T, 3, H, D)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)  # [B, H, T, D]
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        k, v = cache.update(k, v)

        # Context window limit
        kv_len = k.shape[2]
        target_len = T + min(self.cfg.context, kv_len - T)
        if target_len < kv_len:
            k = k[:, :, kv_len - target_len:]
            v = v[:, :, kv_len - target_len:]

        actual_kv_len = k.shape[2]
        mask = _causal_mask(T, actual_kv_len, q.dtype) if T > 1 else None

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.cfg.dim)
        return self.out_proj(out)


class TemporalFFN(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, cfg: TemporalConfig) -> None:
        super().__init__()
        self.ffn_dim = cfg.intermediate_size
        self.linear_in = nn.QuantizedLinear(cfg.dim, 2 * self.ffn_dim, bias=False,
                                             group_size=cfg.group_size, bits=cfg.bits)
        self.linear_out = nn.QuantizedLinear(self.ffn_dim, cfg.dim, bias=False,
                                              group_size=cfg.group_size, bits=cfg.bits)

    def __call__(self, x: mx.array) -> mx.array:
        doubled = self.linear_in(x)
        gate = doubled[..., :self.ffn_dim]
        value = doubled[..., self.ffn_dim:]
        return self.linear_out(nn.silu(gate) * value)


class TemporalTransformerLayer(nn.Module):
    def __init__(self, cfg: TemporalConfig) -> None:
        super().__init__()
        self.norm1 = RMSNormF32(cfg.dim, eps=cfg.rms_norm_eps)
        self.norm2 = RMSNormF32(cfg.dim, eps=cfg.rms_norm_eps)
        self.self_attn = TemporalAttention(cfg)
        self.gating = TemporalFFN(cfg)

    def __call__(self, x: mx.array, cache: KVCache, offset: int) -> mx.array:
        x = x + self.self_attn(self.norm1(x), cache, offset)
        x = x + self.gating(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, cfg: TemporalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = [TemporalTransformerLayer(cfg) for _ in range(cfg.num_layers)]
        self.out_norm = RMSNormF32(cfg.dim, eps=cfg.rms_norm_eps)
        # text_emb: vocab+1 (padding token)
        self.text_emb = nn.Embedding(cfg.text_card + 1, cfg.dim)
        # 16 audio embeddings: card+1 (initial token)
        self.emb = [nn.Embedding(cfg.card + 1, cfg.dim) for _ in range(cfg.num_audio_embeddings)]
        self.text_linear = nn.Linear(cfg.dim, cfg.text_card, bias=False)
        self._kv_caches: list[KVCache] = [KVCache() for _ in range(cfg.num_layers)]

    def reset_cache(self) -> None:
        for c in self._kv_caches:
            c.reset()

    def _embed(self, text_tokens: mx.array, audio_tokens: mx.array) -> mx.array:
        """Build combined embedding from text + audio token IDs."""
        B, T = text_tokens.shape
        hidden = self.text_emb(text_tokens)  # [B, T, dim]
        for i in range(self.cfg.num_audio_embeddings):
            raw = audio_tokens[:, i, :]  # [B, T]
            valid = (raw >= 0)[:, :, None]
            safe = mx.maximum(raw, mx.zeros_like(raw))
            emb = self.emb[i](safe)
            hidden = hidden + mx.where(valid, emb, mx.zeros_like(emb))
        return hidden

    def forward(
        self,
        text_tokens: mx.array,
        audio_tokens: mx.array,
        offset: int,
    ) -> tuple[mx.array, mx.array]:
        """
        text_tokens:  [B, T] int32
        audio_tokens: [B, 16, T] int32  (-1 = invalid/no token)
        Returns: (normed [B, T, dim], text_logits [B, T, text_card])
        """
        hidden = self._embed(text_tokens, audio_tokens)
        for layer, cache in zip(self.layers, self._kv_caches):
            hidden = layer(hidden, cache, offset)
        normed = self.out_norm(hidden)
        return normed, self.text_linear(normed)

    def forward_batch_embedding(self, embeddings: mx.array, offset: int) -> None:
        """Feed pre-computed embeddings through the transformer (voice prompt prefill)."""
        hidden = embeddings  # [1, T, dim]
        for layer, cache in zip(self.layers, self._kv_caches):
            hidden = layer(hidden, cache, offset)
        mx.eval(hidden)
