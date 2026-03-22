from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# Constant tokens used during prompting phases
SINE_TOKENS: List[int] = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
SILENCE_TOKENS: List[int] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]

# Default system prompt: "<system> You are a helpful assistant. Answer questions clearly and concisely. <system>"
DEFAULT_SYSTEM_PROMPT_TOKENS: List[int] = [
    607, 4831, 578, 493, 298, 272, 3850, 5019, 263,
    506, 1292, 2366, 267, 22876, 362, 263, 607, 4831, 578,
]


@dataclass
class TemporalConfig:
    dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    hidden_scale: float = 4.125
    nQ: int = 8           # audio codebooks per side
    card: int = 2048      # audio vocab size
    text_card: int = 32000
    context: int = 3000
    max_period: int = 10000
    rms_norm_eps: float = 1e-8
    group_size: int = 64
    bits: int = 4

    @property
    def intermediate_size(self) -> int:
        return int(self.dim * 2 / 3 * self.hidden_scale)  # 11264

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads  # 128

    @property
    def num_audio_embeddings(self) -> int:
        return 2 * self.nQ  # 16

    @property
    def text_padding_id(self) -> int:
        return 3


@dataclass
class DepformerConfig:
    dim: int = 1024
    num_layers: int = 6
    num_heads: int = 16
    dim_feedforward: int = 2816
    num_steps: int = 16
    card: int = 2048
    text_card: int = 32000
    context: int = 8
    rms_norm_eps: float = 1e-8
    group_size: int = 64
    bits: int = 4

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads  # 64


@dataclass
class PersonaPlexConfig:
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    depformer: DepformerConfig = field(default_factory=DepformerConfig)
    delays: List[int] = field(
        default_factory=lambda: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    )
    sample_rate: int = 24000

    @property
    def num_streams(self) -> int:
        return 1 + self.temporal.nQ * 2  # 17

    @property
    def max_delay(self) -> int:
        return max(self.delays)  # 1
