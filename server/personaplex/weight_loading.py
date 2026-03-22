from __future__ import annotations
import re
from pathlib import Path
from typing import Dict
import mlx.core as mx


# ---------------------------------------------------------------------------
# Temporal weight sanitization
# ---------------------------------------------------------------------------

def sanitize_temporal(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """
    - *.alpha (1,1,D) → *.weight (D)   (RMSNorm)
    - *.in_proj_weight → *.in_proj.weight  (+ _scales/_biases)
    """
    out: Dict[str, mx.array] = {}
    for key, val in weights.items():
        new_key = key
        new_val = val

        if key.endswith(".alpha"):
            new_key = key[:-6] + ".weight"
            if new_val.ndim == 3:
                new_val = new_val.squeeze(0).squeeze(0)

        for suffix in ("_weight", "_scales", "_biases"):
            needle = ".in_proj" + suffix
            if key.endswith(needle):
                dot_suffix = "." + suffix[1:]
                new_key = key[: -len(needle)] + ".in_proj" + dot_suffix
                break

        out[new_key] = new_val
    return out


# ---------------------------------------------------------------------------
# Depformer weight sanitization + MultiLinear packing
# ---------------------------------------------------------------------------

_GATING_PAT = re.compile(
    r"^(layers\.\d+\.gating)\.(\d+)\.(linear_in|linear_out)\.(weight|scales|biases)$"
)


def sanitize_depformer(weights: Dict[str, mx.array], num_steps: int = 16) -> Dict[str, mx.array]:
    """
    - *.alpha → *.weight
    - *.in_proj_weight → *.in_proj.weight  (+ scales/biases)
    - *.out_proj_weight → *.out_proj.weight
    - layers.L.gating.{step}.linear_{in/out}.{weight/scales/biases}
        → concatenated layers.L.gating.linear_{in/out}.{weight/scales/biases}
    """
    out: Dict[str, mx.array] = {}
    per_step: Dict[str, list[tuple[int, mx.array]]] = {}

    for key, val in weights.items():
        new_key = key
        new_val = val

        # RMSNorm alpha
        if key.endswith(".alpha"):
            new_key = key[:-6] + ".weight"
            if new_val.ndim == 3:
                new_val = new_val.squeeze(0).squeeze(0)
            out[new_key] = new_val
            continue

        # Attention in_proj/out_proj
        matched = False
        for proj in ("in_proj", "out_proj"):
            for suffix in ("_weight", "_scales", "_biases"):
                needle = f".{proj}{suffix}"
                if key.endswith(needle):
                    dot_suffix = "." + suffix[1:]
                    new_key = key[: -len(needle)] + f".{proj}" + dot_suffix
                    out[new_key] = new_val
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue

        # Per-step FFN gating
        m = _GATING_PAT.match(key)
        if m:
            layer_prefix, step_str, linear_name, tensor_type = m.groups()
            step = int(step_str)
            packed_key = f"{layer_prefix}.{linear_name}.{tensor_type}"
            per_step.setdefault(packed_key, []).append((step, val))
            continue

        out[new_key] = new_val

    # Pack per-step tensors by concatenating along axis 0 in step order
    for packed_key, step_vals in per_step.items():
        step_vals.sort(key=lambda x: x[0])
        out[packed_key] = mx.concatenate([v for _, v in step_vals], axis=0)

    return out


# ---------------------------------------------------------------------------
# Embedding key splitting
# ---------------------------------------------------------------------------

_TEMPORAL_EMB_PREFIXES = ("text_emb.", "emb.", "text_linear.")
_DEPFORMER_EMB_PREFIXES = ("depformer_emb.", "depformer_text_emb.", "linears.")


def split_embedding_weights(
    weights: Dict[str, mx.array],
) -> tuple[Dict[str, mx.array], Dict[str, mx.array]]:
    temporal: Dict[str, mx.array] = {}
    depformer: Dict[str, mx.array] = {}
    for key, val in weights.items():
        if any(key.startswith(p) for p in _TEMPORAL_EMB_PREFIXES):
            temporal[key] = val
        elif any(key.startswith(p) for p in _DEPFORMER_EMB_PREFIXES):
            depformer[key] = val
    return temporal, depformer


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_weights(model, directory: str | Path) -> None:
    """Load PersonaPlex weights from the aufklarer split-format directory."""
    d = Path(directory)
    all_weights: list[tuple[str, mx.array]] = []

    # --- Temporal transformer ---
    temporal_file = d / "temporal.safetensors"
    if temporal_file.exists():
        raw = dict(mx.load(str(temporal_file)))
        for k, v in sanitize_temporal(raw).items():
            all_weights.append((f"temporal.{k}", v))

    # --- Embeddings (split between temporal + depformer) ---
    emb_file = d / "embeddings.safetensors"
    if emb_file.exists():
        raw = dict(mx.load(str(emb_file)))
        t_emb, d_emb = split_embedding_weights(raw)
        for k, v in t_emb.items():
            all_weights.append((f"temporal.{k}", v))
        for k, v in d_emb.items():
            all_weights.append((f"depformer.{k}", v))

    # --- Depformer ---
    dep_file = d / "depformer.safetensors"
    if dep_file.exists():
        raw = dict(mx.load(str(dep_file)))
        for k, v in sanitize_depformer(raw, model.depformer.cfg.num_steps).items():
            all_weights.append((f"depformer.{k}", v))

    model.load_weights(all_weights)
    mx.eval(model)


def load_voice(directory: str | Path, voice: str) -> tuple[mx.array, mx.array | None]:
    """Load voice embeddings and optional cache from voices/{voice}.safetensors."""
    path = Path(directory) / "voices" / f"{voice}.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Voice file not found: {path}")
    data = dict(mx.load(str(path)))
    embeddings = data["embeddings"]  # [T, 1, 1, dim]
    cache = data.get("cache")        # [1, 17, CT] or None
    return embeddings, cache
