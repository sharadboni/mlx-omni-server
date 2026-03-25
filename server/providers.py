from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
from typing import Any

import mlx.core as mx

from .config import DEFAULT_LLM_MODEL, DEFAULT_LLM_FAST_MODEL, DEFAULT_STT_MODEL, DEFAULT_TTS_MODEL, DEFAULT_TTS_CLONE_MODEL, DEFAULT_VLM_MODEL, DEFAULT_S2S_MODEL_REPO

log = logging.getLogger(__name__)

# Set by --keep-in-memory when starting the server.
keep_in_memory: bool = False
_cache: dict[str, Any] = {}

def _unload(key: str):
    _cache.pop(key, None)
    gc.collect()
    mx.clear_cache()


@contextmanager
def load_tts():
    """Load the fast TTS model (Kokoro) for preset voices."""
    if keep_in_memory and "tts" in _cache:
        yield _cache["tts"]
        return

    log.info(f"Loading TTS model: {DEFAULT_TTS_MODEL}")
    from mlx_audio.tts.utils import load_model
    model = load_model(DEFAULT_TTS_MODEL)

    if keep_in_memory:
        _cache["tts"] = model
        yield model
    else:
        try:
            yield model
        finally:
            del model
            _unload("tts")
            log.info("TTS model unloaded from memory")


@contextmanager
def load_tts_clone():
    """Load the voice cloning TTS model (Qwen3-TTS 1.7B)."""
    if keep_in_memory and "tts_clone" in _cache:
        yield _cache["tts_clone"]
        return

    log.info(f"Loading TTS clone model: {DEFAULT_TTS_CLONE_MODEL}")
    from mlx_audio.tts.utils import load_model
    model = load_model(DEFAULT_TTS_CLONE_MODEL)

    if keep_in_memory:
        _cache["tts_clone"] = model
        yield model
    else:
        try:
            yield model
        finally:
            del model
            _unload("tts_clone")
            log.info("TTS clone model unloaded from memory")


@contextmanager
def load_stt():
    if keep_in_memory and "stt" in _cache:
        yield _cache["stt"]
        return

    log.info(f"Loading STT model: {DEFAULT_STT_MODEL}")
    from mlx_audio.stt.utils import load_model
    model = load_model(DEFAULT_STT_MODEL)
    if keep_in_memory:
        _cache["stt"] = model
        yield model
    else:
        try:
            yield model
        finally:
            del model
            _unload("stt")
            log.info("STT model unloaded from memory")


@contextmanager
def load_s2s():
    """Load PersonaPlex S2S model and return (model, model_dir)."""
    if keep_in_memory and "s2s" in _cache:
        yield _cache["s2s"]
        return

    log.info(f"Loading S2S model: {DEFAULT_S2S_MODEL_REPO}")
    from .personaplex import PersonaPlexModel

    model, model_dir = PersonaPlexModel.from_pretrained(DEFAULT_S2S_MODEL_REPO)

    result = (model, model_dir)
    if keep_in_memory:
        _cache["s2s"] = result
        yield result
    else:
        try:
            yield result
        finally:
            del model, result
            _unload("s2s")
            log.info("S2S model unloaded from memory")


@contextmanager
def load_llm(model_id: str | None = None):
    actual_id = model_id or DEFAULT_LLM_MODEL
    cache_key = f"llm:{actual_id}"

    if keep_in_memory and cache_key in _cache:
        yield _cache[cache_key]
        return

    log.info(f"Loading LLM: {actual_id}")
    from mlx_lm import load
    try:
        model, tokenizer = load(actual_id, tokenizer_config={"trust_remote_code": True})
    except ValueError as e:
        if "does not exist or is not currently imported" not in str(e):
            raise
        log.warning(f"Unknown tokenizer class, falling back to PreTrainedTokenizerFast: {e}")
        from mlx_lm.utils import _download, load_model as _load_weights
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        from transformers import PreTrainedTokenizerFast
        model_path = _download(actual_id)
        model, _ = _load_weights(model_path)
        hf_tok = PreTrainedTokenizerFast.from_pretrained(str(model_path))
        tokenizer = TokenizerWrapper(hf_tok)
    result = (model, tokenizer)

    if keep_in_memory:
        _cache[cache_key] = result
        yield result
    else:
        try:
            yield result
        finally:
            del model, tokenizer, result
            _unload(cache_key)
            log.info(f"LLM {actual_id} unloaded from memory")


@contextmanager
def load_vlm():
    if keep_in_memory and "vlm" in _cache:
        yield _cache["vlm"]
        return

    log.info(f"Loading VLM model: {DEFAULT_VLM_MODEL}")
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    model, processor = load(DEFAULT_VLM_MODEL)
    config = load_config(DEFAULT_VLM_MODEL)
    result = (model, processor, config)
    if keep_in_memory:
        _cache["vlm"] = result
        yield result
    else:
        try:
            yield result
        finally:
            del model, processor, config, result
            _unload("vlm")
            log.info("VLM model unloaded from memory")
