from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
from typing import Any

import mlx.core as mx

from .config import DEFAULT_STT_MODEL, DEFAULT_TTS_MODEL, DEFAULT_TTS_CLONE_MODEL, DEFAULT_VLM_MODEL

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
