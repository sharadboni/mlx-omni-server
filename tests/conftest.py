"""Shared fixtures and helpers for all tests."""
from __future__ import annotations

import io
import struct
import wave
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Minimal WAV helpers
# ---------------------------------------------------------------------------

def make_wav_bytes(duration_s: float = 0.1, sample_rate: int = 24000) -> bytes:
    """Return a valid mono PCM WAV as bytes."""
    n_samples = int(sample_rate * duration_s)
    samples = (np.zeros(n_samples, dtype=np.int16)).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples)
    return buf.getvalue()


def make_wav_b64(duration_s: float = 0.1) -> str:
    import base64
    return base64.b64encode(make_wav_bytes(duration_s)).decode()


# ---------------------------------------------------------------------------
# Minimal 1×1 PNG helper
# ---------------------------------------------------------------------------

def make_png_bytes() -> bytes:
    """Return a minimal valid 1×1 white PNG."""
    import zlib
    def _chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_row = b"\x00\xff\xff\xff"  # filter byte + RGB white
    idat = _chunk(b"IDAT", zlib.compress(raw_row))
    iend = _chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def make_png_b64() -> str:
    import base64
    return base64.b64encode(make_png_bytes()).decode()


# ---------------------------------------------------------------------------
# Mock context-manager factories
# ---------------------------------------------------------------------------

def mock_llm_cm(response_text: str = "Hello, world!"):
    """Context manager yielding a fake (model, tokenizer) pair."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<prompt>"
    tokenizer.encode.return_value = [1, 2, 3]

    @contextmanager
    def _cm(model_id=None):
        yield model, tokenizer

    return _cm, model, tokenizer


def mock_tts_cm(audio: np.ndarray | None = None, sample_rate: int = 24000):
    """Context manager yielding a fake TTS model."""
    if audio is None:
        audio = np.zeros(2400, dtype=np.float32)

    result = MagicMock()
    result.audio = audio
    result.sample_rate = sample_rate

    tts_model = MagicMock()
    tts_model.generate.side_effect = lambda **kwargs: iter([result])

    @contextmanager
    def _cm():
        yield tts_model

    return _cm, tts_model


def mock_stt_cm(text: str = "hello world"):
    """Context manager yielding a fake STT model."""
    stt_model = MagicMock()
    stt_model.generate.return_value = {"text": text}

    @contextmanager
    def _cm():
        yield stt_model

    return _cm, stt_model


def mock_vlm_cm(response_text: str = "A white image."):
    """Context manager yielding a fake (model, processor, config) triple."""
    model = MagicMock()
    processor = MagicMock()
    config = MagicMock()
    output = MagicMock()
    output.text = response_text

    @contextmanager
    def _cm():
        yield model, processor, config

    return _cm, model, processor, config, output
