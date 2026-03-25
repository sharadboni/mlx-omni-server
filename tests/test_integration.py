"""Integration tests — load real models, make real requests, clean up after.

Run with:  uv run pytest -m integration
Skip with: uv run pytest -m "not integration"  (default)
"""
from __future__ import annotations

import gc
import io
import wave

import mlx.core as mx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.providers as providers
from server.app import app
from server.config import (
    DEFAULT_LLM_FAST_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VLM_MODEL,
)
from tests.conftest import make_png_b64, make_wav_b64


@pytest.fixture(autouse=True)
def cleanup_models():
    """Clear provider cache and free MLX memory after every integration test."""
    yield
    providers._cache.clear()
    gc.collect()
    mx.clear_cache()


@pytest.fixture
def http_client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Provider-level: verify models load and tokenizers work
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_load_fast_llm():
    """Fast model has a TokenizersBackend tokenizer_class — fallback must succeed."""
    with providers.load_llm(DEFAULT_LLM_FAST_MODEL) as (model, tokenizer):
        assert model is not None
        assert tokenizer is not None
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0


@pytest.mark.integration
def test_load_default_llm():
    with providers.load_llm(DEFAULT_LLM_MODEL) as (model, tokenizer):
        assert model is not None
        assert tokenizer is not None


# ---------------------------------------------------------------------------
# HTTP-level: full round-trip through the chat endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_chat_completions_fast_model(http_client):
    resp = http_client.post(
        "/v1/chat/completions",
        json={
            "model": DEFAULT_LLM_FAST_MODEL,
            "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],
            "max_tokens": 8,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"]


@pytest.mark.integration
def test_chat_completions_streaming_fast_model(http_client):
    resp = http_client.post(
        "/v1/chat/completions",
        json={
            "model": DEFAULT_LLM_FAST_MODEL,
            "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],
            "max_tokens": 8,
            "stream": True,
        },
    )
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    lines = [l for l in resp.text.splitlines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"


# ---------------------------------------------------------------------------
# TTS: /v1/audio/speech and /v1/audio/dialogue
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_load_tts():
    with providers.load_tts() as model:
        assert model is not None


@pytest.mark.integration
def test_speech_wav(http_client):
    resp = http_client.post(
        "/v1/audio/speech",
        json={"input": "Hello.", "voice": "af_heart", "response_format": "wav"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "audio/wav"
    # Verify it's a valid WAV by parsing the header
    with wave.open(io.BytesIO(resp.content)) as wf:
        assert wf.getnframes() > 0


@pytest.mark.integration
def test_dialogue_wav(http_client):
    resp = http_client.post(
        "/v1/audio/dialogue",
        json={
            "segments": [
                {"voice": "af_heart", "text": "Hi."},
                {"voice": "am_adam", "text": "Hello."},
            ],
            "response_format": "wav",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(resp.content)) as wf:
        assert wf.getnframes() > 0


# ---------------------------------------------------------------------------
# STT: /v1/audio/transcriptions
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_load_stt():
    with providers.load_stt() as model:
        assert model is not None


@pytest.mark.integration
def test_transcription_returns_text(http_client):
    resp = http_client.post(
        "/v1/audio/transcriptions",
        json={"audio": make_wav_b64(duration_s=1.0), "language": "en"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "text" in body
    assert isinstance(body["text"], str)


# ---------------------------------------------------------------------------
# VLM: /v1/vision
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_load_vlm():
    with providers.load_vlm() as (model, processor, config):
        assert model is not None
        assert processor is not None


@pytest.mark.integration
def test_vision_describes_image(http_client):
    resp = http_client.post(
        "/v1/vision",
        json={
            "image": make_png_b64(),
            "prompt": "What color is this image?",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "text" in body
    assert isinstance(body["text"], str)
    assert len(body["text"]) > 0
