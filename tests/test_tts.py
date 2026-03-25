"""Tests for POST /v1/audio/speech and /v1/audio/dialogue."""
from __future__ import annotations

import io
import wave
from unittest.mock import patch

import numpy as np
import pytest

from tests.conftest import client, make_wav_b64, mock_tts_cm  # noqa: F401


def _fake_audio() -> np.ndarray:
    return np.zeros(2400, dtype=np.float32)


def _patch_audio_write(buf_out: io.BytesIO | None = None):
    """Patch mlx_audio.audio_io.write to write a valid WAV into the buffer."""

    def _write(buf, audio, sample_rate, format="wav"):
        samples = (audio * 32767).astype(np.int16).tobytes()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples)

    return patch("server.routes.tts.audio_write", side_effect=_write)


def test_speech_basic(client):
    cm, tts_model = mock_tts_cm(_fake_audio())

    with (
        patch("server.routes.tts.load_tts", cm),
        _patch_audio_write(),
    ):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello world", "voice": "af_heart", "response_format": "wav"},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 0


def test_speech_mp3_format(client):
    cm, tts_model = mock_tts_cm(_fake_audio())

    with (
        patch("server.routes.tts.load_tts", cm),
        _patch_audio_write(),
        patch("server.routes.tts._convert_with_ffmpeg", return_value=b"fake-mp3-data"),
    ):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "Hello", "response_format": "mp3"},
        )

    assert resp.status_code == 200
    assert "audio/mpeg" in resp.headers["content-type"]


def test_speech_voice_cloning(client):
    cm_clone, clone_model = mock_tts_cm(_fake_audio())
    ref_b64 = make_wav_b64()

    with (
        patch("server.routes.tts.load_tts_clone", cm_clone),
        _patch_audio_write(),
    ):
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": "Clone my voice",
                "ref_audio": ref_b64,
                "ref_text": "Reference text",
                "response_format": "wav",
            },
        )

    assert resp.status_code == 200
    clone_model.generate.assert_called_once()
    call_kwargs = clone_model.generate.call_args.kwargs
    assert call_kwargs["text"] == "Clone my voice"
    assert call_kwargs["ref_text"] == "Reference text"


def test_speech_missing_input(client):
    resp = client.post("/v1/audio/speech", json={})
    assert resp.status_code == 422


def test_dialogue_basic(client):
    cm, tts_model = mock_tts_cm(_fake_audio())

    with (
        patch("server.routes.tts.load_tts", cm),
        _patch_audio_write(),
    ):
        resp = client.post(
            "/v1/audio/dialogue",
            json={
                "segments": [
                    {"voice": "af_heart", "text": "Hello there"},
                    {"voice": "am_adam", "text": "Hi back"},
                ],
                "response_format": "wav",
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert tts_model.generate.call_count == 2


def test_dialogue_empty_segments(client):
    resp = client.post("/v1/audio/dialogue", json={"segments": []})
    assert resp.status_code == 400


def test_dialogue_pause_between_segments(client):
    """Silence padding should make combined audio longer than individual segments."""
    audio = np.ones(2400, dtype=np.float32) * 0.1
    cm, tts_model = mock_tts_cm(audio)

    captured: list[np.ndarray] = []

    def _capture_write(buf, arr, sample_rate, format="wav"):
        captured.append(arr.copy())
        samples = (arr * 32767).astype(np.int16).tobytes()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples)

    with (
        patch("server.routes.tts.load_tts", cm),
        patch("server.routes.tts.audio_write", side_effect=_capture_write),
    ):
        resp = client.post(
            "/v1/audio/dialogue",
            json={
                "segments": [
                    {"voice": "af_heart", "text": "A"},
                    {"voice": "am_adam", "text": "B"},
                ],
                "pause_ms": 500,
                "response_format": "wav",
            },
        )

    assert resp.status_code == 200
    assert len(captured) == 1
    # 2 segments × 2400 samples + 500ms×24000 silence = 4800 + 12000 = 16800
    assert len(captured[0]) == 2 * 2400 + int(24000 * 0.5)
