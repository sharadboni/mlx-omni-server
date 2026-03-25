"""Tests for POST /v1/audio/transcriptions."""
from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from tests.conftest import client, make_wav_b64, make_wav_bytes, mock_stt_cm  # noqa: F401


def test_transcription_json_body(client):
    cm, stt_model = mock_stt_cm("hello world")

    with patch("server.routes.stt.load_stt", cm):
        resp = client.post(
            "/v1/audio/transcriptions",
            json={"audio": make_wav_b64(), "language": "en"},
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"text": "hello world"}


def test_transcription_file_upload(client):
    cm, stt_model = mock_stt_cm("file upload test")
    wav_bytes = make_wav_bytes()

    with patch("server.routes.stt.load_stt", cm):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )

    assert resp.status_code == 200
    assert resp.json()["text"] == "file upload test"


def test_transcription_language_passed(client):
    cm, stt_model = mock_stt_cm("bonjour")

    with patch("server.routes.stt.load_stt", cm):
        resp = client.post(
            "/v1/audio/transcriptions",
            json={"audio": make_wav_b64(), "language": "fr"},
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    stt_model.generate.assert_called_once()
    _, kwargs = stt_model.generate.call_args
    assert kwargs.get("language") == "fr"


def test_transcription_no_input(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        headers={"content-type": "multipart/form-data"},
    )
    assert resp.status_code == 400
