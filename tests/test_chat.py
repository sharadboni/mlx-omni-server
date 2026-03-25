"""Tests for POST /v1/chat/completions."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from tests.conftest import client, mock_llm_cm  # noqa: F401


MESSAGES = [{"role": "user", "content": "Say hi"}]


def _patch_llm(response_text: str = "Hello!"):
    cm, model, tokenizer = mock_llm_cm(response_text)

    generate_mock = MagicMock(return_value=response_text)
    make_sampler_mock = MagicMock(return_value=MagicMock())

    return (
        patch("server.providers.load_llm", cm),
        patch("server.routes.chat.load_llm", cm),
        patch("mlx_lm.generate", generate_mock),
        patch("mlx_lm.sample_utils.make_sampler", make_sampler_mock),
        model,
        tokenizer,
        generate_mock,
    )


def test_chat_completions_basic(client):
    cm, model, tokenizer = mock_llm_cm("Hello!")
    generate_mock = MagicMock(return_value="Hello!")
    make_sampler_mock = MagicMock(return_value=MagicMock())

    with (
        patch("server.routes.chat.load_llm", cm),
        patch("mlx_lm.generate", generate_mock),
        patch("mlx_lm.sample_utils.make_sampler", make_sampler_mock),
    ):
        resp = client.post("/v1/chat/completions", json={"messages": MESSAGES})

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert "usage" in body
    assert body["usage"]["total_tokens"] >= 0


def test_chat_completions_model_field(client):
    cm, model, tokenizer = mock_llm_cm("Hi")
    generate_mock = MagicMock(return_value="Hi")
    make_sampler_mock = MagicMock(return_value=MagicMock())

    with (
        patch("server.routes.chat.load_llm", cm),
        patch("mlx_lm.generate", generate_mock),
        patch("mlx_lm.sample_utils.make_sampler", make_sampler_mock),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "mlx-community/Qwen3.5-4B-4bit", "messages": MESSAGES},
        )

    assert resp.status_code == 200


def test_chat_completions_missing_messages(client):
    resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 422


def test_chat_completions_streaming(client):
    cm, model, tokenizer = mock_llm_cm()

    chunk_mock = MagicMock()
    chunk_mock.text = "Hello"

    make_sampler_mock = MagicMock(return_value=MagicMock())

    def fake_stream_generate(*args, **kwargs):
        yield chunk_mock

    with (
        patch("server.routes.chat.load_llm", cm),
        patch("mlx_lm.stream_generate", fake_stream_generate),
        patch("mlx_lm.sample_utils.make_sampler", make_sampler_mock),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": MESSAGES, "stream": True},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [l for l in resp.text.splitlines() if l.startswith("data:")]
    assert len(lines) >= 2  # at least one token chunk + [DONE]
    assert lines[-1] == "data: [DONE]"


def test_chat_completions_temperature(client):
    cm, model, tokenizer = mock_llm_cm("response")
    generate_mock = MagicMock(return_value="response")
    make_sampler_mock = MagicMock(return_value=MagicMock())

    with (
        patch("server.routes.chat.load_llm", cm),
        patch("mlx_lm.generate", generate_mock),
        patch("mlx_lm.sample_utils.make_sampler", make_sampler_mock),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": MESSAGES, "temperature": 0.0, "max_tokens": 256},
        )

    assert resp.status_code == 200
    make_sampler_mock.assert_called_once_with(temp=0.0)
