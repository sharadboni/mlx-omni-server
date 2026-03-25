"""Tests for POST /v1/vision."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import client, make_png_b64, mock_vlm_cm  # noqa: F401


def test_vision_basic(client):
    cm, model, processor, config, output = mock_vlm_cm("A white image.")

    with (
        patch("server.routes.vlm.load_vlm", cm),
        patch("mlx_vlm.apply_chat_template", return_value="<prompt>"),
        patch("mlx_vlm.generate", return_value=output),
    ):
        resp = client.post(
            "/v1/vision",
            json={"image": make_png_b64(), "prompt": "What do you see?"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"text": "A white image."}


def test_vision_default_prompt(client):
    cm, model, processor, config, output = mock_vlm_cm("Described.")

    with (
        patch("server.routes.vlm.load_vlm", cm),
        patch("mlx_vlm.apply_chat_template", return_value="<prompt>") as template_mock,
        patch("mlx_vlm.generate", return_value=output),
    ):
        resp = client.post("/v1/vision", json={"image": make_png_b64()})

    assert resp.status_code == 200
    # Default prompt is "Describe the image in detail."
    call_args = template_mock.call_args
    messages = call_args.args[2]
    assert "Describe the image" in messages[0]["content"]


def test_vision_invalid_image(client):
    with patch("server.routes.vlm.load_vlm"):
        resp = client.post(
            "/v1/vision",
            json={"image": "bm90YW5pbWFnZQ=="},  # b64 of "notanimage"
        )
    assert resp.status_code == 500


def test_vision_missing_image(client):
    resp = client.post("/v1/vision", json={})
    assert resp.status_code == 422
