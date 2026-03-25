"""Tests for /health, /state, and /instance/previews endpoints."""
from __future__ import annotations

import server.providers as providers
from tests.conftest import client  # noqa: F401 — fixture import


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_state_empty_cache(client):
    providers._cache.clear()
    providers.keep_in_memory = False

    resp = client.get("/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["keep_in_memory"] is False
    assert body["loaded_models"] == []


def test_state_with_cached_model(client):
    providers._cache["llm:test-model"] = object()
    try:
        resp = client.get("/state")
        assert resp.status_code == 200
        assert "llm:test-model" in resp.json()["loaded_models"]
    finally:
        providers._cache.pop("llm:test-model", None)


def test_instance_previews_not_loaded(client):
    providers._cache.clear()
    resp = client.get("/instance/previews", params={"model_id": "mlx-community/Qwen3.5-4B-4bit"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "mlx-community/Qwen3.5-4B-4bit"
    assert body["loaded"] is False


def test_instance_previews_loaded(client):
    key = "llm:mlx-community/Qwen3.5-4B-4bit"
    providers._cache[key] = object()
    try:
        resp = client.get("/instance/previews", params={"model_id": "mlx-community/Qwen3.5-4B-4bit"})
        assert resp.status_code == 200
        assert resp.json()["loaded"] is True
    finally:
        providers._cache.pop(key, None)


def test_instance_previews_no_model_id(client):
    resp = client.get("/instance/previews")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] is None
    assert body["loaded"] is False
