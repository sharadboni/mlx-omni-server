from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models import ChatCompletionRequest
from ..providers import load_llm


# Recommended sampling parameters per Qwen3.5 model card.
# thinking=True/None uses thinking defaults; thinking=False uses instruct defaults.
_THINKING_DEFAULTS = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)
_INSTRUCT_DEFAULTS = dict(temperature=0.7, top_p=0.8,  top_k=20, min_p=0.0)


def _make_generation_kwargs(req: ChatCompletionRequest) -> dict:
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    defaults = _INSTRUCT_DEFAULTS if req.thinking is False else _THINKING_DEFAULTS
    kwargs: dict = {
        "max_tokens": req.max_tokens,
        "sampler": make_sampler(
            temp=req.temperature  if req.temperature  is not None else defaults["temperature"],
            top_p=req.top_p       if req.top_p        is not None else defaults["top_p"],
            top_k=req.top_k       if req.top_k        is not None else defaults["top_k"],
            min_p=req.min_p       if req.min_p        is not None else defaults["min_p"],
        ),
    }
    lp = make_logits_processors(
        presence_penalty=req.presence_penalty,
        repetition_penalty=req.repetition_penalty,
        frequency_penalty=req.frequency_penalty,
    )
    if lp:
        kwargs["logits_processors"] = lp
    return kwargs

log = logging.getLogger(__name__)
router = APIRouter()

# Max seconds to wait for the next token before declaring generation stuck.
# Reasoning models can pause 30-60 s mid-stream, so keep this generous.
_TOKEN_TIMEOUT = 120.0
# Max seconds for a non-streaming (blocking) completion.
_BLOCKING_TIMEOUT = 300.0


def _build_prompt(tokenizer, messages: list, thinking: bool | None = None) -> str:
    raw = [{"role": m.role, "content": m.content} for m in messages]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    return tokenizer.apply_chat_template(raw, **kwargs)


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    try:
        if req.stream:
            return StreamingResponse(
                _stream_response(req, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_blocking_response, req),
                timeout=_BLOCKING_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail=f"Generation timed out after {_BLOCKING_TIMEOUT:.0f}s")
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat completions failed")
        raise HTTPException(status_code=500, detail=str(e))


def _blocking_response(req: ChatCompletionRequest) -> dict:
    from mlx_lm import generate

    model_id = req.model or _default_model_id()
    with load_llm(req.model) as (model, tokenizer):
        prompt = _build_prompt(tokenizer, req.messages, req.thinking)
        prompt_tokens = len(tokenizer.encode(prompt))

        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            **_make_generation_kwargs(req),
        )
        completion_tokens = len(tokenizer.encode(text))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _stream_response(req: ChatCompletionRequest, request: Request):
    from mlx_lm import stream_generate

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_id = req.model or _default_model_id()

    # First chunk: announce the assistant role
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    q: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    cancel = threading.Event()

    gen_kwargs = _make_generation_kwargs(req)

    def generate_thread():
        try:
            with load_llm(req.model) as (model, tokenizer):
                prompt = _build_prompt(tokenizer, req.messages, req.thinking)
                for chunk in stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    **gen_kwargs,
                ):
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, chunk.text)
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, f"\n\n[ERROR: {exc}]")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    thread = threading.Thread(target=generate_thread, daemon=True)
    thread.start()

    try:
        while True:
            if await request.is_disconnected():
                log.info("Client disconnected, cancelling generation")
                cancel.set()
                return

            try:
                token = await asyncio.wait_for(q.get(), timeout=_TOKEN_TIMEOUT)
            except asyncio.TimeoutError:
                log.error("No token in %.0fs — generation appears stuck, aborting stream", _TOKEN_TIMEOUT)
                cancel.set()
                return

            if token is None:
                break
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    finally:
        cancel.set()

    # Final chunk
    done_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join(timeout=5.0)


def _default_model_id() -> str:
    from ..config import DEFAULT_LLM_MODEL
    return DEFAULT_LLM_MODEL
