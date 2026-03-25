from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models import ChatCompletionRequest
from ..providers import load_llm

router = APIRouter()


def _build_prompt(tokenizer, messages: list, thinking: bool | None = None) -> str:
    raw = [{"role": m.role, "content": m.content} for m in messages]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    return tokenizer.apply_chat_template(raw, **kwargs)


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        if req.stream:
            return StreamingResponse(
                _stream_response(req),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await asyncio.to_thread(_blocking_response, req)
    except HTTPException:
        raise
    except Exception as e:
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
            max_tokens=req.max_tokens,
            temp=req.temperature,
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


async def _stream_response(req: ChatCompletionRequest):
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

    def generate_thread():
        try:
            with load_llm(req.model) as (model, tokenizer):
                prompt = _build_prompt(tokenizer, req.messages, req.thinking)
                for chunk in stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    temp=req.temperature,
                ):
                    loop.call_soon_threadsafe(q.put_nowait, chunk.text)
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, f"\n\n[ERROR: {exc}]")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    thread = threading.Thread(target=generate_thread, daemon=True)
    thread.start()

    while True:
        token = await q.get()
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

    thread.join()


def _default_model_id() -> str:
    from ..config import DEFAULT_LLM_MODEL
    return DEFAULT_LLM_MODEL
