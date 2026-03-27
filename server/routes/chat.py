from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models import ChatCompletionRequest
from ..providers import load_llm


def _strip_thinking(text: str) -> str:
    """Remove thinking content up to and including </think>.

    The chat template prepends '<think>\\n' as the generation prompt, so the
    model output starts with raw thinking content and ends the block with
    '</think>\\n\\n' before the actual response.
    """
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip("\n")
    return text


class _ThinkingStreamFilter:
    """Buffer streaming tokens until </think> is seen, then pass through.

    The chat template adds '<think>\\n' to the generation prompt, so model
    output starts with thinking content and contains '</think>' before the
    real response. Everything up to and including '</think>' is suppressed.
    flush() emits any remaining buffer (safety net if </think> never arrives).
    """

    def __init__(self):
        self._buf = ""
        self._done = False

    def feed(self, token: str) -> str:
        if self._done:
            return token
        self._buf += token
        if "</think>" in self._buf:
            _, after = self._buf.split("</think>", 1)
            self._done = True
            self._buf = ""
            return after.lstrip("\n")
        return ""

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return out


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


_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>',
    re.DOTALL,
)
_PARAM_RE = re.compile(r'<parameter=([^>]+)>\n?(.*?)\n?</parameter>', re.DOTALL)
_TC_START = "<tool_call>"
_TC_END   = "</tool_call>"


def _parse_tool_calls(text: str) -> list[dict]:
    """Parse Qwen3.5 tool call XML into OpenAI-format tool_calls list."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        args: dict = {}
        for p in _PARAM_RE.finditer(m.group(2)):
            k, v = p.group(1).strip(), p.group(2).strip()
            try:
                args[k] = json.loads(v)
            except json.JSONDecodeError:
                args[k] = v
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return calls


def _build_prompt(tokenizer, messages: list, thinking: bool | None = None,
                  tools: list | None = None) -> str:
    raw = []
    for m in messages:
        msg: dict = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"function": {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }}
                for tc in m.tool_calls
            ]
        raw.append(msg)

    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    # Don't pass tools when tool_choice is "none" or no tools provided
    if tools:
        kwargs["tools"] = [t.model_dump() for t in tools]
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

    effective_tools = req.tools if req.tool_choice != "none" else None
    model_id = req.model or _default_model_id()
    with load_llm(req.model) as (model, tokenizer):
        prompt = _build_prompt(tokenizer, req.messages, req.thinking, effective_tools)
        prompt_tokens = len(tokenizer.encode(prompt))

        text = generate(model, tokenizer, prompt=prompt, **_make_generation_kwargs(req))
        if req.thinking is not False:
            text = _strip_thinking(text)
        completion_tokens = len(tokenizer.encode(text))

    tool_calls = _parse_tool_calls(text) if effective_tools else []
    if tool_calls:
        content = text[:text.index(_TC_START)].strip() if _TC_START in text else None
        message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish_reason = "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
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
    strip_thinking = req.thinking is not False
    effective_tools = req.tools if req.tool_choice != "none" else None

    def generate_thread():
        think_filter = _ThinkingStreamFilter() if strip_thinking else None
        # Peek buffer: hold back up to len(_TC_START) chars so we can detect
        # a tool call opening tag that spans multiple tokens without emitting early.
        peek_buf = ""
        tool_buf: str | None = None  # non-None while collecting a tool call block

        def _emit(text: str) -> None:
            if text:
                loop.call_soon_threadsafe(q.put_nowait, text)

        def _process(text: str) -> None:
            nonlocal peek_buf, tool_buf
            if tool_buf is not None:
                tool_buf += text
                if _TC_END in tool_buf:
                    calls = _parse_tool_calls(tool_buf)
                    if calls:
                        loop.call_soon_threadsafe(q.put_nowait, ("tool_calls", calls))
                    tool_buf = None
                return
            peek_buf += text
            if _TC_START in peek_buf:
                pre, rest = peek_buf.split(_TC_START, 1)
                _emit(pre)
                peek_buf = ""
                tool_buf = _TC_START + rest
                if _TC_END in tool_buf:
                    calls = _parse_tool_calls(tool_buf)
                    if calls:
                        loop.call_soon_threadsafe(q.put_nowait, ("tool_calls", calls))
                    tool_buf = None
            elif len(peek_buf) > len(_TC_START):
                safe, peek_buf = peek_buf[:-len(_TC_START)], peek_buf[-len(_TC_START):]
                _emit(safe)

        try:
            with load_llm(req.model) as (model, tokenizer):
                prompt = _build_prompt(tokenizer, req.messages, req.thinking, effective_tools)
                for chunk in stream_generate(model, tokenizer, prompt=prompt, **gen_kwargs):
                    if cancel.is_set():
                        break
                    text = think_filter.feed(chunk.text) if think_filter else chunk.text
                    if text:
                        _process(text) if effective_tools else _emit(text)
            # Flush
            if think_filter:
                rem = think_filter.flush()
                if rem:
                    _process(rem) if effective_tools else _emit(rem)
            if effective_tools:
                if peek_buf:
                    _emit(peek_buf)
                if tool_buf:
                    calls = _parse_tool_calls(tool_buf)
                    if calls:
                        loop.call_soon_threadsafe(q.put_nowait, ("tool_calls", calls))
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, f"\n\n[ERROR: {exc}]")
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    thread = threading.Thread(target=generate_thread, daemon=True)
    thread.start()

    had_tool_calls = False
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

            if isinstance(token, tuple) and token[0] == "tool_calls":
                had_tool_calls = True
                delta = {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for i, tc in enumerate(token[1])
                    ]
                }
            else:
                delta = {"content": token}

            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
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
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if had_tool_calls else "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join(timeout=5.0)


def _default_model_id() -> str:
    from ..config import DEFAULT_LLM_MODEL
    return DEFAULT_LLM_MODEL
