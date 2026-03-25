from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from . import providers
from .routes import chat
from .routes import vlm
from .routes import stt
from .routes import tts
from .routes import s2s

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    log.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    log.info("← %s %s %d  %.3fs", request.method, request.url.path, response.status_code, elapsed)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(chat.router)
app.include_router(vlm.router)
app.include_router(stt.router)
app.include_router(tts.router)
app.include_router(s2s.router)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/state")
async def state():
    return {
        "status": "running",
        "keep_in_memory": providers.keep_in_memory,
        "loaded_models": list(providers._cache.keys()),
    }

@app.get("/instance/previews")
async def instance_previews(model_id: Optional[str] = Query(None)):
    cache_key = f"llm:{model_id}" if model_id else None
    loaded = cache_key in providers._cache if cache_key else False
    return {
        "model_id": model_id,
        "loaded": loaded,
    }

def main():
    parser = argparse.ArgumentParser(description="MLX Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--keep-in-memory", action="store_true", help="Keep models in memory between requests")
    args = parser.parse_args()

    providers.keep_in_memory = args.keep_in_memory

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, ws_ping_timeout=None)

if __name__ == "__main__":
    main()