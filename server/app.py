from __future__ import annotations

import argparse
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import providers
from .routes import vlm
from .routes import stt
from .routes import tts

logging.basicConfig(level=logging.INFO)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(vlm.router)
app.include_router(stt.router)
app.include_router(tts.router)

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    parser = argparse.ArgumentParser(description="MLX Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--keep-in-memory", action="store_true", help="Keep models in memory between requests")
    args = parser.parse_args()

    providers.KEEP_IN_MEMORY = args.keep_in_memory

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()