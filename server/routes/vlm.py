from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..models import VisionRequest
from ..providers import load_vlm

router = APIRouter()

_SIGNATURES = {
    b"\x89PNG": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"GIF8": ".gif",
    b"BM": ".bmp",
    b"RIFF": ".webp"
}

def _detect_extension(img_bytes: bytes) -> str:
    for signature, ext in _SIGNATURES.items():
        if img_bytes.startswith(signature):
            return ext
    raise ValueError("Unsupported image format")



@router.post("/v1/vision")
async def vision(request: VisionRequest):
    tmp_path = None
    try:
        img_bytes = base64.b64decode(request.image)
        ext = _detect_extension(img_bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(img_bytes)
            tmp_path = tmp_file.name
        with load_vlm() as (model, processor, config):
            from mlx_vlm import apply_chat_template, generate
            prompt = apply_chat_template(
                processor,
                config,
                [{"role": "user", "content": request.prompt}],
                num_images=1,
            )
            output = generate(model, processor, prompt, image=[tmp_path], max_tokens=request.max_tokens, temperature=request.temperature)
            text = output.text if hasattr(output, "text") else str(output)
            return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        
