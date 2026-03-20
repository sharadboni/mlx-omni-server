from __future__ import annotations

import base64
from pyexpat import model
import tempfile
from pathlib import Path
from turtle import write
from typing import Optional
from urllib import request

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..models import TranscriptionRequest
from ..providers import load_stt

router = APIRouter()

@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form(None),
):

    content_type = request.headers.get("content-type", )
    if "application/json" in content_type:
        body = await request.json()
        req = TranscriptionRequest(**body)
        return await _transcribe_base64(req)
    elif file is not None:
        return await _transcribe_file(file=file, language=language)
    else:
        raise HTTPException(status_code=400, detail="Provide either a file upload or JSON body with base64 audio")

async def _transcribe_file(file: UploadFile, language: Optional[str]) -> dict:
    tmp_path = None
    try:
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        return _run_stt(tmp_path, language)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

async def _transcribe_base64(req: TranscriptionRequest) -> dict: 
    tmp_path = None
    try:
        audio_bytes = base64.b64decode(req.audio)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        return _run_stt(tmp_path, req.language)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

def _run_stt(audio_path: str, language: Optional[str]) -> dict:
    try:
        with load_stt() as model:
            kwargs = {}
            if language:
                kwargs ["Language"] = language

        result = model.generate(audio_path, **kwargs)
        text = result["text"] if isinstance(result, dict) else result.text
        return {"text": text}      
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")