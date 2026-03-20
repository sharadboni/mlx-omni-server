from __future__ import annotations

import io
from xml.parsers.expat import model

import numpy as np
from fastapi import APIRouter, HTTPException 
from fastapi.responses import StreamingResponse 
from mlx_audio.audio_io import write as audio_write

from ..models import SpeechRequest
from ..providers import load_tts

router = APIRouter()

MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

@router.post("/v1/audio/speech") 
async def create_speech(req: SpeechRequest):
    try:
        with load_tts() as model: 
            chunks: list[np.ndarray] = [] 
            sample_rate = 24000
            for result in model.generate(
                text=req.input, 
                voice=req.voice, 
                speed=req.speed,
             ):
                chunks.append(np.array(result.audio)) 
                sample_rate = result.sample_rate
            
            if not chunks:
                raise HTTPException(status_code=500, detail="TTS produced no audio")

            audio = np.concatenate(chunks)
            buf = io.BytesIO()
            audio_write(buf, audio, sample_rate, format=req.response_format)
            buf.seek(0)

        mime = MIME_TYPES.get(req.response_format, "application/octet-stream")
        return StreamingResponse(buf, media_type=mime)

    except HTTPException:
        raise  # Re-raise HTTP exceptions so they are handled properly

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")