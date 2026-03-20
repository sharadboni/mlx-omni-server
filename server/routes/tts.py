from __future__ import annotations

import io
import subprocess
import tempfile
import os

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
    "opus": "audio/ogg; codecs=opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

# Formats that mlx_audio.audio_io.write supports natively
NATIVE_FORMATS = {"wav", "mp3"}


def _convert_with_ffmpeg(wav_bytes: bytes, target_fmt: str) -> bytes:
    """Convert wav audio to target format using ffmpeg."""
    in_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    ext = "ogg" if target_fmt == "opus" else target_fmt
    out_file = in_file.name.replace(".wav", f".{ext}")
    try:
        in_file.write(wav_bytes)
        in_file.close()
        cmd = ["ffmpeg", "-y", "-i", in_file.name]
        if target_fmt == "opus":
            cmd += ["-c:a", "libopus", "-b:a", "64k"]
        elif target_fmt == "aac":
            cmd += ["-c:a", "aac", "-b:a", "64k"]
        elif target_fmt == "flac":
            cmd += ["-c:a", "flac"]
        cmd.append(out_file)
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
        with open(out_file, "rb") as f:
            return f.read()
    finally:
        os.unlink(in_file.name)
        if os.path.exists(out_file):
            os.unlink(out_file)


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

            if req.response_format in NATIVE_FORMATS:
                audio_write(buf, audio, sample_rate, format=req.response_format)
                buf.seek(0)
            else:
                # Write as wav first, then convert with ffmpeg
                audio_write(buf, audio, sample_rate, format="wav")
                converted = _convert_with_ffmpeg(buf.getvalue(), req.response_format)
                buf = io.BytesIO(converted)

        mime = MIME_TYPES.get(req.response_format, "application/octet-stream")
        return StreamingResponse(buf, media_type=mime)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")
