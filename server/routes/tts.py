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
from ..providers import load_tts, load_tts_clone

router = APIRouter()

MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/ogg; codecs=opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

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


def _generate_and_collect(model, gen_kwargs: dict) -> tuple[np.ndarray, int]:
    """Run model.generate() and collect all chunks into one array."""
    chunks: list[np.ndarray] = []
    sample_rate = 24000
    for result in model.generate(**gen_kwargs):
        chunks.append(np.array(result.audio))
        sample_rate = result.sample_rate
    if not chunks:
        raise HTTPException(status_code=500, detail="TTS produced no audio")
    return np.concatenate(chunks), sample_rate


def _encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> io.BytesIO:
    """Encode audio array to the requested format."""
    buf = io.BytesIO()
    if fmt in NATIVE_FORMATS:
        audio_write(buf, audio, sample_rate, format=fmt)
        buf.seek(0)
    else:
        audio_write(buf, audio, sample_rate, format="wav")
        converted = _convert_with_ffmpeg(buf.getvalue(), fmt)
        buf = io.BytesIO(converted)
    return buf


@router.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    try:
        if req.ref_audio:
            # Voice cloning → use Qwen3-TTS 1.7B
            import base64
            ref_bytes = base64.b64decode(req.ref_audio)
            ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            ref_file.write(ref_bytes)
            ref_file.close()
            try:
                with load_tts_clone() as model:
                    gen_kwargs = dict(
                        text=req.input,
                        ref_audio=ref_file.name,
                        speed=req.speed,
                    )
                    if req.ref_text:
                        gen_kwargs["ref_text"] = req.ref_text
                    audio, sr = _generate_and_collect(model, gen_kwargs)
            finally:
                os.unlink(ref_file.name)
        else:
            # Regular TTS → use Kokoro (fast)
            with load_tts() as model:
                gen_kwargs = dict(
                    text=req.input,
                    voice=req.voice,
                    speed=req.speed,
                )
                # Kokoro needs lang_code
                if req.voice.startswith(("af_", "am_")):
                    gen_kwargs["lang_code"] = "a"  # American English
                elif req.voice.startswith(("bf_", "bm_")):
                    gen_kwargs["lang_code"] = "b"  # British English
                elif req.voice.startswith("jf_") or req.voice.startswith("jm_"):
                    gen_kwargs["lang_code"] = "j"  # Japanese
                else:
                    gen_kwargs["lang_code"] = "a"  # default American English
                audio, sr = _generate_and_collect(model, gen_kwargs)

        buf = _encode_audio(audio, sr, req.response_format)
        mime = MIME_TYPES.get(req.response_format, "application/octet-stream")
        return StreamingResponse(buf, media_type=mime)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")
