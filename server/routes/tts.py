from __future__ import annotations

import io
import logging
import subprocess
import tempfile
import os

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from mlx_audio.audio_io import write as audio_write

from ..models import DialogueRequest, SpeechRequest
from ..providers import load_tts, load_tts_clone

log = logging.getLogger(__name__)
router = APIRouter()

LANG_MAP = {
    "a": "a",  # American English
    "b": "b",  # British English
    "j": "j",  # Japanese
    "z": "z",  # Chinese
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "p": "p",  # Portuguese
}

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
                # Kokoro needs lang_code — derive from voice prefix
                voice_prefix = req.voice[0] if req.voice else "a"
                lang_code = LANG_MAP.get(voice_prefix, "a")
                # Try with lang_code, fall back without it (version compatibility)
                try:
                    gen_kwargs["lang_code"] = lang_code
                    audio, sr = _generate_and_collect(model, gen_kwargs)
                except TypeError:
                    del gen_kwargs["lang_code"]
                    audio, sr = _generate_and_collect(model, gen_kwargs)

        buf = _encode_audio(audio, sr, req.response_format)
        mime = MIME_TYPES.get(req.response_format, "application/octet-stream")
        return StreamingResponse(buf, media_type=mime)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("TTS failed")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


def _generate_segment_audio(
    segment, speed: float, tts_model, clone_model_loader
) -> tuple[np.ndarray, int]:
    """Generate audio for a single dialogue segment."""
    import base64

    if segment.ref_audio:
        ref_bytes = base64.b64decode(segment.ref_audio)
        ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ref_file.write(ref_bytes)
        ref_file.close()
        try:
            with clone_model_loader() as model:
                gen_kwargs = dict(text=segment.text, ref_audio=ref_file.name, speed=speed)
                if segment.ref_text:
                    gen_kwargs["ref_text"] = segment.ref_text
                return _generate_and_collect(model, gen_kwargs)
        finally:
            os.unlink(ref_file.name)
    else:
        voice_prefix = segment.voice[0] if segment.voice else "a"
        lang_code = LANG_MAP.get(voice_prefix, "a")
        gen_kwargs = dict(text=segment.text, voice=segment.voice, speed=speed)
        try:
            gen_kwargs["lang_code"] = lang_code
            return _generate_and_collect(tts_model, gen_kwargs)
        except TypeError:
            del gen_kwargs["lang_code"]
            return _generate_and_collect(tts_model, gen_kwargs)


@router.post("/v1/audio/dialogue")
async def create_dialogue(req: DialogueRequest):
    if not req.segments:
        raise HTTPException(status_code=400, detail="No segments provided")

    try:
        audio_parts: list[np.ndarray] = []
        sample_rate = 24000

        with load_tts() as tts_model:
            for seg in req.segments:
                audio, sr = _generate_segment_audio(
                    seg, req.speed, tts_model, load_tts_clone
                )
                sample_rate = sr
                audio_parts.append(audio)
                # Add silence between segments
                if req.pause_ms > 0:
                    silence_samples = int(sample_rate * req.pause_ms / 1000)
                    audio_parts.append(np.zeros(silence_samples, dtype=audio.dtype))

        # Remove trailing silence
        if req.pause_ms > 0 and len(audio_parts) > 1:
            audio_parts.pop()

        combined = np.concatenate(audio_parts)
        buf = _encode_audio(combined, sample_rate, req.response_format)
        mime = MIME_TYPES.get(req.response_format, "application/octet-stream")
        return StreamingResponse(buf, media_type=mime)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Dialogue TTS failed")
        raise HTTPException(status_code=500, detail=f"Dialogue TTS failed: {e}")
