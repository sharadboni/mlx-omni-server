from __future__ import annotations

import base64
import io
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from mlx_audio.audio_io import write as audio_write

from ..models import DialogueRequest, SpeechRequest
from ..providers import load_tts, load_tts_clone, load_tts_vibevoice
from ..config import DEFAULT_TTS_MODEL, DEFAULT_TTS_VIBEVOICE_MODEL

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

# Kokoro voice prefix → (language, gender)
KOKORO_PREFIX = {
    "af": ("en-us", "female"), "am": ("en-us", "male"),
    "bf": ("en-gb", "female"), "bm": ("en-gb", "male"),
    "ef": ("es",    "female"), "em": ("es",    "male"),
    "ff": ("fr",    "female"),
    "hf": ("hi",    "female"), "hm": ("hi",    "male"),
    "if": ("it",    "female"), "im": ("it",    "male"),
    "jf": ("ja",    "female"), "jm": ("ja",    "male"),
    "pf": ("pt",    "female"), "pm": ("pt",    "male"),
    "zf": ("zh",    "female"), "zm": ("zh",    "male"),
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

# Kokoro language code → full language name
KOKORO_LANG = {
    "en-us": "American English",
    "en-gb": "British English",
    "es":    "Spanish",
    "fr":    "French",
    "hi":    "Hindi",
    "it":    "Italian",
    "ja":    "Japanese",
    "pt":    "Portuguese",
    "zh":    "Chinese",
}

# VibeVoice language prefix → full language name
# Note: VibeVoice uses non-standard codes (jp/kr/sp instead of ja/ko/es)
VIBEVOICE_LANG = {
    "de": "German",
    "en": "English",
    "fr": "French",
    "in": "Indian English",
    "it": "Italian",
    "jp": "Japanese",
    "kr": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "sp": "Spanish",
}

# Default alternating voices when the caller omits voice on a segment.
# Index 0 = first speaker, index 1 = second speaker (cycles for longer dialogues).
DEFAULT_VOICES: dict[str, list[str]] = {
    "vibevoice": ["en-Emma_woman", "en-Carter_man"],
    "kokoro":    ["af_heart",      "am_adam"],
}


def _list_voices(model_id: str) -> list[dict]:
    """Return voice metadata for a model without loading its weights."""
    from mlx_audio.utils import get_model_path
    try:
        model_path = Path(get_model_path(model_id))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not resolve model path: {e}")

    voices_dir = model_path / "voices"
    if not voices_dir.exists():
        return []

    voices = []
    for f in sorted(voices_dir.glob("*.safetensors")):
        name = f.stem
        entry: dict = {"name": name}

        # Kokoro: af_heart → language=en-us, language_name=American English, gender=female
        prefix = name[:2]
        if prefix in KOKORO_PREFIX:
            lang, gender = KOKORO_PREFIX[prefix]
            entry["language"] = lang
            entry["language_name"] = KOKORO_LANG.get(lang, lang)
            entry["gender"] = gender
        # VibeVoice: en-Emma_woman → language=en, language_name=English, gender=female
        elif "-" in name and "_" in name:
            lang_part, rest = name.split("-", 1)
            entry["language"] = lang_part
            entry["language_name"] = VIBEVOICE_LANG.get(lang_part, lang_part)
            entry["gender"] = "female" if rest.endswith("_woman") else "male" if rest.endswith("_man") else "unknown"

        voices.append(entry)

    return voices



@router.get("/v1/audio/voices")
async def list_voices(
    model: Literal["kokoro", "vibevoice"] = Query(
        default="kokoro",
        description="Which TTS model to list voices for. Use 'kokoro' for /speech, 'vibevoice' for /dialogue.",
    )
):
    model_id = DEFAULT_TTS_VIBEVOICE_MODEL if model == "vibevoice" else DEFAULT_TTS_MODEL
    voices = _list_voices(model_id)
    return {"model": model, "voices": voices, "count": len(voices)}


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



@router.post("/v1/audio/dialogue")
async def create_dialogue(req: DialogueRequest):
    if not req.segments:
        raise HTTPException(status_code=400, detail="No segments provided")

    try:
        defaults = DEFAULT_VOICES[req.model]
        valid_voices = {v["name"] for v in _list_voices(
            DEFAULT_TTS_VIBEVOICE_MODEL if req.model == "vibevoice" else DEFAULT_TTS_MODEL
        )}
        voices = [
            seg.voice if seg.voice in valid_voices else defaults[i % len(defaults)]
            for i, seg in enumerate(req.segments)
        ]
        substituted = [seg.voice for seg in req.segments if seg.voice not in valid_voices]
        if substituted:
            log.info(f"Voices not supported by {req.model}, substituted with defaults: {substituted}")

        # Split by generation method: preset voice vs voice cloning
        preset = [(i, seg, voices[i]) for i, seg in enumerate(req.segments) if not seg.ref_audio]
        cloned = [(i, seg) for i, seg in enumerate(req.segments) if seg.ref_audio]

        results: dict[int, tuple[np.ndarray, int]] = {}

        # --- Preset-voice segments: one model load for all ---
        if preset:
            if req.model == "vibevoice" and not cloned:
                # Fast path: native multi-speaker batch, one GPU pass
                with load_tts_vibevoice() as model:
                    audio, sr = _generate_and_collect(model, {
                        "text": [seg.text for _, seg, _ in preset],
                        "voice": [v for _, _, v in preset],
                    })
                # Native batch returns one combined array; split is not needed —
                # we store it all at index -1 and bypass reassembly below.
                sample_rate = sr
                final_audio = audio
            else:
                # Mixed or kokoro: generate each preset segment within one model load
                if req.model == "vibevoice":
                    loader = load_tts_vibevoice
                else:
                    loader = load_tts

                with loader() as model:
                    for i, seg, voice in preset:
                        if req.model == "vibevoice":
                            seg_audio, sr = _generate_and_collect(
                                model, {"text": seg.text, "voice": voice}
                            )
                        else:
                            voice_prefix = voice[0] if voice else "a"
                            gen_kwargs = dict(text=seg.text, voice=voice, speed=req.speed,
                                              lang_code=LANG_MAP.get(voice_prefix, "a"))
                            try:
                                seg_audio, sr = _generate_and_collect(model, gen_kwargs)
                            except TypeError:
                                del gen_kwargs["lang_code"]
                                seg_audio, sr = _generate_and_collect(model, gen_kwargs)
                        results[i] = (seg_audio, sr)

        # --- Cloned-voice segments: one Qwen3-TTS load for all ---
        if cloned:
            with load_tts_clone() as model:
                for i, seg in cloned:
                    ref_bytes = base64.b64decode(seg.ref_audio)
                    ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    ref_file.write(ref_bytes)
                    ref_file.close()
                    try:
                        gen_kwargs = dict(text=seg.text, ref_audio=ref_file.name, speed=req.speed)
                        if seg.ref_text:
                            gen_kwargs["ref_text"] = seg.ref_text
                        seg_audio, sr = _generate_and_collect(model, gen_kwargs)
                        results[i] = (seg_audio, sr)
                    finally:
                        os.unlink(ref_file.name)

        # --- Reassemble in original segment order ---
        if not results:
            # Pure vibevoice fast path already set final_audio above
            pass
        else:
            audio_parts: list[np.ndarray] = []
            sample_rate = 24000
            for i in range(len(req.segments)):
                seg_audio, sr = results[i]
                sample_rate = sr
                audio_parts.append(seg_audio)
                if req.pause_ms > 0:
                    silence = int(sample_rate * req.pause_ms / 1000)
                    audio_parts.append(np.zeros(silence, dtype=seg_audio.dtype))
            if req.pause_ms > 0 and len(audio_parts) > 1:
                audio_parts.pop()
            final_audio = np.concatenate(audio_parts)

        buf = _encode_audio(final_audio, sample_rate, req.response_format)
        mime = MIME_TYPES.get(req.response_format, "application/octet-stream")
        return StreamingResponse(buf, media_type=mime)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Dialogue TTS failed")
        raise HTTPException(status_code=500, detail=f"Dialogue TTS failed: {e}")
