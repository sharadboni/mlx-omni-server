from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile

import asyncio

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..models import SpeechToSpeechRequest
from ..providers import load_s2s

router = APIRouter()

MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/ogg; codecs=opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


def _convert_with_ffmpeg(wav_bytes: bytes, target_fmt: str) -> bytes:
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


def _pcm_to_wav(pcm: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert float32 PCM array to WAV bytes."""
    import wave, struct
    pcm_int16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _read_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """Read WAV bytes into float32 PCM at 24kHz mono."""
    import wave, struct
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        sampwidth = wf.getsampwidth()
        nchannels = wf.getnchannels()
    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    if nchannels > 1:
        pcm = pcm.reshape(-1, nchannels)[:, 0]
    return pcm


def _pcm_chunk_to_int16_bytes(chunk: np.ndarray) -> bytes:
    return np.clip(chunk * 32767, -32768, 32767).astype(np.int16).tobytes()


@router.post("/v1/audio/speech-to-speech")
async def speech_to_speech(req: SpeechToSpeechRequest):
    try:
        audio_bytes = base64.b64decode(req.audio)
        user_audio = _read_wav_bytes(audio_bytes)

        if req.stream:
            # True streaming: yield raw int16 PCM chunks as each frame is generated.
            # The load_s2s context manager must remain open for the duration of streaming,
            # so we use a generator that holds the context.
            def _generate():
                with load_s2s() as (model, model_dir):
                    for chunk in model.stream_respond(
                        user_audio=user_audio,
                        voice=req.voice,
                        model_dir=model_dir,
                    ):
                        yield _pcm_chunk_to_int16_bytes(chunk)

            return StreamingResponse(
                _generate(),
                media_type="audio/pcm",
                headers={"X-Sample-Rate": "24000", "X-Channels": "1", "X-Bit-Depth": "16"},
            )

        # Non-streaming: generate everything, then return.
        with load_s2s() as (model, model_dir):
            response_audio, _ = model.respond(
                user_audio=user_audio,
                voice=req.voice,
                model_dir=model_dir,
            )

        wav_bytes = _pcm_to_wav(response_audio, sample_rate=24000)

        if req.response_format == "wav":
            audio_out = wav_bytes
        else:
            audio_out = _convert_with_ffmpeg(wav_bytes, req.response_format)

        buf = io.BytesIO(audio_out)
        mime = MIME_TYPES.get(req.response_format, "application/octet-stream")
        return StreamingResponse(buf, media_type=mime)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S2S failed: {e}")


@router.websocket("/v1/audio/speech-to-speech/ws")
async def speech_to_speech_ws(websocket: WebSocket):
    """
    VAD-triggered S2S streaming over WebSocket.

    Protocol:
      1. Client  → Server : JSON config  {"voice": "NATF2", "vad_threshold": 0.02, "silence_frames": 12}
      2. Server  → Client : JSON ready   {"status": "ready"}
      3. Client  → Server : binary frames  (signed int16 PCM, 24kHz mono, any chunk size)
      4. Server  → Client : JSON status  {"status": "listening" | "processing" | "speaking"}
                            binary frames  (signed int16 PCM, 24kHz mono, 3840 bytes = 1920 samples)
      5. Client closes connection to end the session.

    The server buffers incoming audio and uses a simple energy-based VAD to detect end-of-speech.
    Once silence is detected after speech, it processes the buffered utterance and streams
    the response back frame by frame while continuing to listen for the next turn.
    """
    await websocket.accept()
    try:
        config = await websocket.receive_json()
        voice = config.get("voice", "NATF2")
        # RMS energy threshold — raise if mic picks up background noise
        vad_threshold = float(config.get("vad_threshold", 0.02))
        # Consecutive silent frames required to trigger processing (~800ms at 12.5Hz)
        silence_trigger = int(config.get("silence_frames", 10))
        # Minimum speech frames before an utterance is considered real (~320ms)
        min_speech_frames = int(config.get("min_speech_frames", 4))

        with load_s2s() as (model, model_dir):
            await websocket.send_json({"status": "ready"})

            FRAME_SIZE = 1920
            loop = asyncio.get_event_loop()
            in_buf = np.array([], dtype=np.float32)
            speech_frames: list[np.ndarray] = []
            silence_count = 0
            speaking = False

            # Queue used to pipe generated chunks from the inference thread to the send task
            out_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

            async def send_chunks() -> None:
                """Drain out_queue and forward chunks to the client."""
                while True:
                    item = await out_queue.get()
                    if item is None:
                        break
                    await websocket.send_bytes(item)

            def run_inference(user_audio: np.ndarray) -> None:
                """Blocking: runs stream_respond and puts chunks into out_queue."""
                for chunk in model.stream_respond(
                    user_audio=user_audio, voice=voice, model_dir=model_dir,
                    max_steps=250,  # ~20s max response
                ):
                    out_bytes = np.clip(chunk * 32767, -32768, 32767).astype(np.int16).tobytes()
                    loop.call_soon_threadsafe(out_queue.put_nowait, out_bytes)
                loop.call_soon_threadsafe(out_queue.put_nowait, None)  # sentinel

            while True:
                data = await websocket.receive_bytes()
                if len(data) == 0:
                    break

                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                in_buf = np.concatenate([in_buf, chunk])

                while len(in_buf) >= FRAME_SIZE:
                    frame, in_buf = in_buf[:FRAME_SIZE], in_buf[FRAME_SIZE:]
                    rms = float(np.sqrt(np.mean(frame ** 2)))

                    if rms > vad_threshold:
                        if not speaking:
                            speaking = True
                            await websocket.send_json({"status": "listening"})
                        silence_count = 0
                        speech_frames.append(frame)
                    elif speaking:
                        silence_count += 1
                        speech_frames.append(frame)

                        if silence_count >= silence_trigger:
                            # End of utterance — only process if long enough to be real speech
                            real_speech = [f for f in speech_frames[:-silence_count]
                                           if float(np.sqrt(np.mean(f ** 2))) > vad_threshold]
                            user_audio = np.concatenate(speech_frames)
                            speech_frames = []
                            silence_count = 0
                            speaking = False

                            if len(real_speech) < min_speech_frames:
                                # Too short — noise burst, ignore
                                await websocket.send_json({"status": "ready"})
                                continue

                            import logging
                            logging.getLogger(__name__).info(
                                f"S2S processing {len(user_audio)/24000:.1f}s utterance"
                            )
                            await websocket.send_json({"status": "processing"})
                            # Run inference in a thread while send_chunks drains the queue
                            send_task = asyncio.create_task(send_chunks())
                            await asyncio.to_thread(run_inference, user_audio)
                            await send_task
                            await websocket.send_json({"status": "ready"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except Exception:
            pass
