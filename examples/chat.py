#!/usr/bin/env python3
"""
Real-time CLI chat with PersonaPlex via WebSocket.

Usage:
    python examples/chat.py [--voice NATF2] [--url ws://localhost:8765]

Press Ctrl+C to end the session.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 24000
FRAME_SIZE = 1920
PREBUFFER_FRAMES = 3    # 240ms buffer


def _check_deps():
    missing = []
    for mod in ("sounddevice", "websockets"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"Missing: pip install {' '.join(missing)}")
        sys.exit(1)


def _status(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


async def run(url: str, voice: str, vad_threshold: float, silence_frames: int) -> None:
    import sounddevice as sd
    import websockets

    playback_buf: collections.deque[np.ndarray] = collections.deque()
    playback_started = threading.Event()

    def output_callback(outdata: np.ndarray, frames: int, time_info, status_flags) -> None:
        if not playback_started.is_set():
            if len(playback_buf) >= PREBUFFER_FRAMES:
                playback_started.set()
            else:
                outdata[:] = 0.0
                return
        if playback_buf:
            outdata[:, 0] = playback_buf.popleft()
        else:
            outdata[:] = 0.0

    out_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=FRAME_SIZE, callback=output_callback,
    )

    loop = asyncio.get_event_loop()
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def input_callback(indata: np.ndarray, frames: int, time_info, status_flags) -> None:
        pcm = np.clip(indata[:, 0] * 32767, -32768, 32767).astype(np.int16)
        loop.call_soon_threadsafe(mic_queue.put_nowait, pcm.tobytes())

    in_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=FRAME_SIZE, callback=input_callback,
    )

    ws_url = f"{url}/v1/audio/speech-to-speech/ws"
    print(f"Connecting to {ws_url} ...")

    async with websockets.connect(ws_url, max_size=None, ping_timeout=None) as ws:
        cfg = json.dumps({
            "voice": voice,
            "vad_threshold": vad_threshold,
            "silence_frames": silence_frames,
        })
        await ws.send(cfg)
        await ws.recv()  # {"status": "ready"}

        print(f"Connected  (voice={voice}, vad_threshold={vad_threshold})")
        print(f"Speak now — the model will respond after a short pause.\n")

        out_stream.start()
        in_stream.start()

        chunks_received = 0
        agent_speaking = False  # mute mic while agent talks to prevent feedback

        async def send_loop() -> None:
            while True:
                data = await mic_queue.get()
                if not agent_speaking:
                    await ws.send(data)

        async def recv_loop() -> None:
            nonlocal chunks_received, agent_speaking
            t_start = None
            async for message in ws:
                if isinstance(message, bytes):
                    pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    playback_buf.append(pcm)
                    chunks_received += 1
                    if chunks_received == 1:
                        agent_speaking = True
                        elapsed = f"{time.time() - t_start:.1f}s" if t_start else "?"
                        _status(f"Agent speaking  (first audio after {elapsed})")
                else:
                    event = json.loads(message)
                    s = event.get("status", "")
                    if s == "listening":
                        playback_buf.clear()
                        playback_started.clear()
                        chunks_received = 0
                        _status("Listening...")
                    elif s == "processing":
                        t_start = time.time()
                        _status("Processing your speech — first audio in ~3-10s...")
                    elif s == "ready":
                        # Wait for playback buffer to drain before unmuting mic
                        while playback_buf:
                            await asyncio.sleep(0.1)
                        await asyncio.sleep(0.4)  # extra tail silence
                        agent_speaking = False
                        _status("Ready — speak now")
                    elif "error" in event:
                        _status(f"Server error: {event['error']}")

        try:
            await asyncio.gather(send_loop(), recv_loop())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            in_stream.stop()
            in_stream.close()
            out_stream.stop()
            out_stream.close()


def main() -> None:
    _check_deps()
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="NATF2")
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--vad-threshold", type=float, default=0.02)
    parser.add_argument("--silence-frames", type=int, default=10,
                        help="Silent frames to trigger response (~800ms)")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.url, args.voice, args.vad_threshold, args.silence_frames))
    except KeyboardInterrupt:
        print("\nSession ended.")


if __name__ == "__main__":
    main()
