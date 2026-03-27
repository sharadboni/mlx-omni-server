# MLX Omni Server

An OpenAI-compatible API server for chat completions, text-to-speech, speech-to-text, vision language models, and speech-to-speech — powered by [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

## Features

- **Chat Completions** — OpenAI-compatible `/v1/chat/completions` powered by Qwen3.5, with tool calling, optional chain-of-thought thinking mode, and streaming
- **Text-to-Speech** — Dual-model TTS: Kokoro (82M, sub-second) for preset voices, Qwen3-TTS (1.7B) for voice cloning
- **Multi-Voice Dialogue** — Generate multi-speaker audio from a list of segments, each with its own voice, stitched with configurable pauses
- **Voice Cloning** — Clone any voice from a 3-second audio sample via `ref_audio` (automatically uses the larger model)
- **Speech-to-Text** — Transcribe audio from file uploads or base64-encoded input
- **Speech-to-Speech** — NVIDIA PersonaPlex 7B: full audio-in, audio-out conversation with voice presets, streaming and non-streaming modes
- **Vision Language Model** — Analyze and describe images using a multimodal LLM
- **OpenAI-compatible endpoints** — Drop-in replacement for existing OpenAI API integrations
- **Format conversion** — Automatic wav-to-opus/aac/flac conversion via ffmpeg
- **Memory management** — Models are loaded on demand and optionally cached between requests

## Requirements

- Python >= 3.11
- Apple Silicon Mac (for MLX acceleration)
- [uv](https://github.com/astral-sh/uv) package manager
- ffmpeg (for opus/aac/flac output): `brew install ffmpeg`

## Installation

```bash
make install
```

This runs `uv sync`, installs pip in the venv, and downloads the spacy English model needed by Kokoro for G2P.

Or manually:

```bash
uv sync
uv pip install pip
uv run python -m spacy download en_core_web_sm
```

## Usage

### Start the server

```bash
# Default: models unloaded after each request
make run

# Keep models in memory for faster repeated requests
make run-cached

# Custom host/port
make run HOST=0.0.0.0 PORT=8000
```

Or run directly:

```bash
uv run python -m server.app --host 0.0.0.0 --port 8765 --keep-in-memory
```

### Stop the server

```bash
make stop PORT=8765
```

## API Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible chat endpoint powered by [Qwen3.5-9B](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) (4-bit, default) or any `mlx-lm`-compatible model.

**Basic usage:**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}
```

**With all options:**

```json
{
  "model": "mlx-community/Qwen3.5-4B-4bit",
  "messages": [{"role": "user", "content": "Explain quantum entanglement."}],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false,
  "thinking": true
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `Qwen3.5-9B-4bit` | Any HuggingFace mlx-lm model ID |
| `messages` | array | required | Array of `{role, content}` objects (`system`/`user`/`assistant`/`tool`) |
| `temperature` | float | `0.7` | Sampling temperature |
| `max_tokens` | int | `1024` | Maximum tokens to generate |
| `stream` | boolean | `false` | Stream tokens via SSE |
| `thinking` | boolean | `null` | `true` enables chain-of-thought (`<think>` blocks), `false` disables it, `null` uses model default |
| `tools` | array | `null` | List of tool definitions (OpenAI function-calling format) |
| `tool_choice` | string | `"auto"` | `"auto"` lets the model decide, `"none"` disables tool use, `"required"` forces a call |

**Streaming example:**

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

**Thinking mode (Qwen3.5):**

Qwen3.5 supports a chain-of-thought reasoning mode. When enabled the response includes a `<think>...</think>` block before the final answer:

```bash
# Thinking on — model reasons step-by-step before answering
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 17 * 34?"}], "thinking": true}'

# Thinking off — direct answer, no reasoning trace
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 17 * 34?"}], "thinking": false}'
```

**Tool calling (Qwen3.5):**

Qwen3.5 supports function/tool calling natively. Pass a `tools` array in OpenAI format and the model will emit tool calls when appropriate. Multi-turn conversations with tool results use `role: "tool"` messages.

```json
{
  "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

When the model calls a tool the response has `finish_reason: "tool_calls"` and a `tool_calls` array. Pass the result back as a `role: "tool"` message:

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temperature\": 18, \"condition\": \"cloudy\"}"}
  ],
  "tools": [...]
}
```

**Fast model shortcut:**

Use `mlx-community/Qwen3.5-4B-4bit` for lower latency at the cost of some quality:

```json
{"model": "mlx-community/Qwen3.5-4B-4bit", "messages": [...]}
```

### Health Check

```
GET /health
```

Returns `{"status": "ok"}`.

### Server State

```
GET /state
```

Returns the current server state:

```json
{
  "status": "running",
  "keep_in_memory": false,
  "loaded_models": ["llm:mlx-community/Qwen3.5-9B-4bit"]
}
```

### Model Load Check

```
GET /instance/previews?model_id=<model_id>
```

Check whether a specific model is currently loaded in the cache:

```json
{
  "model_id": "mlx-community/Qwen3.5-9B-4bit",
  "loaded": true
}
```

### Text-to-Speech

```
POST /v1/audio/speech
```

**Basic usage (Kokoro — fast, preset voices):**

```json
{
  "input": "Hello, world!",
  "voice": "af_heart",
  "speed": 1.0,
  "response_format": "opus"
}
```

Kokoro voices: `af_heart`, `af_bella`, `af_nova`, `am_adam`, `am_echo`, `bf_alice`, `bm_daniel`, etc.

**Voice cloning (Qwen3-TTS 1.7B — automatic when ref_audio provided):**

```json
{
  "input": "Text to speak in the cloned voice.",
  "ref_audio": "<base64-encoded reference audio>",
  "ref_text": "Transcript of the reference audio.",
  "response_format": "opus"
}
```

The server automatically selects the right model: Kokoro for preset voices, Qwen3-TTS 1.7B for voice cloning.

Supported formats: `mp3`, `wav`, `aac`, `opus`, `flac`, `pcm`

Returns an audio stream with the appropriate MIME type.

### Multi-Voice Dialogue

```
POST /v1/audio/dialogue
```

Generate a single audio file from multiple speakers. Each segment specifies its own voice — preset voices use Kokoro, segments with `ref_audio` use Qwen3-TTS for voice cloning.

```json
{
  "segments": [
    {"voice": "af_heart", "text": "Welcome to the show! Today we're talking about AI."},
    {"voice": "am_adam", "text": "Thanks for having me. This is a fascinating topic."},
    {"voice": "af_heart", "text": "Let's dive right in."}
  ],
  "speed": 1.0,
  "response_format": "opus",
  "pause_ms": 500
}
```

**With voice cloning (mix preset and cloned voices):**

```json
{
  "segments": [
    {"voice": "af_heart", "text": "Welcome to the show!"},
    {"voice": "custom", "text": "Great to be here.", "ref_audio": "<base64-wav>", "ref_text": "Reference transcript."}
  ],
  "pause_ms": 300
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `segments` | array | required | List of `{voice, text, ref_audio?, ref_text?}` objects |
| `speed` | float | `1.0` | Playback speed multiplier |
| `response_format` | string | `"mp3"` | Output format: `mp3`, `wav`, `aac`, `opus`, `flac`, `pcm` |
| `pause_ms` | int | `500` | Silence between segments in milliseconds |

Returns an audio stream with the appropriate MIME type.

### Speech-to-Text

```
POST /v1/audio/transcriptions
```

**Multipart form upload:**

```bash
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "language=en"
```

**JSON with base64 audio:**

```json
{
  "audio": "<base64-encoded audio>",
  "language": "en"
}
```

Returns:

```json
{
  "text": "Transcribed text here"
}
```

### Speech-to-Speech

```
POST /v1/audio/speech-to-speech
```

Powered by [NVIDIA PersonaPlex 7B](https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit) (4-bit quantized MLX, ~4.9 GB). Takes audio in, produces audio out — no text intermediate. On first use the model weights are downloaded automatically.

**Non-streaming (returns complete audio when generation finishes):**

```json
{
  "audio": "<base64-encoded WAV at 24kHz>",
  "voice": "NATF2",
  "response_format": "wav"
}
```

Returns: audio file with the appropriate MIME type.

**Streaming (delivers 80ms PCM chunks as they are generated):**

```json
{
  "audio": "<base64-encoded WAV at 24kHz>",
  "voice": "NATF2",
  "stream": true
}
```

Returns: chunked `audio/pcm` stream (signed 16-bit little-endian, 24kHz mono) with headers:
- `X-Sample-Rate: 24000`
- `X-Channels: 1`
- `X-Bit-Depth: 16`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | string | required | Base64-encoded WAV at 24kHz |
| `voice` | string | `"NATF2"` | Voice preset (see below) |
| `response_format` | string | `"wav"` | Output format for non-streaming: `wav`, `mp3`, `aac`, `opus`, `flac`, `pcm` |
| `stream` | boolean | `false` | Enable streaming PCM output |

**Available voices:**

| Type | Voices |
|------|--------|
| Female (natural) | `NATF0`, `NATF1`, `NATF2`, `NATF3` |
| Male (natural) | `NATM0`, `NATM1`, `NATM2`, `NATM3` |
| Female (variant) | `VARF0`, `VARF1`, `VARF2`, `VARF3`, `VARF4` |
| Male (variant) | `VARM0`, `VARM1`, `VARM2`, `VARM3`, `VARM4` |

**Example (streaming with curl):**

```bash
# Encode input audio
AUDIO_B64=$(base64 -i input.wav)

# Stream raw PCM to ffplay for real-time playback
curl -s -X POST http://localhost:8765/v1/audio/speech-to-speech \
  -H "Content-Type: application/json" \
  -d "{\"audio\": \"$AUDIO_B64\", \"voice\": \"NATF2\", \"stream\": true}" \
  | ffplay -f s16le -ar 24000 -ac 1 -
```

### Real-time Conversation (WebSocket)

```
WS /v1/audio/speech-to-speech/ws
```

Voice-activity-detection (VAD) triggered conversation: the server listens until you stop speaking, then streams the agent's response back frame by frame as it is generated.

**Protocol:**

| Step | Direction | Content |
|------|-----------|---------|
| 1 | Client → Server | JSON config: `{"voice": "NATF2", "vad_threshold": 0.02, "silence_frames": 10}` |
| 2 | Server → Client | JSON: `{"status": "ready"}` |
| 3+ | Client → Server | Binary: signed int16 PCM at 24kHz (1920 samples / 80ms per frame) |
| — | Server → Client | JSON: `{"status": "listening"}` — new turn started |
| — | Server → Client | JSON: `{"status": "processing"}` — speech detected, inference running |
| — | Server → Client | Binary: signed int16 PCM at 24kHz (3840 bytes = 1920 samples per frame) |
| — | Server → Client | JSON: `{"status": "ready"}` — response complete, listening again |
| End | Client closes | — |

**Config parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voice` | `"NATF2"` | Voice preset |
| `vad_threshold` | `0.02` | RMS energy threshold to detect speech |
| `silence_frames` | `10` | Silent frames (~800ms) after speech to trigger response |
| `min_speech_frames` | `4` | Minimum speech frames to avoid reacting to noise |

**CLI chat tool:**

```bash
# Install audio dependencies
uv sync  # already included in project deps

# Start a real-time voice conversation
uv run python examples/chat.py --voice NATF2 --url ws://localhost:8765
```

Options: `--voice`, `--url`, `--vad-threshold`, `--silence-frames`

Press Ctrl+C to end the session. The tool mutes the microphone while the agent is speaking to prevent acoustic feedback.

### Vision

```
POST /v1/vision
```

```json
{
  "image": "<base64-encoded image>",
  "prompt": "Describe the image in detail.",
  "max_tokens": 2048,
  "temperature": 0.7
}
```

Supports PNG, JPEG, GIF, BMP, and WebP images.

Returns:

```json
{
  "text": "A detailed description of the image..."
}
```

## Default Models

| Capability | Model | Notes |
|------------|-------|-------|
| Chat (default) | `mlx-community/Qwen3.5-9B-4bit` | 9B params, 4-bit quantized |
| Chat (fast) | `mlx-community/Qwen3.5-4B-4bit` | 4B params, lower latency |
| TTS (preset voices) | `mlx-community/Kokoro-82M-bf16` | 82M params, sub-second, 54 voices |
| TTS (voice cloning) | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit` | Used automatically when `ref_audio` provided |
| STT | `mlx-community/Qwen3-ASR-0.6B-8bit` | |
| S2S | `aufklarer/PersonaPlex-7B-MLX-4bit` | 7B params, 4-bit quantized, ~4.9 GB |
| VLM | `mlx-community/Qwen2.5-VL-3B-Instruct-8bit` | |

## Project Structure

```
server/
├── app.py          # FastAPI application and CLI entry point
├── config.py       # Default model identifiers
├── models.py       # Pydantic request/response schemas
├── providers.py    # Model loading and memory management
├── personaplex/    # PersonaPlex S2S implementation (MLX port)
│   ├── model.py        # PersonaPlexModel + PersonaPlexStreamSession
│   ├── temporal.py     # TemporalTransformer (32-layer, 4096-dim)
│   ├── depformer.py    # Depformer (6-layer, per-step MultiLinear)
│   ├── weight_loading.py  # aufklarer split-format weight loading
│   ├── config.py       # Model hyperparameters and token constants
│   ├── sampling.py     # Top-k sampling with repetition penalty
│   └── kv_cache.py     # KV cache
└── routes/
    ├── chat.py     # /v1/chat/completions (streaming + thinking mode + tool calling)
    ├── tts.py      # /v1/audio/speech + /v1/audio/dialogue (voice cloning + multi-voice + ffmpeg)
    ├── stt.py      # /v1/audio/transcriptions
    ├── s2s.py      # /v1/audio/speech-to-speech (HTTP + WebSocket streaming)
    └── vlm.py      # /v1/vision

examples/
└── chat.py         # Real-time CLI voice chat via WebSocket
```
