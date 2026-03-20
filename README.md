# MLX Omni Server

An OpenAI-compatible API server for text-to-speech, speech-to-text, and vision language models — powered by [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

## Features

- **Text-to-Speech** — Generate speech from text with multiple voice and format options
- **Speech-to-Text** — Transcribe audio from file uploads or base64-encoded input
- **Vision Language Model** — Analyze and describe images using a multimodal LLM
- **OpenAI-compatible endpoints** — Drop-in replacement for existing OpenAI API integrations
- **Memory management** — Models are loaded on demand and optionally cached between requests

## Requirements

- Python >= 3.11
- Apple Silicon Mac (for MLX acceleration)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
make install
# or
uv sync
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

### Health Check

```
GET /health
```

Returns `{"status": "ok"}`.

### Text-to-Speech

```
POST /v1/audio/speech
```

```json
{
  "input": "Hello, world!",
  "voice": "Chelsie",
  "speed": 1.0,
  "response_format": "mp3"
}
```

Supported formats: `mp3`, `wav`, `aac`, `opus`, `flac`, `pcm`

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

| Capability | Model |
|------------|-------|
| TTS | `mlx-community/Qwen-TTS-12Hz-0.6B-Base-bf16` |
| STT | `mlx-community/Qwen-ASR-0.6B-8BIT` |
| VLM | `mlx-community/Qwen2.5-VL-3B-Instruct-8bit` |

## Project Structure

```
server/
├── app.py          # FastAPI application and CLI entry point
├── config.py       # Default model identifiers
├── models.py       # Pydantic request/response schemas
├── providers.py    # Model loading and memory management
└── routes/
    ├── tts.py      # /v1/audio/speech
    ├── stt.py      # /v1/audio/transcriptions
    └── vlm.py      # /v1/vision
```
