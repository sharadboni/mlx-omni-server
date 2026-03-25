from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel

## Chat Completions

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None   # defaults to DEFAULT_LLM_MODEL if omitted
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    thinking: Optional[bool] = None  # True=enable chain-of-thought, False=disable, None=model default

## TTS

class SpeechRequest(BaseModel):
    input: str
    voice: str = "Chelsie"
    speed: float = 1.0
    response_format: Literal["wav", "mp3", "aac", "opus", "flac", "pcm"] = "mp3"
    ref_audio: str | None = None   # base64-encoded reference audio for voice cloning
    ref_text: str | None = None    # transcript of the reference audio

class DialogueSegment(BaseModel):
    voice: str                        # voice identifier (e.g. "af_heart", "am_adam")
    text: str                         # text for this segment
    ref_audio: str | None = None      # base64 reference audio (voice cloning)
    ref_text: str | None = None       # transcript of reference audio

class DialogueRequest(BaseModel):
    segments: list[DialogueSegment]
    speed: float = 1.0
    response_format: Literal["wav", "mp3", "aac", "opus", "flac", "pcm"] = "mp3"
    pause_ms: int = 500               # silence between segments in milliseconds

## STT

class TranscriptionRequest(BaseModel):
    audio: str
    language: str = "en"

## S2S

class SpeechToSpeechRequest(BaseModel):
    audio: str  # base64-encoded WAV at 24kHz
    voice: str = "NATF2"
    persona: str = "You are a helpful assistant."
    response_format: Literal["wav", "mp3", "aac", "opus", "flac", "pcm"] = "wav"
    seed: int = 42424242
    stream: bool = False  # if True, stream raw int16 PCM chunks as they're generated

## VLM

class VisionRequest(BaseModel):
    image: str # base64 encoded image
    prompt: str = "Describe the image in detail."
    max_tokens: int = 2048
    temperature: float = 0.7
