from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel

## TTS

class SpeechRequest(BaseModel):
    input: str
    voice: str = "Chelsie"
    speed: float = 1.0
    response_format: Literal["wav", "mp3", "aac", "opus", "flac", "pcm"] = "mp3"

## STT

class TranscriptionRequest(BaseModel):
    audio: str
    language: str = "en"

## VLM

class VisionRequest(BaseModel):
    image: str # base64 encoded image
    prompt: str = "Describe the image in detail."
    max_tokens: int = 2048
    temperature: float = 0.7
