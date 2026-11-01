"""ASR interface and backend factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..audio_in import Audio
from ..config import AsrConfig


@dataclass
class Transcript:
    """A recognition result."""

    text: str
    no_speech_prob: float = 0.0


class AsrBackend(Protocol):
    """A speech recognizer taking 16 kHz mono float32 audio."""

    name: str

    def transcribe(self, audio: Audio) -> Transcript:
        """Recognize one utterance."""
        ...


def create_backend(cfg: AsrConfig) -> AsrBackend:
    """Build the configured backend."""
    if cfg.backend == "parakeet":
        from .parakeet import create_parakeet

        return create_parakeet(cfg)
    from .whisper import create_whisper

    return create_whisper(cfg)
