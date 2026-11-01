"""Whisper backends: faster-whisper (CUDA/CPU) and whisper.cpp (Apple Silicon)."""

from __future__ import annotations

import platform
import sys
from typing import Any

import numpy as np

from ..audio_in import Audio
from ..config import AsrConfig
from ..models import ensure_whisper, models_dir
from .base import AsrBackend, Transcript


def cuda_available() -> bool:
    """Return True if CTranslate2 can see a CUDA device."""
    import ctranslate2

    return int(ctranslate2.get_cuda_device_count()) > 0


def is_apple_silicon() -> bool:
    """Return True on arm64 macOS."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _initial_prompt(hotwords: list[str]) -> str | None:
    return ", ".join(hotwords) + "." if hotwords else None


class FasterWhisperBackend:
    """faster-whisper on CTranslate2."""

    name = "faster-whisper"

    def __init__(self, cfg: AsrConfig, device: str, model: Any = None) -> None:
        self.cfg = cfg
        if model is None:
            from faster_whisper import WhisperModel

            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel(
                str(ensure_whisper(cfg.model)), device=device, compute_type=compute_type
            )
        self.model = model

    def transcribe(self, audio: Audio) -> Transcript:
        """Recognize one utterance."""
        hotwords = self.cfg.hotwords
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language or None,
            beam_size=self.cfg.beam_size,
            initial_prompt=_initial_prompt(hotwords),
            hotwords=" ".join(hotwords) or None,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        parts = list(segments)
        return Transcript(
            "".join(s.text for s in parts).strip(),
            min((float(s.no_speech_prob) for s in parts), default=0.0),
        )


class WhisperCppBackend:
    """whisper.cpp through pywhispercpp (Metal on Apple Silicon)."""

    name = "whisper.cpp"

    def __init__(self, cfg: AsrConfig, model: Any = None) -> None:
        self.cfg = cfg
        if model is None:
            from pywhispercpp.model import Model

            cache = models_dir() / "whispercpp"
            cache.mkdir(parents=True, exist_ok=True)
            model = Model(cfg.model, models_dir=str(cache), print_progress=False)
        self.model = model

    def transcribe(self, audio: Audio) -> Transcript:
        """Recognize one utterance."""
        segments = self.model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.cfg.language or "auto",
            beam_search={"beam_size": self.cfg.beam_size, "patience": -1.0},
            initial_prompt=_initial_prompt(self.cfg.hotwords) or "",
            no_context=True,
        )
        return Transcript("".join(s.text for s in segments).strip())


def create_whisper(cfg: AsrConfig) -> AsrBackend:
    """Choose the engine: auto prefers CUDA, then Metal, then CPU."""
    device = cfg.device
    if device == "metal":
        return WhisperCppBackend(cfg)
    if device == "auto":
        if cuda_available():
            device = "cuda"
        elif is_apple_silicon():
            try:
                return WhisperCppBackend(cfg)
            except ImportError:
                print("pywhispercpp not installed; using CPU. Install silentdictate[metal].")
                device = "cpu"
        else:
            device = "cpu"
    return FasterWhisperBackend(cfg, device)
