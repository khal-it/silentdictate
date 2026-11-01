"""Local text-to-speech backends: Piper, Kokoro and OS-native voices."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile

from .audio_in import Audio
from .config import TtsConfig
from .models import ensure_piper, models_dir


class TtsBackend(Protocol):
    """Text to mono float32 audio."""

    def synthesize(self, text: str) -> tuple[Audio, int]:
        """Return samples and their rate."""
        ...


class PiperTts:
    """Piper neural TTS."""

    def __init__(self, cfg: TtsConfig, voice: Any = None) -> None:
        if voice is None:
            from piper import PiperVoice

            voice = PiperVoice.load(ensure_piper(cfg.voice))
        self.voice = voice
        self.speed = cfg.speed

    def synthesize(self, text: str) -> tuple[Audio, int]:
        """Return samples and their rate."""
        from piper import SynthesisConfig

        chunks = list(self.voice.synthesize(text, SynthesisConfig(length_scale=1 / self.speed)))
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return audio, int(chunks[0].sample_rate)


class KokoroTts:
    """Kokoro-82M through kokoro-onnx; model files go in the models dir."""

    def __init__(self, cfg: TtsConfig, model: Any = None) -> None:
        if model is None:
            from kokoro_onnx import Kokoro

            base = models_dir() / "kokoro"
            model = Kokoro(str(base / "kokoro-v1.0.onnx"), str(base / "voices-v1.0.bin"))
        self.model = model
        self.cfg = cfg

    def synthesize(self, text: str) -> tuple[Audio, int]:
        """Return samples and their rate."""
        voice = self.cfg.voice if "-" not in self.cfg.voice else "af_sarah"
        audio, rate = self.model.create(text, voice=voice, speed=self.cfg.speed, lang="en-us")
        return np.asarray(audio, dtype=np.float32), int(rate)


class OsTts:
    """macOS say, Windows SAPI (pyttsx3) or Linux espeak-ng."""

    def __init__(self, platform: str = sys.platform) -> None:
        self.platform = platform

    def synthesize(self, text: str) -> tuple[Audio, int]:
        """Render to a temp file, read it back and delete it."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech"
            if self.platform == "darwin":
                path = path.with_suffix(".aiff")
                subprocess.run(["say", "-o", str(path), "--", text], check=True)
            elif self.platform == "win32":
                import pyttsx3

                path = path.with_suffix(".wav")
                engine = pyttsx3.init()
                engine.save_to_file(text, str(path))
                engine.runAndWait()
            else:
                exe = shutil.which("espeak-ng")
                if exe is None:
                    raise RuntimeError("espeak-ng is not installed")
                path = path.with_suffix(".wav")
                subprocess.run([exe, "-w", str(path), "--", text], check=True)
            audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
        return audio[:, 0], int(rate)


def create_tts(cfg: TtsConfig) -> TtsBackend:
    """Build the configured TTS backend, falling back to OS voices if Piper cannot load."""
    if cfg.backend == "kokoro":
        return KokoroTts(cfg)
    if cfg.backend == "os":
        return OsTts()
    try:
        return PiperTts(cfg)
    except ImportError as exc:
        print(f"Piper is unavailable ({exc}); using the OS voice. Install silentdictate[piper].")
        return OsTts()
