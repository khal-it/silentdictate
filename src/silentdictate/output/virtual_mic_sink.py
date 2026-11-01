"""VirtualMicSink: speak recognized text into a virtual input device."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile
from scipy import signal

from ..audio_in import Audio, find_device
from ..tts import TtsBackend

LEAD_IN_MS = 300
TAIL_MS = 600
FALLBACK_RATE = 48000


def resample(audio: Audio, source: int, target: int) -> Audio:
    """Resample mono audio with a polyphase filter."""
    if source == target:
        return audio
    factor = gcd(source, target)
    return np.asarray(
        signal.resample_poly(audio, target // factor, source // factor), dtype=np.float32
    )


class Player(Protocol):
    """An output device."""

    rate: int

    def play(self, audio: Audio) -> None:
        """Play mono audio at ``rate`` and block until done."""
        ...


class SoundDevicePlayer:
    """A sounddevice output on a named device, at its native rate on all its channels."""

    def __init__(self, device: str, backend: Any = None) -> None:
        if backend is None:
            backend = importlib.import_module("sounddevice")
        self.backend = backend
        info = find_device(device, "output", backend)
        self.device = info.index
        self.rate = int(info.default_samplerate) or FALLBACK_RATE
        self.channels = info.max_output_channels

    def play(self, audio: Audio) -> None:
        """Play and wait."""
        self.backend.play(
            np.tile(audio[:, None], (1, self.channels)),
            samplerate=self.rate,
            device=self.device,
            blocking=True,
        )


@dataclass
class FakeDevice:
    """Records what would have been played, for loopback tests."""

    rate: int = FALLBACK_RATE
    played: list[Audio] = field(default_factory=list)

    def play(self, audio: Audio) -> None:
        """Store the audio."""
        self.played.append(audio)


class FileSink:
    """Writes each playback to a numbered WAV file."""

    def __init__(self, directory: Path, rate: int = FALLBACK_RATE) -> None:
        self.directory = directory
        self.rate = rate
        self.count = 0

    def play(self, audio: Audio) -> None:
        """Write the audio."""
        self.directory.mkdir(parents=True, exist_ok=True)
        soundfile.write(self.directory / f"{self.count:04d}.wav", audio, self.rate)
        self.count += 1


class PttAssist:
    """Holds or taps a key around playback for apps that need push-to-talk."""

    def __init__(self, key: str, mode: str = "hold", keyboard: Any = None) -> None:
        from pynput.keyboard import Controller, Key

        self.key: Any = getattr(Key, key, key)
        self.mode = mode
        self.keyboard = keyboard or Controller()

    def _tap(self) -> None:
        self.keyboard.press(self.key)
        self.keyboard.release(self.key)

    def begin(self) -> None:
        """Run before playback."""
        self.keyboard.press(self.key) if self.mode == "hold" else self._tap()

    def end(self) -> None:
        """Run after playback, even if it failed."""
        self.keyboard.release(self.key) if self.mode == "hold" else self._tap()


class VirtualMicSink:
    """Re-synthesizes text and plays it into a virtual mic."""

    def __init__(
        self,
        tts: TtsBackend,
        player: Player,
        ptt: PttAssist | None = None,
        monitor: Player | None = None,
        lead_in_ms: int = LEAD_IN_MS,
        tail_ms: int = TAIL_MS,
    ) -> None:
        self.tts = tts
        self.player = player
        self.ptt = ptt
        self.monitor = monitor
        self.lead_in_ms = lead_in_ms
        self.tail_ms = tail_ms

    def render(self, text: str) -> Audio:
        """Return speech at the device rate with lead-in and tail silence."""
        speech, rate = self.tts.synthesize(text)
        speech = resample(speech, rate, self.player.rate)
        lead = np.zeros(self.player.rate * self.lead_in_ms // 1000, dtype=np.float32)
        tail = np.zeros(self.player.rate * self.tail_ms // 1000, dtype=np.float32)
        return np.concatenate([lead, speech, tail])

    def deliver(self, text: str) -> None:
        """Speak text into the device."""
        audio = self.render(text)
        if self.ptt:
            self.ptt.begin()
        try:
            self.player.play(audio)
        finally:
            if self.ptt:
                self.ptt.end()
        if self.monitor:
            self.monitor.play(resample(audio, self.player.rate, self.monitor.rate))
