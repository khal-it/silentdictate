"""Shared fixtures. Real-model tests use models from the normal data dir."""

from __future__ import annotations

import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from silentdictate.asr.guard import HallucinationGuard
from silentdictate.asr.whisper import FasterWhisperBackend
from silentdictate.config import AsrConfig, PostprocessConfig, TtsConfig
from silentdictate.models import ensure_piper, ensure_whisper
from silentdictate.pipeline import Pipeline
from silentdictate.postprocess import DefaultPostProcessor
from silentdictate.tts import PiperTts
from silentdictate.vad import silero_model

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep tests away from the user's data and config."""
    monkeypatch.setenv("SILENTDICTATE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SILENTDICTATE_CONFIG", str(tmp_path / "config.toml"))
    # Relative, because tmp_path can exceed macOS's 104-byte AF_UNIX limit.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SILENTDICTATE_SOCKET", "c.sock")
    return tmp_path / "data"


@pytest.fixture(scope="session")
def small_model_path() -> Path:
    """The real faster-whisper small model."""
    return ensure_whisper("small")


@pytest.fixture(scope="session")
def voice_path() -> Path:
    """The real Piper voice."""
    return ensure_piper(TtsConfig().voice)


def make_asr(path: Path) -> FasterWhisperBackend:
    """Load the real model from a local path."""
    return FasterWhisperBackend(AsrConfig(model=str(path)), "cpu")


def make_pipeline(asr: Any, sinks: list[Any] | None = None) -> Pipeline:
    """Build a real pipeline."""
    return Pipeline(
        asr,
        HallucinationGuard(silero_model(), 0.3),
        DefaultPostProcessor(PostprocessConfig()),
        sinks or [],
    )


@pytest.fixture(scope="session")
def real_asr(small_model_path: Path) -> FasterWhisperBackend:
    """Shared real ASR."""
    return make_asr(small_model_path)


@pytest.fixture(scope="session")
def real_tts(voice_path: Path) -> PiperTts:
    """Shared real TTS."""
    from piper import PiperVoice

    return PiperTts(TtsConfig(), PiperVoice.load(voice_path))


@pytest.fixture(scope="session")
def fixtures_root(voice_path: Path) -> Path:
    """Generate audio fixtures on first use."""
    if not (FIXTURES / "silence").is_dir():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_fixtures.py"), "--out", str(FIXTURES)],
            check=True,
        )
    return FIXTURES


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any outbound connection; Unix sockets stay allowed."""
    real_connect = socket.socket.connect

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("outbound network use while blocked")

    def connect(self: socket.socket, address: Any) -> None:
        if self.family == socket.AF_UNIX:
            real_connect(self, address)
        else:
            refuse()

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


class FakeStream:
    """Stands in for a sounddevice InputStream."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def push(self, value: float, blocks: int = 1) -> None:
        for _ in range(blocks):
            data = np.full((512, 1), value, dtype=np.float32)
            self.kwargs["callback"](data, 512, None, None)


class FakeSoundDevice:
    """Stands in for the sounddevice module."""

    def __init__(self) -> None:
        self.played: list[dict[str, Any]] = []

    def query_devices(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "Built-in Mic",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
            },
            {
                "name": "BlackHole 2ch",
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]

    def play(self, data: Any, **kwargs: Any) -> None:
        self.played.append({"data": data, **kwargs})


@pytest.fixture
def fake_sd() -> FakeSoundDevice:
    """A fake sounddevice module."""
    return FakeSoundDevice()


class StubVad:
    """Reports speech wherever amplitude exceeds a level."""

    def __call__(self, audio: Any) -> Any:
        frames = np.asarray(audio).reshape(-1, 512)
        return (np.abs(frames).max(axis=1) > 0.01).astype(float)


@pytest.fixture
def stub_vad() -> StubVad:
    """A deterministic VAD."""
    return StubVad()
