"""Audio capture, device selection, level metering and whisper-volume feedback."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import data_dir

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
FEEDBACK_MARGIN_DB = 8.0

Audio = np.ndarray[Any, np.dtype[np.float32]]


def _sd() -> Any:
    import sounddevice  # imported late: it raises at import time without PortAudio

    return sounddevice


@dataclass
class DeviceInfo:
    """One audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


def list_devices(backend: Any = None) -> list[DeviceInfo]:
    """Enumerate audio devices."""
    backend = backend or _sd()
    return [
        DeviceInfo(
            i,
            d["name"],
            d["max_input_channels"],
            d["max_output_channels"],
            d["default_samplerate"],
        )
        for i, d in enumerate(backend.query_devices())
    ]


def find_device(spec: str | int, kind: str = "input", backend: Any = None) -> DeviceInfo:
    """Select a device by index or case-insensitive name substring."""
    devices = list_devices(backend)
    usable = [
        d
        for d in devices
        if (d.max_input_channels if kind == "input" else d.max_output_channels) > 0
    ]
    text = str(spec)
    if text.isdigit():
        for device in usable:
            if device.index == int(text):
                return device
    else:
        matches = [d for d in usable if text.lower() in d.name.lower()]
        if matches:
            return matches[0]
    names = ", ".join(d.name for d in usable) or "none"
    raise LookupError(f"no {kind} device matching {spec!r}; available: {names}")


def rms_db(audio: np.ndarray[Any, Any]) -> float:
    """Return RMS level in dBFS, or -inf for empty or digital silence."""
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20 * float(np.log10(rms)) if rms > 0 else float("-inf")


def meter(level_db: float, width: int = 30, floor_db: float = -80.0) -> str:
    """Render a level as a text bar."""
    filled = int(np.clip((level_db - floor_db) / -floor_db, 0, 1) * width)
    return "#" * filled + "-" * (width - filled)


@dataclass
class Calibration:
    """Measured noise floor and whisper level."""

    noise_db: float
    whisper_db: float

    @classmethod
    def path(cls) -> Path:
        """Return the calibration file path."""
        return data_dir() / "calibration.json"

    def save(self) -> None:
        """Persist levels only; no audio is stored."""
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls) -> Calibration | None:
        """Load calibration if present."""
        path = cls.path()
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None


def volume_feedback(level_db: float, calibration: Calibration | None) -> str:
    """Compare a level with calibration and return ``too loud``, ``too quiet`` or ``ok``."""
    if calibration is None:
        return "ok"
    if level_db > calibration.whisper_db + FEEDBACK_MARGIN_DB:
        return "too loud"
    floor = max(calibration.noise_db + 3, calibration.whisper_db - FEEDBACK_MARGIN_DB)
    return "too quiet" if level_db < floor else "ok"


class Recorder:
    """Capture 16 kHz mono float32 with a pre-roll ring buffer.

    The stream stays open while dictation runs so pre-roll exists when the key goes down;
    those blocks stay in memory and age out after ``preroll_ms``.
    """

    def __init__(
        self,
        device: str | int | None = None,
        preroll_ms: int = 500,
        stream_factory: Callable[..., Any] | None = None,
        on_block: Callable[[Audio], None] | None = None,
    ) -> None:
        self.device = device or None
        self._stream_factory = stream_factory
        self._on_block = on_block
        blocks = max(1, round(preroll_ms / 1000 * SAMPLE_RATE / BLOCK_SIZE))
        self._ring: deque[Audio] = deque(maxlen=blocks)
        self._captured: list[Audio] = []
        self._recording = False
        self._lock = threading.Lock()
        self._stream: Any = None

    def _callback(self, indata: Any, _frames: int, _time: Any, status: Any) -> None:
        block: Audio = np.array(indata[:, 0], dtype=np.float32)
        with self._lock:
            (self._captured if self._recording else self._ring).append(block)
        if self._on_block:
            self._on_block(block)

    def open(self) -> None:
        """Open the input stream."""
        factory = self._stream_factory or _sd().InputStream
        self._stream = factory(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        """Close the input stream and drop buffered audio."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._ring.clear()
            self._captured = []
            self._recording = False

    def start(self) -> None:
        """Begin an utterance, keeping the pre-roll."""
        with self._lock:
            self._captured = list(self._ring)
            self._ring.clear()
            self._recording = True

    def stop(self) -> Audio:
        """End the utterance and return its samples."""
        with self._lock:
            self._recording = False
            captured, self._captured = self._captured, []
        return np.concatenate(captured) if captured else np.zeros(0, dtype=np.float32)

    def capture(self, seconds: float, sleep: Callable[[float], None]) -> Audio:
        """Record for a fixed time."""
        self.start()
        sleep(seconds)
        return self.stop()
