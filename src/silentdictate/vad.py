"""Silero VAD (the ONNX model bundled with faster-whisper) and a streaming segmenter."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .audio_in import SAMPLE_RATE, Audio

FRAME = 512
WINDOW_FRAMES = 31  # about 1 s of context
_FRAME_MS = FRAME / SAMPLE_RATE * 1000


class VadModel(Protocol):
    """Anything mapping audio to per-frame speech probabilities."""

    def __call__(self, audio: Audio) -> np.ndarray[Any, Any]: ...


def silero_model() -> VadModel:
    """Load the bundled Silero model; no network or torch needed."""
    from faster_whisper.vad import get_vad_model

    model: VadModel = get_vad_model()
    return model


def speech_probs(audio: Audio, model: VadModel) -> np.ndarray[Any, Any]:
    """Return one speech probability per 512-sample frame."""
    usable = audio.size - audio.size % FRAME
    if usable == 0:
        return np.zeros(0)
    return np.asarray(model(audio[:usable])).reshape(-1)


def longest_speech_ms(audio: Audio, model: VadModel, threshold: float) -> float:
    """Return the longest run of speech frames in milliseconds."""
    best = run = 0
    for prob in speech_probs(audio, model):
        run = run + 1 if prob >= threshold else 0
        best = max(best, run)
    return best * _FRAME_MS


class VadSegmenter:
    """Cut a block stream into utterances for hands-free mode."""

    def __init__(
        self,
        model: VadModel,
        threshold: float = 0.3,
        preroll_ms: int = 500,
        min_silence_ms: int = 700,
        min_speech_ms: int = 200,
        max_utterance_s: float = 30.0,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.preroll = int(preroll_ms * SAMPLE_RATE / 1000)
        self.min_silence = max(1, round(min_silence_ms / _FRAME_MS))
        self.min_speech = max(1, round(min_speech_ms / _FRAME_MS))
        self.max_frames = int(max_utterance_s * 1000 / _FRAME_MS)
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._utterance: list[Audio] = []
        self._speech = self._silence = 0

    def _frame_prob(self, frame: Audio) -> float:
        # shortcut: re-scores a 1 s window per frame because the bundled model resets its RNN
        # state on each call; upgrade to a stateful session if CPU use matters.
        self._context = np.concatenate([self._context, frame])[-FRAME * WINDOW_FRAMES :]
        return float(speech_probs(self._context, self.model)[-1])

    def feed(self, block: Audio) -> list[Audio]:
        """Consume audio and return any finished utterances."""
        self._pending = np.concatenate([self._pending, block])
        finished: list[Audio] = []
        while self._pending.size >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            preroll = self._context[-self.preroll :]
            is_speech = self._frame_prob(frame) >= self.threshold
            if not self._utterance:
                if is_speech:
                    self._utterance = [preroll.copy(), frame]
                    self._speech, self._silence = 1, 0
                continue
            self._utterance.append(frame)
            self._speech += is_speech
            self._silence = 0 if is_speech else self._silence + 1
            if self._silence >= self.min_silence or len(self._utterance) >= self.max_frames:
                if self._speech >= self.min_speech:
                    finished.append(np.concatenate(self._utterance))
                self._utterance = []
        return finished
