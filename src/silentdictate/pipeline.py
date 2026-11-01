"""The dictation pipeline from captured audio to delivered text."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from .asr.base import AsrBackend
from .asr.guard import HallucinationGuard
from .audio_in import Audio
from .postprocess import PostProcessor
from .preprocess import preprocess


class Sink(Protocol):
    """A pluggable text destination."""

    def deliver(self, text: str) -> None:
        """Send text onward."""
        ...


@dataclass
class Result:
    """Text produced from one utterance with per-stage milliseconds."""

    text: str = ""
    timings: dict[str, float] = field(default_factory=dict)


class Pipeline:
    """Preprocess, guard, recognize, post-process and deliver."""

    def __init__(
        self,
        asr: AsrBackend,
        guard: HallucinationGuard,
        postprocessor: PostProcessor,
        sinks: Iterable[Sink] = (),
    ) -> None:
        self.asr = asr
        self.guard = guard
        self.postprocessor = postprocessor
        self.sinks = list(sinks)

    def process(self, audio: Audio) -> Result:
        """Run one utterance; output is empty if the guard rejects it."""
        result = Result()

        def timed(stage: str, start: float) -> float:
            now = time.perf_counter()
            result.timings[stage] = (now - start) * 1000
            return now

        mark = time.perf_counter()
        processed = preprocess(audio)
        mark = timed("preprocess", mark)
        speech = self.guard.has_speech(audio, processed)
        mark = timed("guard", mark)
        if not speech:
            return result
        transcript = self.asr.transcribe(processed)
        mark = timed("asr", mark)
        text = self.guard.clean(transcript.text, transcript.no_speech_prob)
        result.text = self.postprocessor.process(text) if text else ""
        mark = timed("postprocess", mark)
        if result.text:
            for sink in self.sinks:
                sink.deliver(result.text)
            timed("output", mark)
        return result
