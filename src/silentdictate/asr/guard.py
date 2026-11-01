"""Hallucination guard: no speech in, no text out."""

from __future__ import annotations

import re

from ..audio_in import Audio, rms_db
from ..vad import VadModel, longest_speech_ms

MIN_RMS_DB = -80.0
MIN_SPEECH_MS = 120.0
MAX_NO_SPEECH_PROB = 0.8
KNOWN_HALLUCINATIONS = {
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "thank you so much for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "see you in the next video",
    "see you next time",
    "subtitles by the amara org community",
    "transcription by castingwords",
    "you",
    "bye",
    "bye bye",
}
_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|[♪♫]")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


class HallucinationGuard:
    """Rejects non-speech audio and known phantom transcriptions."""

    def __init__(self, vad: VadModel, threshold: float = 0.3) -> None:
        self.vad = vad
        self.threshold = threshold

    def has_speech(self, raw: Audio, processed: Audio) -> bool:
        """Check raw energy, then VAD on the normalized audio so faint speech still counts."""
        if rms_db(raw) < MIN_RMS_DB:
            return False
        return longest_speech_ms(processed, self.vad, self.threshold) >= MIN_SPEECH_MS

    def clean(self, text: str, no_speech_prob: float = 0.0) -> str:
        """Return the text with non-speech markers removed, or "" if it is a phantom."""
        if no_speech_prob > MAX_NO_SPEECH_PROB:
            return ""
        text = " ".join(_NON_SPEECH.sub(" ", text).split())
        normalized = _normalize(text)
        return "" if not normalized or normalized in KNOWN_HALLUCINATIONS else text
