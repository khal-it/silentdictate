#!/usr/bin/env python3
"""Generate test audio under tests/fixtures: normal, quiet and silence sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile

from silentdictate.config import TtsConfig
from silentdictate.finetune.prompts import SENTENCES
from silentdictate.output.virtual_mic_sink import resample
from silentdictate.tts import create_tts

RATE = 16000
SENTENCE_COUNT = 24
ATTENUATION_DB = 25.0
NOISE_DBFS = -65.0
CLIPS_PER_KIND = 10
CLIP_SECONDS = 2.0


def room_tone(rng: np.random.Generator, samples: int, level_db: float) -> np.ndarray:
    """Pink-ish noise with mains hum."""
    white = rng.standard_normal(samples)
    spectrum = np.fft.rfft(white) / np.sqrt(np.arange(1, samples // 2 + 2))
    pink = np.fft.irfft(spectrum, samples)
    pink /= np.sqrt(np.mean(pink**2))
    hum = 0.2 * np.sin(2 * np.pi * 50 * np.arange(samples) / RATE)
    return (pink + hum) * 10 ** (level_db / 20)


def keyboard_clicks(rng: np.random.Generator, samples: int) -> np.ndarray:
    """Short decaying bursts over faint room tone."""
    audio = room_tone(rng, samples, -70)
    click = int(0.012 * RATE)
    envelope = np.exp(-np.arange(click) / (click / 5))
    for _ in range(rng.integers(4, 12)):
        start = rng.integers(0, samples - click)
        audio[start : start + click] += (
            rng.uniform(0.02, 0.2) * rng.standard_normal(click) * envelope
        )
    return audio


def write_clip(path: Path, audio: np.ndarray) -> None:
    """Write 16-bit PCM, clipping safely."""
    soundfile.write(path, np.clip(audio, -1, 1), RATE, subtype="PCM_16")


def make(root: Path) -> None:
    """Generate every set. The seed fixes the noise; Piper speech still varies per run."""
    rng = np.random.default_rng(1234)
    tts = create_tts(TtsConfig())
    normal, quiet, silence = root / "normal", root / "quiet", root / "silence"
    for directory in (normal, quiet, silence):
        directory.mkdir(parents=True, exist_ok=True)
    transcripts = {}
    for i, sentence in enumerate(SENTENCES[:SENTENCE_COUNT]):
        name = f"{i:03d}.wav"
        speech, rate = tts.synthesize(sentence)
        speech = resample(speech, rate, RATE)
        transcripts[name] = sentence
        write_clip(normal / name, speech)
        faint = speech * 10 ** (-ATTENUATION_DB / 20)
        write_clip(quiet / name, faint + room_tone(rng, faint.size, NOISE_DBFS))
    for directory in (normal, quiet):
        (directory / "transcripts.json").write_text(json.dumps(transcripts, indent=1))
    samples = int(CLIP_SECONDS * RATE)
    for i in range(CLIPS_PER_KIND):
        write_clip(silence / f"digital_{i:02d}.wav", np.zeros(samples))
        level = rng.uniform(-75, -45)
        write_clip(silence / f"room_{i:02d}.wav", room_tone(rng, samples, level))
        write_clip(silence / f"keys_{i:02d}.wav", keyboard_clicks(rng, samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    make(parser.parse_args().out)
