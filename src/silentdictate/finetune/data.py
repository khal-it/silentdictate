"""Training data: enrollment recordings and user-supplied local corpora."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import data_dir

HOLDOUT_EVERY = 5


@dataclass
class Sample:
    """One clip and its transcript."""

    path: Path
    text: str


def enroll_dir() -> Path:
    """Return the enrollment directory."""
    return data_dir() / "enroll"


def append_sample(sample: Sample) -> None:
    """Record a clip in the manifest."""
    with (enroll_dir() / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"file": sample.path.name, "text": sample.text}) + "\n")


def load_enrollment() -> list[Sample]:
    """Load enrollment samples, skipping any whose audio was removed."""
    manifest = enroll_dir() / "manifest.jsonl"
    if not manifest.exists():
        return []
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    samples = [Sample(enroll_dir() / r["file"], r["text"]) for r in rows]
    return [s for s in samples if s.path.exists()]


def scan_corpus(root: Path) -> list[Sample]:
    """Load a local corpus such as wTIMIT or CHAINS: each .wav beside a same-named .txt.

    Corpora are never downloaded; the user supplies the directory.
    """
    samples = []
    for wav in sorted(root.rglob("*.wav")):
        transcript = wav.with_suffix(".txt")
        if transcript.exists():
            samples.append(Sample(wav, transcript.read_text(encoding="utf-8").strip()))
    return samples


def split(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    """Return (train, test) with every fifth sample held out for the A/B benchmark."""
    test = samples[HOLDOUT_EVERY - 1 :: HOLDOUT_EVERY]
    train = [s for i, s in enumerate(samples) if (i + 1) % HOLDOUT_EVERY]
    return train, test
