"""Benchmarks: WER per fixture set plus p50/p95 latency per stage."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile

from .audio_in import Audio
from .config import Config
from .finetune.data import Sample, load_enrollment, split
from .finetune.train import finetuned_dir
from .output.virtual_mic_sink import resample
from .pipeline import Pipeline

STAGES = ("preprocess", "guard", "asr", "postprocess")
_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),  # type: ignore[no-untyped-call]
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def wer(references: list[str], hypotheses: list[str]) -> float:
    """Return word error rate ignoring case and punctuation; empty hypotheses count."""
    return float(
        jiwer.wer(
            references,
            hypotheses,
            reference_transform=_NORMALIZE,
            hypothesis_transform=_NORMALIZE,
        )
    )


@dataclass
class SetResult:
    """Results for one audio set."""

    name: str
    clips: int = 0
    wer: float | None = None
    output_chars: int = 0
    timings: dict[str, list[float]] = field(default_factory=dict)


def load_audio(path: Path) -> Audio:
    """Read a mono clip at 16 kHz."""
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return resample(audio[:, 0], int(rate), 16000)


def load_set(directory: Path) -> list[Sample]:
    """Load clips and optional transcripts.json from a fixture directory."""
    transcripts_file = directory / "transcripts.json"
    transcripts: dict[str, str] = {}
    if transcripts_file.exists():
        transcripts = json.loads(transcripts_file.read_text(encoding="utf-8"))
    return [Sample(p, transcripts.get(p.name, "")) for p in sorted(directory.glob("*.wav"))]


def run_set(name: str, samples: list[Sample], pipeline: Pipeline) -> SetResult:
    """Run a set; WER is computed only when transcripts exist."""
    result = SetResult(name, len(samples))
    outputs = []
    for sample in samples:
        output = pipeline.process(load_audio(sample.path))
        outputs.append(output.text)
        result.output_chars += len(output.text)
        for stage, ms in output.timings.items():
            result.timings.setdefault(stage, []).append(ms)
    references = [s.text for s in samples]
    if samples and all(references):
        result.wer = wer(references, outputs)
    return result


def format_results(results: list[SetResult]) -> str:
    """Render WER and latency tables."""
    lines = [f"{'set':<20}{'clips':>6}{'WER':>9}{'output chars':>14}", "-" * 49]
    for r in results:
        rate = f"{r.wer:.2%}" if r.wer is not None else "n/a"
        lines.append(f"{r.name:<20}{r.clips:>6}{rate:>9}{r.output_chars:>14}")
    lines += ["", f"{'set':<20}{'stage':<13}{'p50 ms':>10}{'p95 ms':>10}", "-" * 53]
    for r in results:
        for stage in STAGES:
            if r.timings.get(stage):
                p50, p95 = np.percentile(r.timings[stage], [50, 95])
                lines.append(f"{r.name:<20}{stage:<13}{p50:>10.1f}{p95:>10.1f}")
    return "\n".join(lines)


def bench_fixtures(root: Path, pipeline: Pipeline) -> list[SetResult]:
    """Run normal, quiet and silence."""
    results = []
    for name in ("normal", "quiet", "silence"):
        samples = load_set(root / name)
        if not samples:
            raise FileNotFoundError(f"no clips in {root / name}; run scripts/make_fixtures.py")
        results.append(run_set(name, samples, pipeline))
    return results


def bench_personal(cfg: Config, make_pipeline: Any) -> list[SetResult]:
    """A/B the base and fine-tuned models on the held-out enrollment split."""
    _, test = split(load_enrollment())
    if not test:
        raise FileNotFoundError("no enrollment recordings; run: silentdictate enroll")
    results = [run_set(f"base:{cfg.asr.model}", test, make_pipeline(cfg))]
    tuned = finetuned_dir()
    if tuned.exists():
        asr = dataclasses.replace(cfg.asr, model=str(tuned))
        results.append(
            run_set("fine-tuned", test, make_pipeline(dataclasses.replace(cfg, asr=asr)))
        )
    else:
        print("No fine-tuned model yet; run: silentdictate finetune")
    return results
