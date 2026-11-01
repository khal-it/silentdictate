"""Enrollment: record whispered readings with per-utterance level checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import soundfile

from ..audio_in import SAMPLE_RATE, Calibration, Recorder, rms_db, volume_feedback
from .data import Sample, append_sample, enroll_dir, load_enrollment
from .prompts import CODING_VOCABULARY, PROMPTS

MAX_TRIES = 3


def enroll(
    recorder: Recorder,
    prompts: Sequence[str] = PROMPTS,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> int:
    """Walk through prompts, resuming after those already recorded; return the new count."""
    directory = enroll_dir()
    directory.mkdir(parents=True, exist_ok=True)
    calibration = Calibration.load()
    done = {s.text for s in load_enrollment()}
    remaining = [p for p in prompts if p not in done]
    say(f"Audio is saved under {directory}. Remove it with: silentdictate purge-data")
    say(f"Coding words this helps with: {', '.join(CODING_VOCABULARY)}")
    saved = 0
    for prompt in remaining:
        say(f"\nWhisper 3-5 cm from the mic:\n  {prompt}")
        if ask("Enter to start, q to finish: ").strip().lower() == "q":
            break
        for attempt in range(MAX_TRIES):
            recorder.start()
            ask("Recording. Enter to stop: ")
            audio = recorder.stop()
            level = rms_db(audio)
            verdict = volume_feedback(level, calibration)
            if verdict == "ok" and audio.size:
                break
            say(f"{level:.0f} dBFS is {verdict or 'empty'}; try again.")
            if attempt == MAX_TRIES - 1:
                say("Keeping this take anyway.")
        if audio.size:
            path = directory / f"{len(done) + saved:04d}.wav"
            soundfile.write(path, audio, SAMPLE_RATE)
            append_sample(Sample(path, prompt))
            saved += 1
    say(f"Saved {saved} recordings.")
    return saved
