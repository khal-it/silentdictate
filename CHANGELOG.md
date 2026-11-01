# Changelog

## 0.1.0 - 2026-11-01

First public release, covering spec milestones 1–7 and 9.

### Added

- Hold, tap and VAD recording, with a pre-roll buffer, level meter, calibration and volume feedback.
- Whisper through faster-whisper and whisper.cpp, and Parakeet through onnx-asr and NeMo.
- A hallucination guard, custom vocabulary, spoken punctuation, replacements and optional local LLM cleanup.
- TypeSink for macOS, Windows, X11 and the Wayland tool chain.
- VirtualMicSink with Piper, Kokoro and OS voices, PTT assist and monitoring.
- Enrollment, LoRA fine-tuning, CTranslate2 export and `bench --set personal` A/B testing.
- `doctor`, `bench`, `purge-data`, `download-models`, setup scripts and generated fixtures.

### Changed

- `run` and `doctor` now trigger the macOS permission dialogs themselves.
- Each utterance now reports its length, level, detected speech and outcome.
- `run` no longer needs Unix sockets, so it can start on Windows, where `toggle`, `ptt-start` and `ptt-stop` report that they're unavailable.
- PTT assist loads `pynput` only when used, and a failed utterance no longer stops the session.
- Piper moved to the optional `piper` extra because it's GPL-3.0-licensed.

### Not included

- The webcam VSR plugin is an interface stub (milestone 8).
- Ultrasonic sonar is out of scope.
