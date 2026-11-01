# Decisions

`SPEC.md` is the source of truth. This records the choices it left open, by milestone. Measured numbers come from an Apple Silicon Mac (10 cores, 24 GB) on 2026-11-01.

## Milestone 1: skeleton

- **Python 3.12, `<3.14`:** the spec prefers 3.12, and speech libraries lag new Python releases.
- **The mic stays open during `run`:** pre-roll needs audio from before the key goes down. Those blocks stay in a 500 ms in-memory ring and are never written to disk. It is only active while `run` is running, so it's not an always-on listening mode.
- **Added commands:** `download-models` (the spec requires a download command but doesn't name it) and `config-init`. All 12 spec commands are unchanged.
- **`--set-config key=value`:** the CLI override route, so overrides work for every setting and need no per-setting flags.

## Milestone 2: ASR

- **Silero through faster-whisper:** it bundles the ONNX model, so VAD needs neither torch nor a network connection.
- **`bench` defaults to `--device cpu`:** `auto` on Apple Silicon picks whisper.cpp, but the acceptance criterion names faster-whisper. For `bench --set fixtures` the model defaults to `small`.
- **onnx-asr for Parakeet:** it has a smaller API than sherpa-onnx and runs the NeMo exports directly. See [Verification status](#verification-status) for what was exercised.
- **Checksums use Hugging Face metadata:** each file is checked against the Hub's LFS SHA-256, or git blob SHA-1 for small files. The pinned revision is the one the metadata came from. A local manifest then lets `doctor` re-verify offline. This trusts the Hub at download time; nothing hard-coded protects against a compromised repo.

## Milestone 3: TypeSink

- **Newlines become spaces unless `allow_newlines = true`:** the focused window may be a terminal, where a newline is Enter, and the spec says never to send Enter unless `auto_submit = true`. So "new line" is inert by default, and users who want it opt in and accept that risk.
- **`TerminalBackend` writes to a pty master:** the terminal path for acceptance criterion 7, and for embedding a terminal app. `select_backends` doesn't choose it, because a desktop app has no pty master for the focused window.
- **Fallback resumes at the failed chunk:** a backend failing midway doesn't retype what was delivered. A chunk that was partly delivered before the failure can still repeat.
- **Text over 2,000 characters prefers paste:** this applies wherever a paste backend exists.

## Milestone 4: robustness

- **Guard order:** energy on raw audio, Silero on normalized audio, then ASR, then the phantom list. VAD must see normalized audio or faint speech would fail it.
- **`MIN_SPEECH_MS = 120`, threshold 0.3:** the guard stops every `silence/` clip before ASR. Spoken with the macOS Whisper voice, 12 sentences had no false rejections at −25 dB but 3 of 12 at −40 dB, with WER 46.3%. Very faint speech is therefore a known limit, and real recordings must set the final threshold (see `MANUAL_TESTS.md`).
- **Gain cap of 70 dB, up from 50:** a test showed 50 dB stopped a −60 dB peak reaching the −3 dBFS target.
- **"Thank you" and "you" alone are dropped:** they're classic Whisper phantoms, so genuinely dictating only those words produces nothing. Longer sentences containing them are unaffected.
- **"period", "colon" and similar only count at the end of a phrase:** so "a period of time" survives. "...end of the period." still becomes "...end of the.".
- **Volume feedback is a status message:** it compares each utterance with calibration, and it never blocks dictation.

## Milestone 5: fine-tuning

- **Plain PyTorch loop, batch size 1:** less code than `Seq2SeqTrainer`, and enrollment sets are small.
- **`task_type` is left unset in `LoraConfig`:** peft's seq2seq wrapper forwards `input_ids`, which Whisper rejects.
- **Every fifth recording is held out:** `bench --set personal` scores only those, so the A/B never uses training data.
- **Prompts were written for this project and are CC0:** this avoids licensing questions over TIMIT and the Harvard sentences. Corpora such as wTIMIT and CHAINS are local-only, and nothing downloads them.

## Milestone 6: virtual mic

- **`tts.speed` is not linear:** it sets Piper's `length_scale`, but `speed = 2` measured about 1.5× faster, because the voice model limits how far durations compress. Piper also randomizes durations per call, so fixtures differ slightly between generations.
- **Mono is copied to all device channels:** so an app reading either BlackHole channel hears it.
- **`FakeDevice` and `FileSink`:** the loopback runs real TTS, resampling, lead-in and tail through a fake device, then real ASR.

## Milestone 7: LLM cleanup

- **Loopback endpoints only:** any other host raises an error, so cleanup can't become a cloud feature by typo. `trust_env=False` stops proxy variables rerouting it.
- **Replies over 120% of the input's word count are discarded:** a model that adds content gets ignored and the original text is used. Errors do the same.

## Milestone 9: packaging

- **Tray is an optional extra:** terminal status is the default, and `tray = true` adds an icon.
- **Dictated text is never logged:** only state changes and errors are.
- **PyInstaller is documented, not automated:** native model libraries need per-OS tuning, which needs each OS to test on.

## Verification status

Optional paths aren't in the default `uv run pytest` environment, so these were run once for real against the generated fixtures (`normal/`), outside the test suite:

| Path | Result |
|---|---|
| Parakeet TDT v3 through onnx-asr, int8, CPU | 2.1% WER on 24 clips |
| whisper.cpp `tiny` on Metal | 3.1% WER on 6 clips |
| LoRA training on `tiny`, one epoch, 8 clips | Trained, exported to CTranslate2 and loaded in faster-whisper; 0.0% WER on 6 of those training clips, so this proves the path works, not that accuracy improved |

Those runs found two bugs that mocked tests had missed, both now fixed: `transformers` no longer writes `preprocessor_config.json` with the processor, and ONNX Runtime rejects the Hugging Face cache layout. Parakeet therefore downloads to the app data dir.

Not run for real: NeMo (needs CUDA), Kokoro (its model files aren't downloaded by `download-models`), the Windows and Linux injection backends, and the tray icon. These have mocked or no tests. See `MANUAL_TESTS.md` for everything that needs checking by hand.
