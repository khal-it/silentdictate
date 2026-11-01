# SilentDictate — Build Specification (source of truth)

SilentDictate is a cross-platform (macOS, Windows, Linux X11/Wayland) desktop tool. It lets me dictate by WHISPERING very quietly (near-silent, inaudible beyond ~30 cm) into my EXISTING microphone (headset/earbud mic or built-in mic held close). It recognizes the speech locally and delivers the text to whatever app I'm using via:

- **(A) TypeSink (DEFAULT):** keystroke/text injection into the focused window.
- **(B) VirtualMicSink:** the recognized text is re-synthesized with local TTS and played into a virtual audio input device, so any voice-input app can consume it as normal speech.

A webcam lip-reading module is an OPTIONAL, DISABLED-BY-DEFAULT last-resort plugin.

## Why this approach (do not change it)

- Faint-whisper ASR with a close mic is the only no-extra-hardware method with strong evidence for open-vocabulary dictation. SilentWhisper (CHI 2025) reached 97.7% word accuracy on 454 words, with no per-user training, using a headset mic at 30–50 mm. Whisper large-v3 scores 8.2% WER on US-accented whispered wTIMIT, and fine-tuning on whispers cuts the error substantially.
- Ultrasonic "sonar" with a laptop speaker and mic has no laptop demonstration and only supports about 20–54 commands on phones at 2–4 cm. It is OUT OF SCOPE.
- Webcam visual speech recognition (Auto-AVSR, ~20% WER on LRS3) is the only truly silent free-text fallback. It is an optional plugin only.
- For Claude Code, TypeSink is preferred: Claude Code's `/voice` streams audio to Anthropic's servers, so the virtual-mic route means double recognition plus playback latency.

## Hard constraints

- NO extra hardware. Use only the built-in mic and speaker, existing headset/earbuds, and (plugin only) the built-in webcam.
- Local/offline by default. No audio or text leaves the machine unless an optional cloud feature is explicitly enabled. No telemetry. Model downloads are the only default network use, and they happen only via an explicit command or on first run with a clear message.
- Never write raw audio to disk except in explicit enrollment/benchmark modes. Store it under the app data dir, deletable via `purge-data`.
- Python 3.11+ (3.12 preferred). Use the `src/` layout and manage with uv, with a committed lockfile. Verify current library versions and APIs from official docs and PyPI at build time.
- Pluggable backends for input, ASR, post-processing and output. Type hints, docstrings, ruff and mypy clean, pytest.
- Do not weaken, skip or delete tests to make them pass. Hardware-only tests must be marked `@pytest.mark.hardware` and excluded by default. They do not count toward "tests pass".

## Architecture

```
hotkey/VAD controller
  -> audio capture (sounddevice, 16 kHz mono float32, ring buffer with 500 ms pre-roll)
  -> preprocessing (high-pass ~90 Hz, gain normalization to ~-3 dBFS peak; never remove whispers)
  -> ASR backend (pluggable)
  -> post-processing (replacements, spoken punctuation, OPTIONAL local LLM cleanup, off by default)
  -> output sinks: TypeSink and/or VirtualMicSink
  -> status indicator + log
```

## Modules (package `silentdictate`)

- **config.py:** TOML config in the OS config dir (platformdirs), with CLI overrides.
- **audio_in.py:** device enumeration and selection by name or index; level meter; whisper-volume feedback (too loud / too quiet) relative to a calibration step.
- **controller.py:** push-to-talk (hold) and toggle (tap) modes via a global hotkey (pynput on macOS, Windows and X11). On Wayland, expose `silentdictate toggle | ptt-start | ptt-stop` over a local Unix socket so the user binds them in their compositor. Optional Silero VAD mode with a lowered threshold, pre-roll and min-silence settings.
- **asr/base.py** plus backends:
  - **asr/whisper.py:** faster-whisper (CTranslate2) on CUDA/CPU; whisper.cpp (pywhispercpp or subprocess) on Apple Silicon. Models: tiny/base/small for tests; large-v3, large-v3-turbo and distil-large-v3.5 for use. Configurable beam size and language; initial_prompt/hotwords for custom vocabulary.
  - **asr/parakeet.py:** nvidia/parakeet-tdt-0.6b-v3 (or -v2 for English only). Use NeMo on CUDA and an ONNX runtime (sherpa-onnx or onnx-asr, whichever is better maintained) on CPU/Apple Silicon.
  - **Hallucination guard:** reject output when energy/VAD indicates no speech, and drop known hallucination strings.
- **finetune/:** enrollment plus a LoRA fine-tuning pipeline for Whisper (transformers + peft), exporting the merged model to CTranslate2. The enrollment flow shows bundled CC0/public-domain, phonetically diverse prompts plus a coding-vocabulary list, and records whispered readings with per-utterance level checks. Support local wTIMIT/CHAINS corpora if the user provides them; never auto-download datasets with unclear licenses.
- **postprocess.py:** replacements dictionary; spoken punctuation ("new line", "period"); optional LLM cleanup through a local OpenAI-compatible endpoint (Ollama/llama.cpp), off by default. Its strict prompt fixes punctuation and obvious ASR errors only and never adds content.
- **output/type_sink.py:**
  - Windows: SendInput Unicode, with a clipboard-paste fallback for long text.
  - macOS: Quartz unicode typing. Detect a missing Accessibility permission and print exact fix steps. Clipboard + Cmd+V fallback that restores the previous clipboard.
  - Linux X11: `xdotool type --delay N`.
  - Linux Wayland: detect `XDG_SESSION_TYPE`. Try `wtype` (Sway/Hyprland, not GNOME), then `ydotool` (needs ydotoold and /dev/uinput permission), then `wl-copy` + paste.
  - Terminal safety: never send Enter unless `auto_submit=true`; chunk long text; configurable per-character delay (default 2–5 ms); strip control characters.
- **output/virtual_mic_sink.py:**
  - TTS backends: Piper (default), Kokoro-82M (optional), and OS-native voices as a fallback (macOS `say`, Windows SAPI via pyttsx3, Linux espeak-ng).
  - Resample to the device's native rate (usually 48 kHz) and play via a sounddevice output stream on the configured device: macOS "BlackHole 2ch", Windows "CABLE Input (VB-Audio Virtual Cable)", Linux the null sink from the setup script.
  - Optional "PTT assist" that holds or taps a configurable key around playback, for apps that need it (e.g. Claude Code `/voice`). Lead-in and tail silence default to 300 ms and 600 ms.
  - Optional monitoring to headphones, off by default.
  - A `FileSink`/fake-device mode is used for automated loopback tests when no virtual device exists.
- **plugins/vsr_webcam/** (optional extra `silentdictate[vsr]`, disabled by default): define the plugin interface and a documented stub. Full implementation (MediaPipe mouth ROI → Auto-AVSR visual-only, LLM cleanup, camera-active indicator, no video saved outside enrollment) is a later milestone and NOT required for completion.
- **plugins/sonar_experimental/:** README stub only, explaining why it is out of scope.
- **cli.py:** `run | devices | calibrate | enroll | finetune | bench | setup-virtual-mic | doctor | toggle | ptt-start | ptt-stop | purge-data`.
- **doctor:** checks mic permission, macOS Accessibility/Input Monitoring, Wayland uinput, virtual device presence, model files and GPU availability, and prints fixes.

## Configuration (TOML defaults)

- `hotkey`: Right-Ctrl hold (Windows/Linux X11), Right-Option hold (macOS). `mode = hold|tap|vad`.
- `input_device`, `output_mode = type|virtual_mic|both`, `virtual_mic_device`, `auto_submit = false`.
- `asr.backend = whisper|parakeet`, `asr.model`, `asr.device = auto|cuda|cpu|metal`, `asr.language`, `asr.beam_size`, `asr.hotwords`.
- `tts.backend = piper|kokoro|os`, `tts.voice`, `tts.speed`.
- `vad.threshold`, `vad.preroll_ms`, `vad.min_silence_ms`.
- `postprocess.llm_cleanup = false`, `postprocess.replacements`.
- `privacy.save_audio = false`.

## Setup artifacts

- **README.md:** per-OS quickstart, including the recommendation to whisper 3–5 cm from a headset/earbud mic.
- **scripts/setup_virtual_mic_macos.sh:** `brew install --cask blackhole-2ch`, a reboot notice, and instructions for selecting BlackHole as the input or system default input. Apps like Claude Code `/voice` only use the default device.
- **scripts/setup_virtual_mic_windows.md** plus a `.ps1` device-detection helper: install VB-CABLE as admin and reboot. The app outputs to "CABLE Input"; other apps select "CABLE Output".
- **scripts/setup_virtual_mic_linux.sh:**
  - `pactl load-module module-null-sink media.class=Audio/Sink sink_name=silentdict-sink channel_map=mono`
  - `pactl load-module module-remap-source master=silentdict-sink.monitor source_name=silentdict-mic source_properties=device.description=SilentDictate_Mic`
  - An optional persistent PipeWire conf snippet and a teardown script.
- **scripts/setup_wayland_injection.sh:** wtype/ydotool install, ydotoold service, and a udev rule/group for /dev/uinput, with warnings.
- A model download command with checksums, cached under the app data dir.
- **DECISIONS.md, CHANGELOG.md.**
- **MANUAL_TESTS.md:** a step-by-step checklist for everything that cannot be verified in the build environment (see below).

## Test fixtures (generated, not downloaded)

`scripts/make_fixtures.py` generates audio under `tests/fixtures/`:

- `normal/`: at least 20 reference sentences synthesized with Piper (or espeak-ng if Piper is unavailable), with transcripts.
- `quiet/`: the same sentences attenuated by about 25 dB, with low-level noise added. This is a rough proxy for faint speech, not a substitute for real whispers.
- `silence/`: at least 30 clips of digital silence, room-tone-like noise and keyboard-like clicks.

## Milestones

1. Skeleton, config, device listing, push-to-talk recording, level meter, `doctor`.
2. ASR backends behind one interface; `bench` command.
3. TypeSink on all platforms, including the Wayland chain and macOS permission detection. First usable release.
4. Whisper robustness: calibration, gain, hallucination guard, custom vocabulary, spoken punctuation.
5. Enrollment, LoRA fine-tuning and export; `bench --set personal` A/B before and after.
6. VirtualMicSink, setup scripts, PTT assist; loopback verification.
7. Optional local LLM cleanup.
8. (Not required) Full webcam VSR plugin.
9. Packaging (`uv tool install`; optional PyInstaller), docs, tray/status indicator.

## Automated acceptance criteria (must be demonstrated in the build environment)

1. `uv sync` succeeds and `uv run silentdictate --help` lists every CLI command above.
2. `uv run pytest -q` (hardware tests excluded) exits 0. It covers every module with mocked audio devices and injection backends.
3. `uv run ruff check .` and `uv run mypy src` exit 0.
4. `uv run silentdictate bench --set fixtures` runs a real faster-whisper model (small or larger) on the generated fixtures and prints a WER table (via jiwer) per fixture set, plus p50/p95 latency per stage. Requirements: WER ≤ 15% on `normal/`, and the `quiet/` result is reported.
5. Virtual-mic loopback test: TTS of the 20 reference sentences, sent through VirtualMicSink to the fake device, transcribed by our ASR, gives WER ≤ 5%.
6. Hallucination/false-trigger tests: all `silence/` clips, and VAD mode on generated noise, produce zero characters of output.
7. Injection test: TypeSink, via its terminal path driven against a pty-based harness, delivers exact Unicode text (punctuation and non-ASCII included) with no trailing newline when `auto_submit=false`.
8. Offline test: core pipeline tests pass with outbound sockets blocked (monkeypatched), after models are cached.
9. All setup artifacts above exist and are linked from the README.

## Manual acceptance criteria (in MANUAL_TESTS.md; not verifiable in the build environment)

- On my real whispered recordings (at least 100 sentences via `enroll`), `bench --set personal`: WER ≤ 10% with the best base model, ≤ 6% after fine-tuning. Choose the default backend (Whisper vs Parakeet) from this data.
- Latency from key release to text injected: p50 < 300 ms on a CUDA GPU, < 1.5 s on CPU/Apple Silicon; TTS first audio < 150 ms with Piper.
- Real virtual-mic checks on macOS (BlackHole), Windows (VB-CABLE) and Linux (PipeWire), including Claude Code `/voice` with the virtual mic as the default input.
- macOS Accessibility/Input Monitoring prompts; Wayland on GNOME, KDE and Sway.

## Non-goals

No cloud ASR/TTS by default. No ultrasonic sensing. No always-on listening without an explicit mode switch. No transcript storage unless history is enabled.

## Working style

Before each milestone, state the plan and any trade-off decisions, and log them in DECISIONS.md. When something is OS-specific and untestable here, write it defensively and add it to MANUAL_TESTS.md.
