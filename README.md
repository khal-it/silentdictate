# SilentDictate

Dictate by whispering into the mic you already have. SilentDictate recognizes your speech on your machine, then types the text into the focused window or speaks it into a virtual microphone.

Audio and text stay on your computer. The only network use is downloading models, and there is no telemetry.

**Mic position matters most.** Whisper 3–5 cm from a headset or earbud mic, or hold the built-in mic that close. Close-mic faint speech is what makes this work.

Status: everything automated passes on synthetic speech, but real whispers and most OS-specific paths still need the checks in [MANUAL_TESTS.md](MANUAL_TESTS.md).

## Install

You need [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13.

```bash
uv tool install .
silentdictate download-models
silentdictate doctor
```

From a checkout, use `uv sync` and prefix commands with `uv run`.

## Quickstart by OS

### macOS

```bash
silentdictate calibrate
silentdictate run
```

Hold **Right Option**, whisper and release. On first run, macOS asks for microphone, Accessibility and Input Monitoring access. Its dialogs open System Settings, where you turn on your terminal app and then restart it. `silentdictate doctor` shows the same dialogs and what's still missing. On Apple Silicon, `uv tool install '.[metal]'` adds whisper.cpp with Metal.

### Windows

```bash
silentdictate calibrate
silentdictate run
```

Hold **Right Ctrl**, whisper and release.

### Linux X11

Install `xdotool`, then run the same commands and hold **Right Ctrl**.

### Linux Wayland

Wayland apps can't register global hotkeys, so run [scripts/setup_wayland_injection.sh](scripts/setup_wayland_injection.sh) and bind `silentdictate ptt-start`, `ptt-stop` or `toggle` in your compositor.

## Which output should I use?

Use the default, `output_mode = "type"`, whenever you can. It's faster, and for Claude Code it's the better route: `/voice` streams audio to Anthropic's servers, so a virtual mic would mean recognizing your speech twice and adding playback delay.

Use `virtual_mic` for apps that only accept voice input. Typing never presses Enter unless you set `auto_submit = true`.

| OS | Setup |
|---|---|
| macOS | [scripts/setup_virtual_mic_macos.sh](scripts/setup_virtual_mic_macos.sh) |
| Windows | [scripts/setup_virtual_mic_windows.md](scripts/setup_virtual_mic_windows.md), with [scripts/detect_virtual_mic_windows.ps1](scripts/detect_virtual_mic_windows.ps1) to check it |
| Linux | [scripts/setup_virtual_mic_linux.sh](scripts/setup_virtual_mic_linux.sh), undone with [scripts/teardown_virtual_mic_linux.sh](scripts/teardown_virtual_mic_linux.sh) |

`silentdictate setup-virtual-mic` shows the right one for your OS.

## Commands

| Command | Purpose |
|---|---|
| `run` | Start dictating |
| `devices` | List audio devices |
| `calibrate` | Measure your whisper level for volume feedback |
| `enroll` | Record whispered prompts |
| `finetune` | LoRA fine-tune Whisper on your recordings (`--extra finetune`) |
| `bench` | Measure WER and latency, on `--set fixtures` or `--set personal` |
| `setup-virtual-mic` | Show or run the virtual mic setup |
| `doctor` | Check permissions, devices and models |
| `toggle`, `ptt-start`, `ptt-stop` | Control a running session |
| `purge-data` | Delete recordings and calibration; `--models` also deletes models |
| `download-models` | Download models and verify checksums |
| `config-init` | Write a config file |

## Configuration

Run `silentdictate config-init` to write a TOML config to your OS config dir. Any setting can be overridden per run:

```bash
silentdictate --set-config asr.model=small --set-config mode=tap run
```

The main settings are:

| Setting | Default | Notes |
|---|---|---|
| `hotkey` | `alt_r` on macOS, otherwise `ctrl_r` | |
| `mode` | `hold` | `tap` or `vad`; `vad` listens continuously while running |
| `output_mode` | `type` | `virtual_mic` or `both` |
| `auto_submit` | `false` | Presses Enter after the text |
| `allow_newlines` | `false` | Lets "new line" send a newline, which submits in terminals |
| `asr.backend` | `whisper` | `parakeet` needs `--extra parakeet` |
| `asr.model` | `large-v3-turbo` | Also tiny, base, small, large-v3, distil-large-v3.5 |
| `asr.hotwords` | `[]` | Your names and technical terms |
| `vad.threshold` | `0.3` | Lower it if real whispers come back empty |
| `postprocess.replacements` | `{}` | Such as `"pie test" = "pytest"` |
| `postprocess.llm_cleanup` | `false` | Uses a local Ollama or llama.cpp endpoint only |
| `privacy.save_audio` | `false` | Saves utterances under the app data dir |

Say "period", "comma", "question mark", "new line", "open paren" and similar for punctuation.

## Fine-tuning

```bash
uv sync --extra finetune
uv run silentdictate enroll
uv run silentdictate finetune --base-model small
uv run silentdictate bench --set personal --model small
```

Every fifth recording is held out, so the comparison never uses training data. Add a wTIMIT or CHAINS corpus you already have with `--corpus PATH`.

## Development

```bash
uv sync
uv run silentdictate download-models --whisper small
uv run python scripts/make_fixtures.py
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run silentdictate bench --set fixtures
```

[scripts/make_fixtures.py](scripts/make_fixtures.py) generates the test audio. Real-model tests load `small` and the Piper voice, and hardware tests are excluded by default.

For an optional standalone build: `uv run --extra pyinstaller pyinstaller --name silentdictate --collect-all faster_whisper --collect-all piper src/silentdictate/cli.py`.

## Project documents

- [SPEC.md](SPEC.md): the build specification
- [DECISIONS.md](DECISIONS.md): trade-offs and measured limits
- [CHANGELOG.md](CHANGELOG.md): release notes
- [MANUAL_TESTS.md](MANUAL_TESTS.md): checks that need real hardware

The webcam lip-reading plugin is an [interface stub](src/silentdictate/plugins/vsr_webcam/README.md), and [ultrasonic sensing is out of scope](src/silentdictate/plugins/sonar_experimental/README.md).
