# Manual tests

These need real hardware, real whispers or an OS that wasn't available during the build. None of them have been run yet.

## 1. Whisper accuracy (do this first)

The automated numbers come from synthetic speech, not your voice. The guard's settings in particular haven't met a real whisper.

- [ ] `silentdictate download-models --whisper large-v3-turbo`
- [ ] `silentdictate calibrate`, with the mic 3–5 cm from your mouth
- [ ] `silentdictate enroll`, recording at least 100 sentences (they're saved under the app data dir)
- [ ] `silentdictate bench --set personal --model large-v3-turbo`: WER ≤ 10%. Repeat with `large-v3` and `distil-large-v3.5` to find the best base model
- [ ] If clips come back empty, lower `vad.threshold` (for example `--set-config vad.threshold=0.15`), then confirm `bench --set fixtures` still outputs 0 characters on `silence`
- [ ] Install the extra with `uv sync --extra finetune`, then run `silentdictate finetune --base-model small`
- [ ] `silentdictate bench --set personal --model small`: fine-tuned WER ≤ 6%
- [ ] Install `parakeet`, then run the same bench with `--set-config asr.backend=parakeet`, and choose the default backend from this data
- [ ] `silentdictate purge-data` removes the recordings

## 2. Latency

- [ ] Key release to injected text, p50: under 300 ms on a CUDA GPU
- [ ] Key release to injected text, p50: under 1.5 s on CPU or Apple Silicon
- [ ] TTS first audio with Piper: under 150 ms

## 3. macOS

- [ ] `silentdictate doctor` reports the mic, Accessibility and Input Monitoring correctly
- [ ] With Accessibility off, dictation prints the exact fix steps
- [ ] Right-Option hold records and types into TextEdit, a browser and Terminal
- [ ] Non-ASCII text and emoji arrive intact
- [ ] Text over 2,000 characters pastes, and your previous clipboard text returns
- [ ] `uv sync --extra metal`, then `--set-config asr.device=metal` runs through whisper.cpp

## 4. Windows

- [ ] Right-Ctrl hold types into Notepad and Windows Terminal, including emoji
- [ ] Typing into an elevated window fails with a clear message
- [ ] Long text pastes and your clipboard text returns

## 5. Linux

- [ ] X11: `xdotool` types into a terminal and a browser
- [ ] Sway: `wtype` types, and `toggle`, `ptt-start` and `ptt-stop` work through compositor bindings
- [ ] KDE: injection works through the available chain
- [ ] GNOME: `wtype` fails and falls through to `ydotool`, then to `wl-copy`
- [ ] `scripts/setup_wayland_injection.sh` works, and `doctor` reports `/dev/uinput` correctly

## 6. Virtual mic

- [ ] macOS with BlackHole: a recording app hears the speech
- [ ] Windows with VB-CABLE: an app on "CABLE Output" hears the speech
- [ ] Linux with PipeWire: an app on "SilentDictate_Mic" hears the speech, and `--persist` survives a restart
- [ ] `scripts/teardown_virtual_mic_linux.sh` removes everything
- [ ] Claude Code `/voice` with the virtual mic as the default input, using PTT assist if needed
- [ ] Monitoring through `monitor_device` plays to headphones

## 7. Other

- [ ] VAD mode in a real room: speech is segmented, and typing or room noise produces no text
- [ ] `silentdictate run` with `tray = true` and `uv sync --extra tray` shows the status icon
- [ ] `uv tool install .` gives a working `silentdictate` command
- [ ] Optional PyInstaller build starts and finds its models
- [ ] LLM cleanup with Ollama running: punctuation improves and no content is added
