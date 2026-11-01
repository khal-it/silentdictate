# Roadmap

The goal is the best private, free way to dictate without being overheard. Priorities follow real results, so this will change.

## Now: prove it on real whispers

- Publish real-whisper accuracy and latency from `bench --set personal`
- Verify Windows, X11, Wayland and virtual-mic behavior from `MANUAL_TESTS.md`
- Pick the default model and backend from that data
- Publish to PyPI so `uvx silentdictate` works

## Next: make it pleasant

- Multilingual support: better language defaults, and spoken punctuation beyond English and German
- Live partial results while you speak
- Per-app profiles, such as different vocabulary for a terminal and an email client
- A guided first-run setup
- Hotword and replacement management without editing TOML

## Later: make it bigger

- A native tray or menu bar app with signed installers, plus Homebrew and winget packages
- A community benchmark of anonymous WER results, strictly opt-in and with no audio
- Shared vocabulary packs for fields like medicine, law and programming
- Better whisper models, fine-tuned on openly licensed whispered speech
- A webcam lip-reading plugin as an optional fallback (milestone 8 in the spec)

## Out of scope

Cloud recognition by default, telemetry, always-on listening without an explicit mode switch, and ultrasonic sensing.
