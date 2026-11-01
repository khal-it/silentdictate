# Contributing

Thanks for helping. The most valuable contributions right now don't all involve code:

- **Real-whisper results:** the automated numbers come from synthetic speech, so real measurements matter most.
- **Platform testing:** work through [MANUAL_TESTS.md](MANUAL_TESTS.md) on your OS and report what happens.
- **Languages:** spoken punctuation and phantom-text filters cover English and German so far.

Issues labeled `good first issue` are a good place to start.

## Set up

You need [uv](https://docs.astral.sh/uv/). On Linux, also install PortAudio (`libportaudio2`).

```bash
uv sync
uv run silentdictate download-models --whisper small
uv run python scripts/make_fixtures.py
```

## Before opening a pull request

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Then:

- Keep each pull request to one change, and add a test for new behavior.
- Mock audio devices and injection backends in tests. Mark anything that needs real hardware with `@pytest.mark.hardware`.
- Add OS-specific checks you couldn't run to `MANUAL_TESTS.md`.
- Log trade-offs in `DECISIONS.md` and user-facing changes in `CHANGELOG.md`.
- Don't weaken or delete a test to make it pass.

Piper randomizes speech timing, so tests on generated speech shouldn't depend on one short sample.

## Project rules

These come from [SPEC.md](SPEC.md), and changes need a good reason:

- **Local first:** audio and text stay on the machine by default, and there's no telemetry.
- **No raw audio on disk** except in explicit enrollment, benchmark or `privacy.save_audio` modes.
- **Dictated text is never logged.**
- **No extra hardware**, and no ultrasonic sensing.
- **Never send Enter** unless the user set `auto_submit`.
- **Keep the core MIT-compatible.** GPL dependencies must stay optional extras.

By contributing, you agree that your work is licensed under the MIT License. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
