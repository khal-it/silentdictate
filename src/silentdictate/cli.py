"""Command line interface."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .audio_in import Calibration, Recorder, list_devices, meter, rms_db
from .config import Config, ConfigError, config_path, data_dir, load_config, save_config

PURGE_TARGETS = ("enroll", "recordings", "calibration.json", "finetune-merged")


def cmd_run(cfg: Config, _args: argparse.Namespace) -> int:
    """Start dictation."""
    from .app import run

    run(cfg)
    return 0


def cmd_devices(_cfg: Config, _args: argparse.Namespace) -> int:
    """List audio devices."""
    for d in list_devices():
        print(
            f"{d.index:>3}  in:{d.max_input_channels} out:{d.max_output_channels} "
            f"{d.default_samplerate:>7.0f} Hz  {d.name}"
        )
    return 0


def cmd_calibrate(cfg: Config, _args: argparse.Namespace) -> int:
    """Measure noise floor and whisper level; nothing but the levels is stored."""
    recorder = Recorder(cfg.input_device)
    recorder.open()
    try:
        input("Stay silent for 3 seconds. Press Enter to start: ")
        noise = rms_db(recorder.capture(3, time.sleep))
        input("Whisper normally for 5 seconds. Press Enter to start: ")
        whisper = rms_db(recorder.capture(5, time.sleep))
    finally:
        recorder.close()
    print(f"noise   {noise:6.1f} dBFS {meter(noise)}")
    print(f"whisper {whisper:6.1f} dBFS {meter(whisper)}")
    if whisper < noise + 3:
        print("Your whisper barely clears the room noise. Move the mic to 3-5 cm and retry.")
        return 1
    Calibration(noise, whisper).save()
    print("Calibration saved.")
    return 0


def cmd_enroll(cfg: Config, _args: argparse.Namespace) -> int:
    """Record whispered prompts."""
    from .finetune.enroll import enroll

    recorder = Recorder(cfg.input_device)
    recorder.open()
    try:
        enroll(recorder)
    finally:
        recorder.close()
    return 0


def cmd_finetune(cfg: Config, args: argparse.Namespace) -> int:
    """Fine-tune Whisper on enrollment data plus an optional local corpus."""
    from .finetune.data import load_enrollment, scan_corpus, split
    from .finetune.train import TrainConfig, export_ct2, train

    samples, _test = split(load_enrollment())
    if args.corpus:
        samples += scan_corpus(Path(args.corpus))
    model, processor = train(
        samples,
        TrainConfig(base_model=args.base_model, language=cfg.asr.language, epochs=args.epochs),
    )
    print(
        f"Exported to {export_ct2(model, processor)}. Compare: silentdictate bench --set personal"
    )
    return 0


def cmd_bench(cfg: Config, args: argparse.Namespace) -> int:
    """Benchmark a fixture set."""
    from .app import build_pipeline
    from .bench import bench_fixtures, bench_personal, format_results

    if args.model:
        cfg.asr.model = args.model
    cfg.asr.device = args.device
    # Benches measure recognition only, so nothing may be typed into the focused window.
    if args.set == "fixtures":
        results = bench_fixtures(Path(args.fixtures_dir), build_pipeline(cfg, sinks=[]))
    else:
        results = bench_personal(cfg, lambda c: build_pipeline(c, sinks=[]))
    print(f"model={cfg.asr.model} device={cfg.asr.device} beam={cfg.asr.beam_size}\n")
    print(format_results(results))
    return 0


def _scripts_dir() -> Path:
    packaged = Path(__file__).parent / "_scripts"
    return packaged if packaged.is_dir() else Path(__file__).parents[2] / "scripts"


def cmd_setup_virtual_mic(_cfg: Config, args: argparse.Namespace) -> int:
    """Show or run this OS's virtual mic setup."""
    scripts = _scripts_dir()
    if sys.platform == "win32":
        print(f"Follow {scripts / 'setup_virtual_mic_windows.md'}")
        return 0
    name = "macos" if sys.platform == "darwin" else "linux"
    script = scripts / f"setup_virtual_mic_{name}.sh"
    if not args.run:
        print(f"Review, then run: bash '{script}'\nOr: silentdictate setup-virtual-mic --run")
        return 0
    return subprocess.run(["bash", str(script)], check=False).returncode


def cmd_doctor(cfg: Config, _args: argparse.Namespace) -> int:
    """Check the environment."""
    from .app import permission_message
    from .doctor import report, request_macos_permissions, run_checks

    failures = report(run_checks(cfg))
    if sys.platform == "darwin" and (missing := request_macos_permissions()):
        print(permission_message(missing))
    return 1 if failures else 0


def cmd_control(_cfg: Config, args: argparse.Namespace) -> int:
    """Send toggle, ptt-start or ptt-stop to the running session."""
    from .controller import send_command

    try:
        print(send_command(args.command))
    except OSError as exc:
        print(f"Is 'silentdictate run' running? {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_purge(_cfg: Config, args: argparse.Namespace) -> int:
    """Delete recordings, enrollment data and calibration, plus models if asked."""
    root = data_dir()
    targets = [root / name for name in PURGE_TARGETS]
    if args.models:
        targets.append(root / "models")
    existing = [t for t in targets if t.exists()]
    if not existing:
        print("Nothing to purge.")
        return 0
    print("Will permanently delete:", *existing, sep="\n  ")
    if not args.yes and input("Type yes to confirm: ").strip().lower() != "yes":
        print("Cancelled.")
        return 1
    for target in existing:
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    print("Deleted.")
    return 0


def cmd_download_models(cfg: Config, args: argparse.Namespace) -> int:
    """Download models and verify their checksums."""
    from .models import ensure_piper, ensure_whisper, piper_voice_dir, verify, whisper_dir

    failed = False
    for name in args.whisper or [cfg.asr.model]:
        ensure_whisper(name)
        bad = verify(whisper_dir(name))
        print(f"{name}: {'corrupt ' + str(bad) if bad else 'checksums ok'}")
        failed |= bool(bad)
    ensure_piper(cfg.tts.voice)
    bad = verify(piper_voice_dir(cfg.tts.voice))
    print(f"{cfg.tts.voice}: {'corrupt ' + str(bad) if bad else 'checksums ok'}")
    return 1 if failed or bad else 0


def cmd_config_init(cfg: Config, _args: argparse.Namespace) -> int:
    """Write the current settings to the config file."""
    print(save_config(cfg))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="silentdictate", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--set-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=f"override a setting, e.g. asr.model=small (file: {config_path()})",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: object, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        return sub

    add("run", cmd_run, "start dictating")
    add("devices", cmd_devices, "list audio devices")
    add("calibrate", cmd_calibrate, "measure your whisper level")
    add("enroll", cmd_enroll, "record whispered prompts for fine-tuning")
    tune = add("finetune", cmd_finetune, "LoRA fine-tune Whisper on your recordings")
    tune.add_argument("--base-model", default="small")
    tune.add_argument("--epochs", type=int, default=5)
    tune.add_argument("--corpus", help="local wTIMIT/CHAINS-style directory")
    bench = add("bench", cmd_bench, "measure WER and latency")
    bench.add_argument("--set", choices=["fixtures", "personal"], default="fixtures")
    bench.add_argument("--model", help="default: asr.model, or small for fixtures")
    bench.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu", "metal"])
    bench.add_argument("--fixtures-dir", default="tests/fixtures")
    setup = add("setup-virtual-mic", cmd_setup_virtual_mic, "set up the virtual microphone")
    setup.add_argument("--run", action="store_true", help="run the script now")
    add("doctor", cmd_doctor, "check permissions, devices and models")
    add("toggle", cmd_control, "toggle recording in a running session")
    add("ptt-start", cmd_control, "start recording in a running session")
    add("ptt-stop", cmd_control, "stop recording in a running session")
    purge = add("purge-data", cmd_purge, "delete recorded audio and calibration")
    purge.add_argument("--models", action="store_true", help="also delete downloaded models")
    purge.add_argument("--yes", action="store_true", help="skip the confirmation")
    models = add("download-models", cmd_download_models, "download models with checksums")
    models.add_argument("--whisper", action="append", help="model name; repeatable")
    add("config-init", cmd_config_init, "write a config file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(overrides=args.set_config)
        if args.command == "bench" and args.set == "fixtures" and not args.model:
            args.model = "small"
        code: int = args.handler(cfg, args)
    except (ConfigError, LookupError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
