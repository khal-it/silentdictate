"""Environment checks with printed fixes."""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .asr.whisper import cuda_available
from .audio_in import find_device, list_devices
from .config import Config
from .controller import is_wayland
from .models import is_cached, piper_voice_dir, verify, whisper_dir
from .output.type_sink import ACCESSIBILITY_FIX, ax_trusted, select_backends

_IOKIT = "/System/Library/Frameworks/IOKit.framework/IOKit"
_LISTEN_EVENT = 1


@dataclass
class Check:
    """One doctor result."""

    name: str
    ok: bool
    detail: str = ""


def check_mic(backend: object = None) -> Check:
    """Open the default input briefly; on macOS this also triggers the permission prompt."""
    fix = "Allow microphone access in your OS privacy settings and check a mic is connected."
    try:
        import sounddevice

        sd: Any = backend or sounddevice
        with sd.InputStream(samplerate=16000, channels=1):
            pass
    except Exception as exc:  # noqa: BLE001 - PortAudio raises varied errors
        return Check("microphone", False, f"{exc}. {fix}")
    return Check("microphone", True)


def input_monitoring_granted() -> bool:
    """Return True if macOS Input Monitoring allows the global hotkey."""
    return int(ctypes.cdll.LoadLibrary(_IOKIT).IOHIDCheckAccess(_LISTEN_EVENT)) == 0


def check_macos() -> list[Check]:
    """Check Accessibility and Input Monitoring."""
    return [
        Check("macOS Accessibility", ax_trusted(), ACCESSIBILITY_FIX),
        Check(
            "macOS Input Monitoring",
            input_monitoring_granted(),
            "Enable your terminal in System Settings > Privacy & Security > Input Monitoring.",
        ),
    ]


def check_wayland(env: Mapping[str, str] = os.environ) -> list[Check]:
    """Check text injection and uinput on Wayland."""
    fix = "Run scripts/setup_wayland_injection.sh."
    checks = [Check("Wayland text injection", bool(select_backends("linux", env)), fix)]
    if shutil.which("ydotool"):
        checks.append(Check("/dev/uinput access", os.access("/dev/uinput", os.W_OK), fix))
    return checks


def check_virtual_device(cfg: Config) -> Check:
    """Look for the virtual mic device."""
    try:
        find_device(cfg.virtual_mic_device, "output")
    except (LookupError, OSError) as exc:
        return Check("virtual mic", False, f"{exc}. Run: silentdictate setup-virtual-mic")
    return Check("virtual mic", True, cfg.virtual_mic_device)


def check_models(cfg: Config) -> list[Check]:
    """Verify cached models against their checksums."""
    checks = []
    for name, path in [
        (f"whisper model {cfg.asr.model}", whisper_dir(cfg.asr.model)),
        (f"piper voice {cfg.tts.voice}", piper_voice_dir(cfg.tts.voice)),
    ]:
        if not is_cached(path):
            checks.append(Check(name, False, "Run: silentdictate download-models"))
            continue
        bad = verify(path)
        checks.append(Check(name, not bad, f"corrupt: {bad}" if bad else ""))
    return checks


def check_gpu() -> Check:
    """Report CUDA; its absence is fine, so this never fails."""
    if cuda_available():
        return Check("GPU", True, "CUDA available")
    return Check("GPU", True, "no CUDA; using CPU or Metal")


def run_checks(cfg: Config) -> list[Check]:
    """Run all checks for this OS."""
    checks = [check_mic(), check_virtual_device(cfg), *check_models(cfg), check_gpu()]
    try:
        checks.append(Check("audio devices", bool(list_devices())))
    except OSError as exc:
        checks.append(Check("audio devices", False, str(exc)))
    if sys.platform == "darwin":
        checks.extend(check_macos())
    if is_wayland():
        checks.extend(check_wayland())
    return checks


def report(checks: list[Check]) -> int:
    """Print results and return the number of failures."""
    for check in checks:
        print(
            f"[{'ok' if check.ok else 'FAIL'}] {check.name}"
            + (
                f" - {check.detail}"
                if check.detail and (not check.ok or check.name == "GPU")
                else ""
            )
        )
    return sum(not c.ok for c in checks)
