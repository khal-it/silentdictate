"""TypeSink: inject text into the focused window on every supported OS."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

CHUNK_CHARS = 64
PASTE_THRESHOLD = 2000
CLIPBOARD_SETTLE_S = 0.2
ACCESSIBILITY_FIX = (
    "macOS Accessibility permission is missing. Open System Settings > Privacy & Security > "
    "Accessibility, enable your terminal (or the silentdictate app), then restart it."
)
_AX_LIB = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"


class InjectionError(RuntimeError):
    """Raised when a backend cannot deliver text."""


class Backend(Protocol):
    """One way of getting text into the focused app."""

    name: str
    paste: bool

    def type_text(self, text: str, delay_ms: int) -> None:
        """Deliver text; raise InjectionError on failure."""
        ...

    def enter(self) -> None:
        """Press Enter."""
        ...


def sanitize(text: str, allow_newlines: bool = False) -> str:
    """Strip control characters; newlines become spaces because Enter would submit."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    kept = "".join(c for c in text if c == "\n" or unicodedata.category(c) != "Cc")
    kept = kept.rstrip("\n")
    return kept if allow_newlines else kept.replace("\n", " ")


def chunked(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split text into chunks of at most ``size`` characters."""
    return [text[i : i + size] for i in range(0, len(text), size)]


class TerminalBackend:
    """Write directly to a pty master, for embedding a terminal app under SilentDictate."""

    name = "terminal"
    paste = False

    def __init__(self, fd: int) -> None:
        self.fd = fd

    def type_text(self, text: str, delay_ms: int) -> None:
        """Write each character as UTF-8."""
        try:
            for char in text:
                os.write(self.fd, char.encode("utf-8"))
                if delay_ms:
                    time.sleep(delay_ms / 1000)
        except OSError as exc:
            raise InjectionError(str(exc)) from exc

    def enter(self) -> None:
        """Send a carriage return, which is what the Enter key produces in a terminal."""
        os.write(self.fd, b"\r")


def _run(argv: Sequence[str], text_input: str | None = None) -> str:
    try:
        done = subprocess.run(
            argv, input=text_input, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InjectionError(f"{argv[0]}: {exc}") from exc
    if done.returncode != 0:
        raise InjectionError(f"{argv[0]}: {done.stderr.strip() or done.returncode}")
    return done.stdout


class XdotoolBackend:
    """Linux X11."""

    name = "xdotool"
    paste = False

    def type_text(self, text: str, delay_ms: int) -> None:
        """Type through xdotool."""
        _run(["xdotool", "type", "--clearmodifiers", "--delay", str(delay_ms), "--", text])

    def enter(self) -> None:
        """Press Return."""
        _run(["xdotool", "key", "Return"])


class WtypeBackend:
    """Wayland virtual-keyboard protocol; works on Sway and Hyprland, not GNOME."""

    name = "wtype"
    paste = False

    def type_text(self, text: str, delay_ms: int) -> None:
        """Type through wtype."""
        _run(["wtype", "-d", str(delay_ms), "--", text])

    def enter(self) -> None:
        """Press Return."""
        _run(["wtype", "-k", "Return"])


class YdotoolBackend:
    """uinput injection; needs ydotoold and /dev/uinput permission."""

    name = "ydotool"
    paste = False

    def type_text(self, text: str, delay_ms: int) -> None:
        """Type through ydotool."""
        _run(["ydotool", "type", "--key-delay", str(delay_ms), "--", text])

    def enter(self) -> None:
        """Press Enter (Linux keycode 28)."""
        _run(["ydotool", "key", "28:1", "28:0"])


class WlCopyBackend:
    """Clipboard then Ctrl+V, the last resort on Wayland."""

    name = "wl-copy"
    paste = True

    def __init__(self, which: Callable[[str], str | None] = shutil.which) -> None:
        self.which = which

    def type_text(self, text: str, delay_ms: int) -> None:
        """Copy, then paste if a key injector exists; otherwise tell the user."""
        _run(["wl-copy", "--", text])
        if self.which("wtype"):
            _run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"])
        elif self.which("ydotool"):
            _run(["ydotool", "key", "29:1", "47:1", "47:0", "29:0"])
        else:
            print("Text is on the clipboard; press Ctrl+V to paste it.")

    def enter(self) -> None:
        """Enter cannot be sent without a key injector."""
        raise InjectionError("wl-copy cannot press Enter")


def ax_trusted() -> bool:
    """Return True if this process has macOS Accessibility permission."""
    return bool(ctypes.cdll.LoadLibrary(_AX_LIB).AXIsProcessTrusted())


class QuartzBackend:
    """macOS Unicode typing through CGEvents."""

    name = "quartz"
    paste = False

    def __init__(self, quartz: Any = None, trusted: Callable[[], bool] = ax_trusted) -> None:
        self.quartz = quartz
        self.trusted = trusted

    def _api(self) -> Any:
        if not self.trusted():
            raise InjectionError(ACCESSIBILITY_FIX)
        if self.quartz is None:
            import Quartz

            self.quartz = Quartz
        return self.quartz

    def _key(self, keycode: int, char: str = "", flags: int = 0) -> None:
        api = self._api()
        for down in (True, False):
            event = api.CGEventCreateKeyboardEvent(None, keycode, down)
            if char:
                units = len(char.encode("utf-16-le")) // 2
                api.CGEventKeyboardSetUnicodeString(event, units, char)
            if flags:
                api.CGEventSetFlags(event, flags)
            api.CGEventPost(api.kCGHIDEventTap, event)

    def type_text(self, text: str, delay_ms: int) -> None:
        """Post one Unicode key event per character."""
        for char in text:
            self._key(0, char)
            if delay_ms:
                time.sleep(delay_ms / 1000)

    def enter(self) -> None:
        """Press Return."""
        self._key(36)


class MacPasteBackend(QuartzBackend):
    """Clipboard + Cmd+V, restoring the previous clipboard text afterwards."""

    name = "mac-paste"
    paste = True

    def type_text(self, text: str, delay_ms: int) -> None:
        """Paste text."""
        env = {**os.environ, "LANG": "en_US.UTF-8"}
        previous = subprocess.run(
            ["pbpaste"], capture_output=True, text=True, env=env, check=False
        ).stdout
        subprocess.run(["pbcopy"], input=text, text=True, env=env, check=True)
        try:
            self._key(9, flags=self._api().kCGEventFlagMaskCommand)
            time.sleep(CLIPBOARD_SETTLE_S)
        finally:
            # shortcut: restores text only, not images; use NSPasteboard if that bites.
            subprocess.run(["pbcopy"], input=previous, text=True, env=env, check=True)


class _KeybdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    # The padding member gives the union Win32's size, which MOUSEINPUT sets.
    _fields_ = [("ki", _KeybdInput), ("padding", ctypes.c_byte * 32)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _InputUnion)]


_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_VK_CONTROL = 0x11
_VK_RETURN = 0x0D
_VK_V = 0x56


class WindowsBackend:
    """SendInput with KEYEVENTF_UNICODE."""

    name = "sendinput"
    paste = False

    def __init__(self, send: Callable[[list[tuple[int, int, int]]], int] | None = None) -> None:
        self._send = send or self._send_input

    @staticmethod
    def _send_input(events: list[tuple[int, int, int]]) -> int:
        array = (_Input * len(events))()
        for item, (vk, scan, flags) in zip(array, events, strict=True):
            item.type = 1  # INPUT_KEYBOARD
            item.union.ki = _KeybdInput(vk, scan, flags, 0, 0)
        windll: Any = getattr(ctypes, "windll")  # noqa: B009 - absent off Windows
        return int(windll.user32.SendInput(len(events), array, ctypes.sizeof(_Input)))

    @staticmethod
    def events_for(text: str) -> list[tuple[int, int, int]]:
        """Return down and up events per UTF-16 unit so emoji survive."""
        raw = text.encode("utf-16-le")
        units = [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)]
        return [
            (0, unit, flags)
            for unit in units
            for flags in (_KEYEVENTF_UNICODE, _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP)
        ]

    def _post(self, events: list[tuple[int, int, int]]) -> None:
        if self._send(events) != len(events):
            raise InjectionError("SendInput was blocked, possibly by an elevated window")

    def type_text(self, text: str, delay_ms: int) -> None:
        """Send text one character at a time."""
        for char in text:
            self._post(self.events_for(char))
            if delay_ms:
                time.sleep(delay_ms / 1000)

    def enter(self) -> None:
        """Press Return."""
        self._post([(_VK_RETURN, 0, 0), (_VK_RETURN, 0, _KEYEVENTF_KEYUP)])


class WindowsPasteBackend(WindowsBackend):
    """Clipboard + Ctrl+V for long text, restoring the previous clipboard text."""

    name = "windows-paste"
    paste = True

    def type_text(self, text: str, delay_ms: int) -> None:
        """Paste text."""
        previous = _run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"])
        set_clipboard = ["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"]
        _run(set_clipboard, text)
        try:
            self._post(
                [
                    (_VK_CONTROL, 0, 0),
                    (_VK_V, 0, 0),
                    (_VK_V, 0, _KEYEVENTF_KEYUP),
                    (_VK_CONTROL, 0, _KEYEVENTF_KEYUP),
                ]
            )
            time.sleep(CLIPBOARD_SETTLE_S)
        finally:
            _run(set_clipboard, previous)


def select_backends(
    platform: str = sys.platform,
    env: Mapping[str, str] = os.environ,
    which: Callable[[str], str | None] = shutil.which,
) -> list[Backend]:
    """Return the injection chain for this OS, most preferred first."""
    if platform == "darwin":
        return [QuartzBackend(), MacPasteBackend()]
    if platform == "win32":
        return [WindowsBackend(), WindowsPasteBackend()]
    if env.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        chain: list[Backend] = []
        if which("wtype"):
            chain.append(WtypeBackend())
        if which("ydotool"):
            chain.append(YdotoolBackend())
        if which("wl-copy"):
            chain.append(WlCopyBackend(which))
        return chain
    return [XdotoolBackend()] if which("xdotool") else []


class TypeSink:
    """Sanitize, chunk and inject text, falling through the backend chain."""

    def __init__(
        self,
        backends: Sequence[Backend],
        delay_ms: int = 3,
        auto_submit: bool = False,
        allow_newlines: bool = False,
        chunk_size: int = CHUNK_CHARS,
    ) -> None:
        self.backends = list(backends)
        self.delay_ms = delay_ms
        self.auto_submit = auto_submit
        self.allow_newlines = allow_newlines
        self.chunk_size = chunk_size

    def _ordered(self, text: str) -> list[Backend]:
        if len(text) > PASTE_THRESHOLD:
            return sorted(self.backends, key=lambda b: not b.paste)
        return self.backends

    def deliver(self, text: str) -> None:
        """Deliver text, retrying only undelivered chunks on the next backend."""
        clean = sanitize(text, self.allow_newlines)
        if not clean:
            return
        chunks = chunked(clean, self.chunk_size)
        errors: list[str] = []
        for backend in self._ordered(clean):
            try:
                if backend.paste:
                    backend.type_text("".join(chunks), self.delay_ms)
                    chunks = []
                while chunks:
                    backend.type_text(chunks[0], self.delay_ms)
                    chunks.pop(0)
                if self.auto_submit:
                    backend.enter()
                return
            except InjectionError as exc:
                errors.append(f"{backend.name}: {exc}")
        raise InjectionError("; ".join(errors) or "no text injection backend is available")
