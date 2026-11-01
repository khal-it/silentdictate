"""Hotkey and Wayland-socket control of recording."""

from __future__ import annotations

import importlib
import os
import socket
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import platformdirs

from .config import APP_NAME

COMMANDS = {"toggle", "ptt-start", "ptt-stop"}
# Python on Windows has no AF_UNIX, and only Wayland needs the socket.
SOCKETS_AVAILABLE = hasattr(socket, "AF_UNIX")


def is_wayland(env: Mapping[str, str] = os.environ) -> bool:
    """Return True in a Wayland session."""
    return env.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def socket_path() -> Path:
    """Return the control socket path in the per-user runtime dir."""
    override = os.environ.get("SILENTDICTATE_SOCKET")
    if override:
        return Path(override)
    return Path(platformdirs.user_runtime_dir(APP_NAME, ensure_exists=True)) / "control.sock"


class Controller:
    """Recording state machine shared by hotkey, socket and hold or tap modes."""

    def __init__(
        self, on_start: Callable[[], None], on_stop: Callable[[], None], mode: str = "hold"
    ) -> None:
        self.on_start = on_start
        self.on_stop = on_stop
        self.mode = mode
        self.recording = False
        self._lock = threading.Lock()

    def ptt_start(self) -> None:
        """Start recording; ignored if already recording."""
        with self._lock:
            if self.recording:
                return
            self.recording = True
        self.on_start()

    def ptt_stop(self) -> None:
        """Stop recording; ignored if not recording."""
        with self._lock:
            if not self.recording:
                return
            self.recording = False
        self.on_stop()

    def toggle(self) -> None:
        """Flip the recording state."""
        self.ptt_stop() if self.recording else self.ptt_start()

    def key_down(self) -> None:
        """Handle the hotkey going down; auto-repeat is absorbed by the state check."""
        if self.mode == "hold":
            self.ptt_start()
        elif self.mode == "tap":
            self.toggle()

    def key_up(self) -> None:
        """Handle the hotkey going up."""
        if self.mode == "hold":
            self.ptt_stop()

    def command(self, name: str) -> bool:
        """Run a socket command; unknown names are rejected."""
        if name not in COMMANDS:
            return False
        {"toggle": self.toggle, "ptt-start": self.ptt_start, "ptt-stop": self.ptt_stop}[name]()
        return True


class HotkeyListener:
    """Global hotkey through pynput; tap mode must ignore key auto-repeat."""

    def __init__(self, hotkey: str, controller: Controller, keyboard: Any = None) -> None:
        if keyboard is None:
            keyboard = importlib.import_module("pynput.keyboard")
        self.key: Any = getattr(keyboard.Key, hotkey, None)
        if self.key is None:
            self.key = keyboard.KeyCode.from_char(hotkey)
        self.controller = controller
        self._held = False
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release)

    def _press(self, key: Any) -> None:
        if key == self.key and not self._held:
            self._held = True
            self.controller.key_down()

    def _release(self, key: Any) -> None:
        if key == self.key:
            self._held = False
            self.controller.key_up()

    def start(self) -> None:
        """Start listening."""
        self._listener.start()

    def stop(self) -> None:
        """Stop listening."""
        self._listener.stop()


class SocketServer:
    """Accepts one-line commands on a user-only Unix socket."""

    def __init__(self, controller: Controller, path: Path | None = None) -> None:
        self.controller = controller
        self.path = path or socket_path()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind the socket and serve on a daemon thread, if the OS has Unix sockets."""
        if not SOCKETS_AVAILABLE:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.unlink(missing_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.path))
        self.path.chmod(0o600)
        sock.listen()
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        assert self._sock is not None
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                name = conn.recv(64).decode("utf-8", "replace").strip()
                conn.sendall(b"ok\n" if self.controller.command(name) else b"unknown command\n")

    def stop(self) -> None:
        """Close the socket and remove its file."""
        if self._sock is not None:
            # shutdown wakes the blocked accept(); close alone does not on Linux.
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None
        self.path.unlink(missing_ok=True)


def send_command(name: str, path: Path | None = None) -> str:
    """Send a command to a running ``silentdictate run``."""
    if name not in COMMANDS:
        raise ValueError(f"unknown command {name!r}")
    if not SOCKETS_AVAILABLE:
        raise OSError("this command needs Unix sockets; use the hotkey on this OS")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        sock.connect(str(path or socket_path()))
        sock.sendall(name.encode() + b"\n")
        return sock.recv(64).decode().strip()
