"""Status indicator: a terminal line, plus an optional tray icon."""

from __future__ import annotations

import sys
import threading
from typing import Any, TextIO

COLORS = {"idle": "gray", "recording": "red", "working": "orange", "error": "purple"}


class StatusIndicator:
    """Prints state changes. Never prints or logs dictated text."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr
        self.state = "idle"

    def set(self, state: str, detail: str = "") -> None:
        """Show a new state."""
        self.state = state
        suffix = f": {detail}" if detail else ""
        print(f"[silentdictate] {state}{suffix}", file=self.stream, flush=True)


class TrayIndicator(StatusIndicator):
    """Adds a colored tray dot; needs the ``tray`` extra."""

    def __init__(self, stream: TextIO | None = None, icon: Any = None) -> None:
        super().__init__(stream)
        if icon is None:
            import pystray

            icon = pystray.Icon("silentdictate", self._image("idle"), "SilentDictate")
            threading.Thread(target=icon.run, daemon=True).start()
        self.icon = icon

    @staticmethod
    def _image(state: str) -> Any:
        from PIL import Image

        return Image.new("RGB", (32, 32), COLORS[state])

    def set(self, state: str, detail: str = "") -> None:
        """Update the terminal and the tray."""
        super().set(state, detail)
        self.icon.icon = self._image(state)
