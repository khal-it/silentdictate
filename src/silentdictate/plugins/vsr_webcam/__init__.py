"""Webcam visual speech recognition plugin: interface stub only (milestone 8)."""

from __future__ import annotations

from typing import Protocol


class VsrPlugin(Protocol):
    """What a future lip-reading backend must provide."""

    def start(self) -> None:
        """Open the camera and show the camera-active indicator."""
        ...

    def stop(self) -> str:
        """Close the camera and return recognized text; no video may be saved."""
        ...


class WebcamVsrStub:
    """Placeholder that refuses to run, so nothing can open the camera by accident."""

    def start(self) -> None:
        """Always raise; the plugin is not implemented."""
        raise NotImplementedError("The webcam VSR plugin is a later milestone; see its README.")

    def stop(self) -> str:
        """Always raise; the plugin is not implemented."""
        raise NotImplementedError("The webcam VSR plugin is a later milestone; see its README.")
