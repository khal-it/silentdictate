"""TOML configuration stored in the OS config dir, with CLI overrides."""

from __future__ import annotations

import dataclasses
import os
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import platformdirs
import tomli_w

APP_NAME = "silentdictate"


class ConfigError(ValueError):
    """Raised for unknown keys or invalid values."""


@dataclass
class AsrConfig:
    """ASR backend settings."""

    backend: str = "whisper"
    model: str = "large-v3-turbo"
    device: str = "auto"
    language: str = "en"
    beam_size: int = 5
    hotwords: list[str] = field(default_factory=list)


@dataclass
class TtsConfig:
    """Text-to-speech settings for the virtual mic."""

    backend: str = "piper"
    voice: str = "en_US-lessac-medium"
    speed: float = 1.0


@dataclass
class VadConfig:
    """Silero VAD settings; the threshold is lowered for faint speech."""

    threshold: float = 0.3
    preroll_ms: int = 500
    min_silence_ms: int = 700


@dataclass
class PostprocessConfig:
    """Post-processing settings."""

    llm_cleanup: bool = False
    llm_endpoint: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "llama3.2"
    replacements: dict[str, str] = field(default_factory=dict)


@dataclass
class PrivacyConfig:
    """Privacy settings."""

    save_audio: bool = False


@dataclass
class Config:
    """Top-level configuration."""

    hotkey: str = ""
    mode: str = "hold"
    input_device: str = ""
    output_mode: str = "type"
    virtual_mic_device: str = ""
    auto_submit: bool = False
    allow_newlines: bool = False
    type_delay_ms: int = 3
    ptt_assist_key: str = ""
    ptt_assist_mode: str = "hold"
    monitor_device: str = ""
    tray: bool = False
    asr: AsrConfig = field(default_factory=AsrConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    def __post_init__(self) -> None:
        if not self.hotkey:
            self.hotkey = "alt_r" if sys.platform == "darwin" else "ctrl_r"
        if not self.virtual_mic_device:
            self.virtual_mic_device = default_virtual_mic()


def default_virtual_mic(platform: str = sys.platform) -> str:
    """Return the virtual device name created by this OS's setup script."""
    if platform == "darwin":
        return "BlackHole 2ch"
    if platform == "win32":
        return "CABLE Input"
    return "silentdict-sink"


CHOICES = {
    ("mode",): {"hold", "tap", "vad"},
    ("output_mode",): {"type", "virtual_mic", "both"},
    ("ptt_assist_mode",): {"hold", "tap"},
    ("asr", "backend"): {"whisper", "parakeet"},
    ("asr", "device"): {"auto", "cuda", "cpu", "metal"},
    ("tts", "backend"): {"piper", "kokoro", "os"},
}


def config_path() -> Path:
    """Return the config file path."""
    override = os.environ.get("SILENTDICTATE_CONFIG")
    return (
        Path(override) if override else Path(platformdirs.user_config_dir(APP_NAME)) / "config.toml"
    )


def data_dir() -> Path:
    """Return the app data dir, the only place audio and models are stored."""
    override = os.environ.get("SILENTDICTATE_DATA_DIR")
    return Path(override) if override else Path(platformdirs.user_data_dir(APP_NAME))


def _build(cls: type[Any], data: dict[str, Any], prefix: str = "") -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"unknown config key: {prefix}{key}")
        default = known[key].default_factory
        if dataclasses.is_dataclass(default):
            if not isinstance(value, dict):
                raise ConfigError(f"{prefix}{key} must be a table")
            kwargs[key] = _build(default, value, f"{prefix}{key}.")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must be key=value: {override!r}")
    dotted, raw = override.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # bare strings need no quotes on the command line
    *parents, leaf = dotted.strip().split(".")
    node = data
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def load_config(path: Path | None = None, overrides: Iterable[str] = ()) -> Config:
    """Load config from TOML, apply ``key=value`` overrides and validate."""
    path = path or config_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    for override in overrides:
        _apply_override(data, override)
    cfg: Config = _build(Config, data)
    for keys, allowed in CHOICES.items():
        value: Any = cfg
        for key in keys:
            value = getattr(value, key)
        if value not in allowed:
            raise ConfigError(f"{'.'.join(keys)} must be one of {sorted(allowed)}, got {value!r}")
    return cfg


def save_config(cfg: Config, path: Path | None = None) -> Path:
    """Write the config as TOML and return its path."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(dataclasses.asdict(cfg)).encode("utf-8"))
    return path
