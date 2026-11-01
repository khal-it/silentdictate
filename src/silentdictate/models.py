"""Model downloads with checksums, cached under the app data dir."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import data_dir

MANIFEST = "SHA256SUMS.json"
PIPER_REPO = "rhasspy/piper-voices"
WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
}


class ModelError(RuntimeError):
    """Raised for missing or corrupt models."""


def models_dir() -> Path:
    """Return the model cache directory."""
    return data_dir() / "models"


def whisper_dir(name: str) -> Path:
    """Return the cache path for a faster-whisper model."""
    return models_dir() / "whisper" / name


def piper_voice_dir(voice: str) -> Path:
    """Return the cache path for a Piper voice."""
    return models_dir() / "piper" / voice


def piper_patterns(voice: str) -> list[str]:
    """Return repo paths for a voice named like ``en_US-lessac-medium``."""
    locale, name, quality = voice.split("-")
    base = f"{locale.split('_')[0]}/{locale}/{name}/{quality}/{voice}.onnx"
    return [base, f"{base}.json"]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def fetch(repo: str, dest: Path, patterns: list[str] | None = None, api: Any = None) -> None:
    """Download a repo snapshot, check every file against Hub metadata, write a manifest."""
    from huggingface_hub import HfApi, snapshot_download

    api = api or HfApi()
    info = api.model_info(repo, files_metadata=True)
    snapshot_download(repo, revision=info.sha, local_dir=dest, allow_patterns=patterns)
    manifest: dict[str, str] = {}
    for sibling in info.siblings:
        path = dest / sibling.rfilename
        if not path.is_file():
            continue
        digest = _file_sha256(path)
        if sibling.lfs is not None:
            valid = sibling.lfs.sha256 == digest
        else:
            valid = sibling.blob_id == _git_blob_sha1(path)
        if not valid:
            path.unlink()
            raise ModelError(f"checksum mismatch for {repo}/{sibling.rfilename}; deleted it")
        manifest[sibling.rfilename] = digest
    if not manifest:
        raise ModelError(f"{repo} matched no files")
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def is_cached(dest: Path) -> bool:
    """Return True if a completed, checksummed download exists."""
    return (dest / MANIFEST).exists()


def verify(dest: Path) -> list[str]:
    """Return the files that are missing or no longer match the manifest."""
    if not is_cached(dest):
        return [MANIFEST]
    manifest: dict[str, str] = json.loads((dest / MANIFEST).read_text(encoding="utf-8"))
    return [
        name
        for name, digest in manifest.items()
        if not (dest / name).is_file() or _file_sha256(dest / name) != digest
    ]


def ensure_whisper(name: str) -> Path:
    """Return a whisper model dir, downloading on first use with a clear message."""
    if Path(name).is_dir():
        return Path(name)  # e.g. a fine-tuned export
    dest = whisper_dir(name)
    if is_cached(dest):
        return dest
    if name not in WHISPER_REPOS:
        raise ModelError(f"unknown whisper model {name!r}; choose from {sorted(WHISPER_REPOS)}")
    print(f"First use: downloading {name} from Hugging Face. Only the model is transferred.")
    fetch(WHISPER_REPOS[name], dest)
    return dest


def ensure_parakeet(onnx_name: str) -> Path:
    """Return a quantized Parakeet ONNX dir, downloading on first use with a clear message.

    This avoids the Hugging Face cache: ONNX Runtime refuses its blob layout, and models
    belong under the app data dir.
    """
    dest = models_dir() / "parakeet" / onnx_name
    if not is_cached(dest):
        print(f"First use: downloading {onnx_name}. Only the model is transferred.")
        version = onnx_name.rsplit("-", 1)[1]
        repo = f"istupakov/parakeet-tdt-0.6b-{version}-onnx"
        fetch(repo, dest, ["*.int8.onnx", "nemo128.onnx", "vocab.txt", "config.json"])
    return dest


def ensure_piper(voice: str) -> Path:
    """Return the voice's .onnx path, downloading on first use with a clear message."""
    dest = piper_voice_dir(voice)
    patterns = piper_patterns(voice)
    if not is_cached(dest):
        print(f"First use: downloading Piper voice {voice}. Only the model is transferred.")
        fetch(PIPER_REPO, dest, patterns)
    return dest / patterns[0]
