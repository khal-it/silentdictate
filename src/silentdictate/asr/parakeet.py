"""NVIDIA Parakeet TDT: NeMo on CUDA, onnx-asr on CPU and Apple Silicon."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from ..audio_in import SAMPLE_RATE, Audio
from ..config import AsrConfig
from ..models import ensure_parakeet
from .base import AsrBackend, Transcript
from .whisper import cuda_available

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
ONNX_NAMES = {
    "nvidia/parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
    "nvidia/parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
}


def model_name(cfg: AsrConfig) -> str:
    """Return the Parakeet model, ignoring a Whisper model left in the config."""
    return cfg.model if cfg.model in ONNX_NAMES else DEFAULT_MODEL


class OnnxParakeet:
    """Parakeet through onnx-asr."""

    name = "parakeet-onnx"

    def __init__(self, cfg: AsrConfig, model: Any = None) -> None:
        if model is None:
            import onnx_asr

            name = ONNX_NAMES[model_name(cfg)]
            model = onnx_asr.load_model(name, ensure_parakeet(name), quantization="int8")
        self.model = model

    def transcribe(self, audio: Audio) -> Transcript:
        """Recognize one utterance."""
        return Transcript(str(self.model.recognize(np.asarray(audio, dtype=np.float32))).strip())


class NemoParakeet:
    """Parakeet through NeMo."""

    name = "parakeet-nemo"

    def __init__(self, cfg: AsrConfig, model: Any = None) -> None:
        if model is None:
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name(cfg))
        self.model = model

    def transcribe(self, audio: Audio) -> Transcript:
        """Recognize one utterance through a temp file, which NeMo's API needs."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "utterance.wav"
            soundfile.write(path, audio, SAMPLE_RATE)
            result = self.model.transcribe([str(path)])[0]
        return Transcript(str(getattr(result, "text", result)).strip())


def create_parakeet(cfg: AsrConfig) -> AsrBackend:
    """Choose NeMo on CUDA and onnx-asr elsewhere."""
    if cfg.device == "cuda" or (cfg.device == "auto" and cuda_available()):
        return NemoParakeet(cfg)
    return OnnxParakeet(cfg)
