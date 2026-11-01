"""LoRA fine-tuning of Whisper and export to CTranslate2."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile

from ..config import data_dir
from ..models import models_dir
from ..output.virtual_mic_sink import resample
from .data import Sample

HF_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5",
}


@dataclass
class TrainConfig:
    """Fine-tuning hyperparameters."""

    base_model: str = "small"
    language: str = "en"
    epochs: int = 5
    learning_rate: float = 1e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def finetuned_dir() -> Path:
    """Return the fine-tuned CTranslate2 model directory."""
    return models_dir() / "finetuned" / "ct2"


def pick_device(torch: Any) -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def lora_settings(cfg: TrainConfig) -> dict[str, Any]:
    """Return LoraConfig arguments targeting Whisper's attention projections."""
    return {
        "r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": ["q_proj", "v_proj"],
    }


def _read_16k(path: Path) -> Any:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return resample(audio[:, 0], int(rate), 16000)


def train(samples: list[Sample], cfg: TrainConfig) -> Any:
    """Train LoRA adapters on samples and return (merged model, processor)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if not samples:
        raise ValueError("no training samples; run: silentdictate enroll")
    name = HF_MODELS.get(cfg.base_model, cfg.base_model)
    cache = str(models_dir() / "hf")
    processor = WhisperProcessor.from_pretrained(
        name, language=cfg.language, task="transcribe", cache_dir=cache
    )
    model = WhisperForConditionalGeneration.from_pretrained(name, cache_dir=cache)
    device = pick_device(torch)
    # task_type is left unset: the seq2seq wrapper forwards input_ids, which Whisper rejects.
    model = get_peft_model(model, LoraConfig(**lora_settings(cfg))).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate
    )
    for epoch in range(cfg.epochs):
        total = 0.0
        for sample in samples:
            features = processor(
                _read_16k(sample.path), sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            labels = processor.tokenizer(sample.text, return_tensors="pt").input_ids.to(device)
            loss = model(input_features=features, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}/{cfg.epochs} loss {total / len(samples):.4f}")
    return model.merge_and_unload().to("cpu"), processor


def export_ct2(model: Any, processor: Any, output: Path | None = None) -> Path:
    """Convert the merged model to CTranslate2 for faster-whisper."""
    from ctranslate2.converters import TransformersConverter

    output = output or finetuned_dir()
    hf_dir = data_dir() / "finetune-merged"
    model.save_pretrained(hf_dir)
    processor.save_pretrained(hf_dir)
    # Newer transformers merges this into processor_config.json; the converter needs it apart.
    processor.feature_extractor.save_pretrained(hf_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    TransformersConverter(
        str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]
    ).convert(str(output), quantization="int8", force=True)
    shutil.rmtree(hf_dir)
    return output
