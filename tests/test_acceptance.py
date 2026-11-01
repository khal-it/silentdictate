"""Spec acceptance criteria 5-9, using real models and real processes."""

from __future__ import annotations

import os
import pty
import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from conftest import ROOT, make_asr, make_pipeline

from silentdictate.bench import load_audio, load_set, wer
from silentdictate.config import TtsConfig
from silentdictate.finetune.prompts import SENTENCES
from silentdictate.output.type_sink import TerminalBackend, TypeSink
from silentdictate.output.virtual_mic_sink import FakeDevice, VirtualMicSink, resample
from silentdictate.tts import PiperTts
from silentdictate.vad import VadSegmenter, silero_model

UNICODE_TEXT = 'Héllo, wörld! — ¿qué tal? 日本語 🎉 naïve (a+b)=c; "quoted" it\'s 100%'


def loopback_wer(sentences: list[str], tts: PiperTts, asr: object) -> float:
    device = FakeDevice()
    sink = VirtualMicSink(tts, device)
    pipeline = make_pipeline(asr)
    outputs = []
    for sentence in sentences:
        device.played.clear()
        sink.deliver(sentence)
        captured = resample(device.played[0], device.rate, 16000)
        outputs.append(pipeline.process(captured).text)
    return wer(sentences, outputs)


def test_c5_virtual_mic_loopback_wer(real_tts: PiperTts, real_asr: object) -> None:
    rate = loopback_wer(SENTENCES[:20], real_tts, real_asr)
    print(f"loopback WER over 20 sentences: {rate:.2%}")
    assert rate <= 0.05


def test_c6_silence_clips_produce_no_characters(fixtures_root: Path, real_asr: object) -> None:
    samples = load_set(fixtures_root / "silence")
    assert len(samples) >= 30
    pipeline = make_pipeline(real_asr)
    for sample in samples:
        assert pipeline.process(load_audio(sample.path)).text == "", sample.path.name


def _noises() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 16000 * 4
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    brown /= np.abs(brown).max()
    t = np.arange(n) / 16000
    return {
        "white-quiet": white * 10 ** (-60 / 20),
        "white-loud": white * 10 ** (-25 / 20),
        "brown": brown * 0.1,
        "hum": 0.05 * np.sin(2 * np.pi * 60 * t),
        "digital": np.zeros(n),
    }


@pytest.mark.parametrize("name", list(_noises()))
def test_c6_vad_mode_on_noise_produces_no_characters(name: str, real_asr: object) -> None:
    audio = _noises()[name].astype(np.float32)
    segmenter = VadSegmenter(silero_model())
    pipeline = make_pipeline(real_asr)
    output = ""
    for start in range(0, audio.size, 512):
        for utterance in segmenter.feed(audio[start : start + 512]):
            output += pipeline.process(utterance).text
    assert output == ""


def test_vad_mode_does_detect_speech(fixtures_root: Path, real_asr: object) -> None:
    """Guards the test above against passing because the segmenter never fires."""
    clip = load_audio(fixtures_root / "normal" / "000.wav")
    audio = np.concatenate([np.zeros(8000, np.float32), clip, np.zeros(24000, np.float32)])
    segmenter = VadSegmenter(silero_model())
    utterances: list[np.ndarray] = []
    for start in range(0, audio.size, 512):
        utterances += segmenter.feed(audio[start : start + 512])
    assert len(utterances) == 1
    assert "boat" in make_pipeline(real_asr).process(utterances[0]).text


HARNESS = textwrap.dedent(
    """
    import os, sys, tty
    tty.setraw(0)
    os.write(1, b"ready")
    received = bytearray()
    while True:
        try:
            data = os.read(0, 4096)
        except OSError:
            break
        if not data:
            break
        received += data
        if received.endswith(b"\\x00END"):
            break
    os.write(1, bytes(received))
    """
)


def run_pty(text: str, auto_submit: bool) -> bytes:
    """Deliver text to a real child process on a pty and return the bytes it read."""
    master, slave = pty.openpty()
    child = subprocess.Popen(
        [sys.executable, "-c", HARNESS], stdin=slave, stdout=subprocess.PIPE, close_fds=True
    )
    os.close(slave)
    assert child.stdout is not None
    assert child.stdout.read(5) == b"ready"
    TypeSink([TerminalBackend(master)], delay_ms=1, auto_submit=auto_submit).deliver(text)
    os.write(master, b"\x00END")
    output = child.stdout.read()
    child.wait(timeout=10)
    os.close(master)
    assert output.endswith(b"\x00END")
    return output[: -len(b"\x00END")]


def test_c7_pty_exact_unicode_without_trailing_newline() -> None:
    assert run_pty(UNICODE_TEXT, auto_submit=False) == UNICODE_TEXT.encode("utf-8")


def test_c7_pty_strips_trailing_newline_and_controls() -> None:
    received = run_pty("rm -rf ok\x1b[31m\x07\n\n", auto_submit=False)
    assert received == b"rm -rf ok[31m"
    assert b"\n" not in received and b"\r" not in received


def test_c7_pty_long_text_is_chunked_intact() -> None:
    text = "ünï " * 200
    assert run_pty(text, auto_submit=False) == text.encode("utf-8")


def test_c7_pty_enter_only_with_auto_submit() -> None:
    assert run_pty("go", auto_submit=True) == b"go\r"


def test_c8_block_network_really_blocks(block_network: None) -> None:
    with pytest.raises(AssertionError):
        socket.create_connection(("example.com", 443))
    with socket.socket(socket.AF_INET) as sock, pytest.raises(AssertionError):
        sock.connect(("1.1.1.1", 443))


def test_c8_core_pipeline_offline(
    small_model_path: Path, voice_path: Path, fixtures_root: Path, block_network: None
) -> None:
    """Load every model and run all pipeline paths with sockets blocked."""
    from piper import PiperVoice

    asr = make_asr(small_model_path)
    tts = PiperTts(TtsConfig(), PiperVoice.load(voice_path))
    typed: list[str] = []

    class Capture:
        def deliver(self, text: str) -> None:
            typed.append(text)

    pipeline = make_pipeline(asr, [Capture()])
    samples = load_set(fixtures_root / "normal")[:4]
    outputs = [pipeline.process(load_audio(s.path)).text for s in samples]
    assert wer([s.text for s in samples], outputs) <= 0.15
    assert typed == outputs
    faint = load_set(fixtures_root / "quiet")[:2]
    assert all(pipeline.process(load_audio(s.path)).text for s in faint)
    for sample in load_set(fixtures_root / "silence")[::10]:
        assert pipeline.process(load_audio(sample.path)).text == ""
    # All 20 sentences: over a handful, one misheard word alone would exceed 5%.
    assert loopback_wer(SENTENCES[:20], tts, asr) <= 0.05


ARTIFACTS = [
    "DECISIONS.md",
    "CHANGELOG.md",
    "MANUAL_TESTS.md",
    "scripts/setup_virtual_mic_macos.sh",
    "scripts/setup_virtual_mic_windows.md",
    "scripts/detect_virtual_mic_windows.ps1",
    "scripts/setup_virtual_mic_linux.sh",
    "scripts/teardown_virtual_mic_linux.sh",
    "scripts/setup_wayland_injection.sh",
    "scripts/make_fixtures.py",
]


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_c9_artifact_exists_and_is_linked(artifact: str) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (ROOT / artifact).is_file()
    assert f"]({artifact})" in readme


def test_c9_readme_recommends_mic_distance() -> None:
    assert re.search(r"3[–-]5 cm", (ROOT / "README.md").read_text(encoding="utf-8"))


def test_fixture_sets_match_the_spec(fixtures_root: Path) -> None:
    normal = load_set(fixtures_root / "normal")
    quiet = load_set(fixtures_root / "quiet")
    assert len(normal) >= 20 and len(quiet) == len(normal)
    assert all(s.text for s in normal)
    loud = np.sqrt(np.mean(load_audio(normal[0].path) ** 2))
    faint = np.sqrt(np.mean(load_audio(quiet[0].path) ** 2))
    assert 20 < 20 * np.log10(loud / faint) < 30
