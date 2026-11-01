"""Wires config into a running dictation session."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import soundfile

from .asr.base import AsrBackend, create_backend
from .asr.guard import HallucinationGuard
from .audio_in import SAMPLE_RATE, Audio, Calibration, Recorder, rms_db, volume_feedback
from .config import Config, data_dir
from .controller import Controller, HotkeyListener, SocketServer, is_wayland
from .doctor import request_macos_permissions
from .output.type_sink import InjectionError, TypeSink, select_backends
from .output.virtual_mic_sink import PttAssist, SoundDevicePlayer, VirtualMicSink
from .pipeline import Pipeline, Sink
from .postprocess import DefaultPostProcessor
from .status import StatusIndicator, TrayIndicator
from .tts import create_tts
from .vad import VadSegmenter, silero_model

MIC_SILENT_DB = -90.0
MIC_SILENT_HINT = ". The mic is silent: check Privacy & Security > Microphone for your terminal"


def permission_message(missing: list[str]) -> str:
    """Explain what to do after macOS shows its permission dialogs."""
    return (
        f"macOS needs {' and '.join(missing)}. Use the dialog to open System Settings, turn on "
        "your terminal app, then restart it and run this command again."
    )


def build_sinks(cfg: Config) -> list[Sink]:
    """Build the sinks for ``output_mode``."""
    sinks: list[Sink] = []
    if cfg.output_mode in {"type", "both"}:
        sinks.append(
            TypeSink(
                select_backends(),
                delay_ms=cfg.type_delay_ms,
                auto_submit=cfg.auto_submit,
                allow_newlines=cfg.allow_newlines,
            )
        )
    if cfg.output_mode in {"virtual_mic", "both"}:
        sinks.append(
            VirtualMicSink(
                create_tts(cfg.tts),
                SoundDevicePlayer(cfg.virtual_mic_device),
                ptt=PttAssist(cfg.ptt_assist_key, cfg.ptt_assist_mode)
                if cfg.ptt_assist_key
                else None,
                monitor=SoundDevicePlayer(cfg.monitor_device) if cfg.monitor_device else None,
            )
        )
    return sinks


def build_pipeline(
    cfg: Config, sinks: list[Sink] | None = None, asr: AsrBackend | None = None
) -> Pipeline:
    """Build the pipeline from config."""
    return Pipeline(
        asr or create_backend(cfg.asr),
        HallucinationGuard(silero_model(), cfg.vad.threshold),
        DefaultPostProcessor(cfg.postprocess, language=cfg.asr.language),
        build_sinks(cfg) if sinks is None else sinks,
    )


def save_utterance(audio: Audio, directory: Path | None = None) -> Path:
    """Write an utterance; called only when privacy.save_audio is on."""
    directory = directory or data_dir() / "recordings"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.time_ns()}.wav"
    soundfile.write(path, audio, SAMPLE_RATE)
    return path


class Session:
    """Runs utterances through the pipeline on a worker thread."""

    def __init__(self, cfg: Config, pipeline: Pipeline, status: StatusIndicator) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.status = status
        self.calibration = Calibration.load()
        self.jobs: queue.Queue[Audio | None] = queue.Queue()
        self.worker = threading.Thread(target=self._work, daemon=True)

    def submit(self, audio: Audio) -> None:
        """Queue an utterance."""
        self.jobs.put(audio)

    def process(self, audio: Audio) -> None:
        """Process one utterance, reporting failures without ending the session."""
        level = rms_db(audio)
        self.status.set("working", volume_feedback(level, self.calibration))
        if self.cfg.privacy.save_audio:
            save_utterance(audio)
        try:
            result = self.pipeline.process(audio)
            hint = MIC_SILENT_HINT if level < MIC_SILENT_DB else ""
            seconds = audio.size / SAMPLE_RATE
            self.status.set(
                "idle",
                f"{seconds:.1f} s, {level:.0f} dBFS, speech {result.speech_ms:.0f} ms: "
                f"{result.outcome}{hint}",
            )
        except (InjectionError, OSError, ImportError, LookupError, RuntimeError) as exc:
            # One failed utterance must not kill the worker thread and silently end dictation.
            self.status.set("error", str(exc))

    def _work(self) -> None:
        while (audio := self.jobs.get()) is not None:
            self.process(audio)

    def close(self) -> None:
        """Finish queued work."""
        self.jobs.put(None)
        self.worker.join()


def run(cfg: Config) -> None:
    """Dictate until interrupted."""
    status: StatusIndicator = TrayIndicator() if cfg.tray else StatusIndicator()
    if sys.platform == "darwin":
        missing = request_macos_permissions()
        if missing:
            status.set("error", permission_message(missing))
            raise SystemExit(1)
    status.set("working", "loading models")
    session = Session(cfg, build_pipeline(cfg), status)
    segmenter = None
    if cfg.mode == "vad":
        segmenter = VadSegmenter(
            silero_model(), cfg.vad.threshold, cfg.vad.preroll_ms, cfg.vad.min_silence_ms
        )

    def on_block(block: Audio) -> None:
        if segmenter is not None:
            for utterance in segmenter.feed(block):
                session.submit(utterance)

    recorder = Recorder(cfg.input_device, cfg.vad.preroll_ms, on_block=on_block)

    def on_start() -> None:
        recorder.start()
        status.set("recording")

    controller = Controller(on_start, lambda: session.submit(recorder.stop()), cfg.mode)
    server = SocketServer(controller)
    listener = None
    session.worker.start()
    recorder.open()
    if cfg.mode != "vad":
        server.start()
        if not is_wayland():
            listener = HotkeyListener(cfg.hotkey, controller)
            listener.start()
    status.set("idle", f"ready, mode={cfg.mode}, hotkey={cfg.hotkey}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        if listener:
            listener.stop()
        server.stop()
        recorder.close()
        session.close()
