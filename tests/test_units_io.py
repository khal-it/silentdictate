"""Unit tests for backends, sinks, controller, doctor, bench, finetune, CLI and plugins."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile
from conftest import FakeSoundDevice, FakeStream, StubVad

from silentdictate import app, cli, doctor, tts
from silentdictate.asr import base as asr_base
from silentdictate.asr import parakeet as pk
from silentdictate.asr import whisper as wh
from silentdictate.asr.base import Transcript
from silentdictate.asr.guard import HallucinationGuard
from silentdictate.audio_in import Calibration, Recorder
from silentdictate.bench import (
    SetResult,
    bench_fixtures,
    bench_personal,
    format_results,
    load_set,
    run_set,
    wer,
)
from silentdictate.config import AsrConfig, Config, PostprocessConfig, TtsConfig, data_dir
from silentdictate.controller import (
    Controller,
    HotkeyListener,
    SocketServer,
    is_wayland,
    send_command,
)
from silentdictate.finetune import data as fdata
from silentdictate.finetune import train as ftrain
from silentdictate.finetune.enroll import enroll
from silentdictate.finetune.prompts import CODING_VOCABULARY, PROMPTS, SENTENCES
from silentdictate.models import MANIFEST, piper_voice_dir, whisper_dir
from silentdictate.output import type_sink as ts
from silentdictate.output import virtual_mic_sink as vm
from silentdictate.pipeline import Pipeline
from silentdictate.plugins.vsr_webcam import WebcamVsrStub
from silentdictate.postprocess import DefaultPostProcessor
from silentdictate.status import StatusIndicator, TrayIndicator


class Segment(SimpleNamespace):
    pass


class TestWhisperBackends:
    def test_faster_whisper_arguments(self) -> None:
        calls: dict[str, Any] = {}

        class Model:
            def transcribe(self, audio: Any, **kwargs: Any) -> Any:
                calls.update(kwargs)
                parts = [Segment(text=" Hello", no_speech_prob=0.2)]
                parts.append(Segment(text=" there.", no_speech_prob=0.1))
                return iter(parts), None

        cfg = AsrConfig(beam_size=3, language="de", hotwords=["pytest", "uv"])
        result = wh.FasterWhisperBackend(cfg, "cpu", Model()).transcribe(np.zeros(10, np.float32))
        assert result == Transcript("Hello there.", pytest.approx(0.1))
        assert calls["beam_size"] == 3 and calls["language"] == "de"
        assert calls["initial_prompt"] == "pytest, uv." and calls["hotwords"] == "pytest uv"
        assert calls["condition_on_previous_text"] is False

    def test_empty_result(self) -> None:
        model = SimpleNamespace(transcribe=lambda _a, **_k: (iter([]), None))
        out = wh.FasterWhisperBackend(AsrConfig(), "cpu", model).transcribe(np.zeros(1, np.float32))
        assert out == Transcript("", 0.0)

    def test_whisper_cpp(self) -> None:
        seen: dict[str, Any] = {}

        def transcribe(_audio: Any, **kwargs: Any) -> list[Any]:
            seen.update(kwargs)
            return [SimpleNamespace(text=" hi")]

        backend = wh.WhisperCppBackend(
            AsrConfig(hotwords=["uv"]), SimpleNamespace(transcribe=transcribe)
        )
        assert backend.transcribe(np.zeros(1, np.float32)).text == "hi"
        assert seen["initial_prompt"] == "uv." and seen["language"] == "en"

    @pytest.mark.parametrize(
        ("device", "cuda", "silicon", "expected"),
        [
            ("cpu", False, True, "faster-whisper"),
            ("cuda", False, False, "faster-whisper"),
            ("metal", False, False, "whisper.cpp"),
            ("auto", True, True, "faster-whisper"),
            ("auto", False, True, "whisper.cpp"),
            ("auto", False, False, "faster-whisper"),
        ],
    )
    def test_engine_choice(
        self,
        monkeypatch: pytest.MonkeyPatch,
        device: str,
        cuda: bool,
        silicon: bool,
        expected: str,
    ) -> None:
        monkeypatch.setattr(wh, "cuda_available", lambda: cuda)
        monkeypatch.setattr(wh, "is_apple_silicon", lambda: silicon)
        monkeypatch.setattr(
            wh,
            "FasterWhisperBackend",
            lambda cfg, dev: SimpleNamespace(name="faster-whisper", device=dev),
        )
        monkeypatch.setattr(
            wh, "WhisperCppBackend", lambda cfg: SimpleNamespace(name="whisper.cpp")
        )
        assert wh.create_whisper(AsrConfig(device=device)).name == expected

    def test_metal_falls_back_to_cpu_without_pywhispercpp(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def missing(_cfg: AsrConfig) -> Any:
            raise ImportError

        monkeypatch.setattr(wh, "cuda_available", lambda: False)
        monkeypatch.setattr(wh, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(wh, "WhisperCppBackend", missing)
        monkeypatch.setattr(
            wh, "FasterWhisperBackend", lambda cfg, dev: SimpleNamespace(name="fw", device=dev)
        )
        assert wh.create_whisper(AsrConfig()).device == "cpu"  # type: ignore[attr-defined]
        assert "silentdictate[metal]" in capsys.readouterr().out

    def test_compute_type(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import faster_whisper

        seen: list[dict[str, Any]] = []
        monkeypatch.setattr(
            faster_whisper, "WhisperModel", lambda path, **kw: seen.append(kw) or object()
        )
        wh.FasterWhisperBackend(AsrConfig(model=str(tmp_path)), "cuda")
        assert seen[0] == {"device": "cuda", "compute_type": "float16"}

    def test_factory_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pk, "create_parakeet", lambda cfg: "parakeet")
        monkeypatch.setattr(wh, "create_whisper", lambda cfg: "whisper")
        assert asr_base.create_backend(AsrConfig(backend="parakeet")) == "parakeet"
        assert asr_base.create_backend(AsrConfig()) == "whisper"


class TestParakeet:
    def test_onnx(self) -> None:
        model = SimpleNamespace(recognize=lambda _a: " hello ")
        assert (
            pk.OnnxParakeet(AsrConfig(), model).transcribe(np.zeros(1, np.float32)).text == "hello"
        )

    def test_onnx_loads_from_app_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def load_model(name: str, path: Path, quantization: str) -> Any:
            seen.update(name=name, path=path, quantization=quantization)
            return SimpleNamespace()

        monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(load_model=load_model))
        monkeypatch.setattr(pk, "ensure_parakeet", lambda name: data_dir() / name)
        pk.OnnxParakeet(AsrConfig())
        assert seen["name"] == "nemo-parakeet-tdt-0.6b-v3" and seen["quantization"] == "int8"
        assert seen["path"].is_relative_to(data_dir())

    def test_nemo_passes_a_wav_and_removes_it(self) -> None:
        paths: list[str] = []

        def transcribe(files: list[str]) -> list[Any]:
            paths.extend(files)
            assert Path(files[0]).exists()
            return [SimpleNamespace(text=" hi ")]

        backend = pk.NemoParakeet(AsrConfig(), SimpleNamespace(transcribe=transcribe))
        assert backend.transcribe(np.zeros(160, np.float32)).text == "hi"
        assert not Path(paths[0]).exists()

    def test_model_name(self) -> None:
        assert pk.model_name(AsrConfig(model="small")) == pk.DEFAULT_MODEL
        v2 = "nvidia/parakeet-tdt-0.6b-v2"
        assert pk.model_name(AsrConfig(model=v2)) == v2

    @pytest.mark.parametrize(
        ("device", "cuda", "cls"),
        [
            ("cuda", False, "nemo"),
            ("auto", True, "nemo"),
            ("auto", False, "onnx"),
            ("cpu", True, "onnx"),
        ],
    )
    def test_factory(
        self, monkeypatch: pytest.MonkeyPatch, device: str, cuda: bool, cls: str
    ) -> None:
        monkeypatch.setattr(pk, "cuda_available", lambda: cuda)
        monkeypatch.setattr(pk, "NemoParakeet", lambda cfg: "nemo")
        monkeypatch.setattr(pk, "OnnxParakeet", lambda cfg: "onnx")
        assert pk.create_parakeet(AsrConfig(device=device)) == cls


class Recording:
    """An injection backend recording calls, optionally failing after N chunks."""

    def __init__(self, name: str = "rec", paste: bool = False, fail_after: int = -1) -> None:
        self.name = name
        self.paste = paste
        self.fail_after = fail_after
        self.chunks: list[str] = []
        self.enters = 0

    def type_text(self, text: str, delay_ms: int) -> None:
        if len(self.chunks) == self.fail_after:
            raise ts.InjectionError("nope")
        self.chunks.append(text)

    def enter(self) -> None:
        self.enters += 1


class TestTypeSink:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ("hi\x00\x07\x1b there", "hi there"),
            ("line\n", "line"),
            ("a\nb", "a b"),
            ("a\r\nb", "a b"),
            ("a\tb", "a b"),
            ("日本語 🎉", "日本語 🎉"),
        ],
    )
    def test_sanitize(self, raw: str, clean: str) -> None:
        assert ts.sanitize(raw) == clean

    def test_allow_newlines_keeps_inner_only(self) -> None:
        assert ts.sanitize("a\nb\n", allow_newlines=True) == "a\nb"

    def test_chunks_and_no_enter(self) -> None:
        backend = Recording()
        ts.TypeSink([backend], chunk_size=4).deliver("abcdefghij")
        assert backend.chunks == ["abcd", "efgh", "ij"] and backend.enters == 0

    def test_auto_submit(self) -> None:
        backend = Recording()
        ts.TypeSink([backend], auto_submit=True).deliver("x")
        assert backend.enters == 1

    def test_empty_text_sends_nothing(self) -> None:
        backend = Recording()
        ts.TypeSink([backend], auto_submit=True).deliver("\n\x00")
        assert backend.chunks == [] and backend.enters == 0

    def test_fallback_resumes_without_duplicates(self) -> None:
        first, second = Recording("a", fail_after=1), Recording("b")
        ts.TypeSink([first, second], chunk_size=2).deliver("aabbcc")
        assert first.chunks == ["aa"] and second.chunks == ["bb", "cc"]

    def test_all_backends_fail(self) -> None:
        with pytest.raises(ts.InjectionError, match="a: nope"):
            ts.TypeSink([Recording("a", fail_after=0)]).deliver("x")
        with pytest.raises(ts.InjectionError, match="no text injection"):
            ts.TypeSink([]).deliver("x")

    def test_long_text_prefers_paste(self) -> None:
        typer, paster = Recording("t"), Recording("p", paste=True)
        text = "x" * (ts.PASTE_THRESHOLD + 1)
        ts.TypeSink([typer, paster]).deliver(text)
        assert typer.chunks == [] and paster.chunks == [text]

    def test_short_text_prefers_typing(self) -> None:
        typer, paster = Recording("t"), Recording("p", paste=True)
        ts.TypeSink([typer, paster]).deliver("short")
        assert typer.chunks == ["short"] and paster.chunks == []


def which_of(*tools: str) -> Any:
    return lambda name: f"/usr/bin/{name}" if name in tools else None


class TestBackendSelection:
    def names(self, platform: str, session: str, *tools: str) -> list[str]:
        env = {"XDG_SESSION_TYPE": session}
        return [b.name for b in ts.select_backends(platform, env, which_of(*tools))]

    def test_platforms(self) -> None:
        assert self.names("darwin", "") == ["quartz", "mac-paste"]
        assert self.names("win32", "") == ["sendinput", "windows-paste"]
        assert self.names("linux", "x11", "xdotool") == ["xdotool"]
        assert self.names("linux", "x11") == []

    def test_wayland_chain_order(self) -> None:
        chain = self.names("linux", "wayland", "wtype", "ydotool", "wl-copy", "xdotool")
        assert chain == ["wtype", "ydotool", "wl-copy"]
        assert self.names("linux", "wayland", "wl-copy") == ["wl-copy"]


class TestLinuxCommands:
    @pytest.fixture
    def commands(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake(argv: list[str], **_kwargs: Any) -> Any:
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(ts.subprocess, "run", fake)
        return calls

    def test_xdotool(self, commands: list[list[str]]) -> None:
        ts.XdotoolBackend().type_text("-hi", 4)
        assert commands[0] == [
            "xdotool", "type", "--clearmodifiers", "--delay", "4", "--", "-hi",
        ]  # fmt: skip

    def test_wtype_and_ydotool(self, commands: list[list[str]]) -> None:
        ts.WtypeBackend().type_text("hi", 2)
        ts.WtypeBackend().enter()
        ts.YdotoolBackend().type_text("hi", 2)
        ts.YdotoolBackend().enter()
        assert commands == [
            ["wtype", "-d", "2", "--", "hi"],
            ["wtype", "-k", "Return"],
            ["ydotool", "type", "--key-delay", "2", "--", "hi"],
            ["ydotool", "key", "28:1", "28:0"],
        ]

    def test_wl_copy_uses_available_injector(self, commands: list[list[str]]) -> None:
        ts.WlCopyBackend(which_of("ydotool")).type_text("hi", 2)
        assert commands[0] == ["wl-copy", "--", "hi"] and commands[1][0] == "ydotool"

    def test_wl_copy_alone_tells_the_user(
        self, commands: list[list[str]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        ts.WlCopyBackend(which_of()).type_text("hi", 2)
        assert "Ctrl+V" in capsys.readouterr().out and len(commands) == 1
        with pytest.raises(ts.InjectionError):
            ts.WlCopyBackend(which_of()).enter()

    def test_gnome_wtype_failure_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(argv: list[str], **_kwargs: Any) -> Any:
            code = 1 if argv[0] == "wtype" else 0
            return subprocess.CompletedProcess(argv, code, "", "unsupported")

        monkeypatch.setattr(ts.subprocess, "run", fake)
        ts.TypeSink([ts.WtypeBackend(), ts.YdotoolBackend()]).deliver("hi")

    def test_missing_binary_is_injection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(argv: list[str], **_kwargs: Any) -> Any:
            raise FileNotFoundError(argv[0])

        monkeypatch.setattr(ts.subprocess, "run", fake)
        with pytest.raises(ts.InjectionError):
            ts.XdotoolBackend().type_text("hi", 1)


class FakeQuartz:
    kCGHIDEventTap = 0
    kCGEventFlagMaskCommand = 1 << 20

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def CGEventCreateKeyboardEvent(self, _src: Any, key: int, down: bool) -> dict[str, Any]:
        return {"key": key, "down": down}

    def CGEventKeyboardSetUnicodeString(self, event: Any, units: int, text: str) -> None:
        event.update(units=units, text=text)

    def CGEventSetFlags(self, event: Any, flags: int) -> None:
        event["flags"] = flags

    def CGEventPost(self, _tap: int, event: Any) -> None:
        self.events.append(event)


class TestQuartz:
    def test_unicode_including_surrogates(self) -> None:
        quartz = FakeQuartz()
        ts.QuartzBackend(quartz, lambda: True).type_text("é🎉", 0)
        assert [e["text"] for e in quartz.events if e["down"]] == ["é", "🎉"]
        assert [e["units"] for e in quartz.events if e["down"]] == [1, 2]
        assert len(quartz.events) == 4

    def test_missing_permission_prints_fix(self) -> None:
        with pytest.raises(ts.InjectionError, match="Privacy & Security > Accessibility"):
            ts.QuartzBackend(FakeQuartz(), lambda: False).type_text("x", 0)

    def test_paste_restores_clipboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str | None]] = []

        def fake(argv: list[str], **kwargs: Any) -> Any:
            calls.append((argv[0], kwargs.get("input")))
            return subprocess.CompletedProcess(argv, 0, "previous", "")

        monkeypatch.setattr(ts.subprocess, "run", fake)
        monkeypatch.setattr(ts, "CLIPBOARD_SETTLE_S", 0)
        quartz = FakeQuartz()
        ts.MacPasteBackend(quartz, lambda: True).type_text("new", 0)
        assert calls == [("pbpaste", None), ("pbcopy", "new"), ("pbcopy", "previous")]
        assert quartz.events[0]["key"] == 9
        assert quartz.events[0]["flags"] == quartz.kCGEventFlagMaskCommand

    def test_real_ax_call_returns_bool(self) -> None:
        if sys.platform == "darwin":
            assert isinstance(ts.ax_trusted(), bool)


class TestWindows:
    def test_utf16_surrogates(self) -> None:
        events = ts.WindowsBackend.events_for("🎉")
        assert len(events) == 4
        assert [e[1] for e in events] == [0xD83C, 0xD83C, 0xDF89, 0xDF89]

    def test_blocked_input_raises(self) -> None:
        with pytest.raises(ts.InjectionError, match="elevated"):
            ts.WindowsBackend(lambda _events: 0).type_text("x", 0)

    def test_enter(self) -> None:
        sent: list[Any] = []

        def send(events: list[Any]) -> int:
            sent.extend(events)
            return len(events)

        ts.WindowsBackend(send).enter()
        assert [e[0] for e in sent] == [0x0D, 0x0D]

    def test_paste_restores_clipboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs: list[str | None] = []

        def fake(argv: list[str], text: str | None = None) -> str:
            inputs.append(text)
            return "previous"

        monkeypatch.setattr(ts, "_run", fake)
        monkeypatch.setattr(ts, "CLIPBOARD_SETTLE_S", 0)
        ts.WindowsPasteBackend(lambda events: len(events)).type_text("new", 0)
        assert inputs == [None, "new", "previous"]


def tone(rate: int = 22050) -> tuple[Any, int]:
    return np.ones(rate, np.float32) * 0.1, rate


class FakeTts:
    def synthesize(self, text: str) -> tuple[Any, int]:
        return tone()


class FakeKeyboard:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def press(self, key: Any) -> None:
        self.events.append(("press", key))

    def release(self, key: Any) -> None:
        self.events.append(("release", key))


class TestVirtualMic:
    def test_lead_in_tail_and_rate(self) -> None:
        device = vm.FakeDevice()
        vm.VirtualMicSink(FakeTts(), device).deliver("hi")
        audio = device.played[0]
        assert audio.size == pytest.approx(48000 * 1.9, abs=2)
        assert np.all(audio[: 48000 * 3 // 10 - 50] == 0)
        assert np.all(audio[-48000 * 6 // 10 + 50 :] == 0)

    def test_resample_identity(self) -> None:
        audio = np.ones(10, np.float32)
        assert vm.resample(audio, 48000, 48000) is audio

    def test_soundcard_player_uses_native_rate_all_channels(self, fake_sd: FakeSoundDevice) -> None:
        player = vm.SoundDevicePlayer("BlackHole", fake_sd)
        assert player.rate == 48000
        player.play(np.zeros(100, np.float32))
        played = fake_sd.played[0]
        assert played["data"].shape == (100, 2)
        assert (played["device"], played["samplerate"], played["blocking"]) == (1, 48000, True)

    def test_missing_device_is_reported(self, fake_sd: FakeSoundDevice) -> None:
        with pytest.raises(LookupError):
            vm.SoundDevicePlayer("CABLE Input", fake_sd)

    def test_file_sink(self, tmp_path: Path) -> None:
        sink = vm.FileSink(tmp_path / "out")
        vm.VirtualMicSink(FakeTts(), sink).deliver("hi")
        info = soundfile.info(tmp_path / "out" / "0000.wav")
        assert info.samplerate == 48000 and sink.count == 1

    def test_ptt_hold_wraps_playback(self) -> None:
        keyboard = FakeKeyboard()
        order: list[str] = []

        class Device(vm.FakeDevice):
            def play(self, audio: Any) -> None:
                order.append(f"played after {len(keyboard.events)}")

        ptt = vm.PttAssist("f9", "hold", keyboard)
        vm.VirtualMicSink(FakeTts(), Device(), ptt=ptt).deliver("hi")
        assert order == ["played after 1"]
        assert [e[0] for e in keyboard.events] == ["press", "release"]

    def test_ptt_tap(self) -> None:
        keyboard = FakeKeyboard()
        ptt = vm.PttAssist("x", "tap", keyboard)
        vm.VirtualMicSink(FakeTts(), vm.FakeDevice(), ptt=ptt).deliver("hi")
        assert len(keyboard.events) == 4

    def test_ptt_released_when_playback_fails(self) -> None:
        class Broken(vm.FakeDevice):
            def play(self, audio: Any) -> None:
                raise OSError("device gone")

        keyboard = FakeKeyboard()
        ptt = vm.PttAssist("x", "hold", keyboard)
        with pytest.raises(OSError):
            vm.VirtualMicSink(FakeTts(), Broken(), ptt=ptt).deliver("hi")
        assert keyboard.events[-1][0] == "release"

    def test_monitor_off_by_default_and_optional(self) -> None:
        monitor = vm.FakeDevice(rate=44100)
        vm.VirtualMicSink(FakeTts(), vm.FakeDevice(), monitor=monitor).deliver("hi")
        assert monitor.played[0].size == pytest.approx(44100 * 1.9, abs=2)


class TestTts:
    def test_piper_real(self, real_tts: Any) -> None:
        audio, rate = real_tts.synthesize("Hello there.")
        assert rate == 22050 and audio.dtype == np.float32 and np.abs(audio).max() > 0.01

    def test_piper_speed(self, real_tts: Any) -> None:
        # Piper randomizes durations per call, so single runs overlap; compare averages.
        text = "Testing the speed setting."
        fast = tts.PiperTts(TtsConfig(speed=2.0), real_tts.voice)
        normal = np.mean([real_tts.synthesize(text)[0].size for _ in range(8)])
        assert np.mean([fast.synthesize(text)[0].size for _ in range(8)]) < normal * 0.7

    def test_kokoro(self) -> None:
        model = SimpleNamespace(create=lambda *_a, **_k: (np.zeros(5), 24000))
        audio, rate = tts.KokoroTts(TtsConfig(), model).synthesize("hi")
        assert rate == 24000 and audio.dtype == np.float32

    def _fake_writer(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake(argv: list[str], check: bool) -> None:
            calls.append(argv)
            soundfile.write(argv[2], np.zeros(10), 16000)

        monkeypatch.setattr(tts.subprocess, "run", fake)
        return calls

    def test_macos_say(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._fake_writer(monkeypatch)
        audio, rate = tts.OsTts("darwin").synthesize("-v evil")
        assert calls[0][0] == "say" and calls[0][-2:] == ["--", "-v evil"]
        assert (audio.size, rate) == (10, 16000)

    def test_espeak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._fake_writer(monkeypatch)
        monkeypatch.setattr(tts.shutil, "which", lambda _n: "/usr/bin/espeak-ng")
        tts.OsTts("linux").synthesize("hi")
        assert calls[0][1] == "-w"

    def test_espeak_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tts.shutil, "which", lambda _n: None)
        with pytest.raises(RuntimeError, match="espeak-ng"):
            tts.OsTts("linux").synthesize("hi")

    def test_sapi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Engine:
            def save_to_file(self, _text: str, path: str) -> None:
                self.path = path

            def runAndWait(self) -> None:
                soundfile.write(self.path, np.zeros(4), 16000)

        monkeypatch.setitem(sys.modules, "pyttsx3", SimpleNamespace(init=Engine))
        assert tts.OsTts("win32").synthesize("hi")[0].size == 4

    def test_factory_and_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert isinstance(tts.create_tts(TtsConfig(backend="os")), tts.OsTts)
        monkeypatch.setattr(tts, "KokoroTts", lambda cfg: "kokoro")
        assert tts.create_tts(TtsConfig(backend="kokoro")) == "kokoro"

        def broken(_cfg: TtsConfig) -> Any:
            raise ImportError("no piper")

        monkeypatch.setattr(tts, "PiperTts", broken)
        assert isinstance(tts.create_tts(TtsConfig()), tts.OsTts)


class TestController:
    def make(self, mode: str = "hold") -> tuple[Controller, list[str]]:
        events: list[str] = []
        return (
            Controller(lambda: events.append("start"), lambda: events.append("stop"), mode),
            events,
        )

    def test_hold(self) -> None:
        controller, events = self.make("hold")
        controller.key_down()
        controller.key_down()
        controller.key_up()
        controller.key_up()
        assert events == ["start", "stop"]

    def test_tap(self) -> None:
        controller, events = self.make("tap")
        controller.key_down()
        controller.key_up()
        controller.key_down()
        assert events == ["start", "stop"]

    def test_vad_mode_ignores_hotkey(self) -> None:
        controller, events = self.make("vad")
        controller.key_down()
        assert events == []

    def test_listener_ignores_auto_repeat_and_other_keys(self) -> None:
        class Keyboard:
            Key = SimpleNamespace(alt_r="alt_r")
            KeyCode = SimpleNamespace(from_char=lambda c: c)

            class Listener:
                def __init__(self, on_press: Any, on_release: Any) -> None:
                    self.on_press, self.on_release = on_press, on_release

        controller, events = self.make("tap")
        listener = HotkeyListener("alt_r", controller, Keyboard)
        listener._press("a")
        listener._press("alt_r")
        listener._press("alt_r")
        listener._release("alt_r")
        assert events == ["start"]

    def test_socket_commands(self) -> None:
        controller, events = self.make()
        server = SocketServer(controller, Path("s.sock"))
        server.start()
        try:
            assert (Path("s.sock")).stat().st_mode & 0o777 == 0o600
            assert send_command("ptt-start", Path("s.sock")) == "ok"
            assert send_command("ptt-stop", Path("s.sock")) == "ok"
            assert send_command("toggle", Path("s.sock")) == "ok"
        finally:
            server.stop()
        assert events == ["start", "stop", "start"]
        assert not (Path("s.sock")).exists()

    def test_unknown_commands_rejected(self) -> None:
        controller, events = self.make()
        assert controller.command("rm -rf") is False and events == []
        with pytest.raises(ValueError):
            send_command("nope")

    def test_stale_socket_replaced(self) -> None:
        path = Path("s.sock")
        path.write_text("stale")
        server = SocketServer(self.make()[0], path)
        server.start()
        server.stop()

    def test_wayland_detection(self) -> None:
        assert is_wayland({"XDG_SESSION_TYPE": "wayland"})
        assert not is_wayland({"XDG_SESSION_TYPE": "x11"}) and not is_wayland({})


class TestStatus:
    def test_terminal(self) -> None:
        stream = io.StringIO()
        indicator = StatusIndicator(stream)
        indicator.set("recording")
        indicator.set("idle", "ready")
        assert "recording" in stream.getvalue() and "idle: ready" in stream.getvalue()

    def test_tray(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TrayIndicator, "_image", staticmethod(lambda state: state))
        icon = SimpleNamespace(icon=None)
        TrayIndicator(io.StringIO(), icon).set("error")
        assert icon.icon == "error"


def tone_1k() -> Any:
    return (0.1 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)).astype(np.float32)


class TestApp:
    def fake_pipeline(self, stub_vad: StubVad, sinks: list[Any]) -> Pipeline:
        class Asr:
            name = "fake"

            def transcribe(self, _audio: Any) -> Transcript:
                return Transcript("hello")

        return Pipeline(
            Asr(), HallucinationGuard(stub_vad), DefaultPostProcessor(PostprocessConfig()), sinks
        )

    def test_build_sinks_by_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app, "create_tts", lambda cfg: FakeTts())
        monkeypatch.setattr(app, "SoundDevicePlayer", lambda device: vm.FakeDevice())
        assert [type(s).__name__ for s in app.build_sinks(Config())] == ["TypeSink"]
        assert [type(s).__name__ for s in app.build_sinks(Config(output_mode="virtual_mic"))] == [
            "VirtualMicSink"
        ]
        both = Config(output_mode="both", ptt_assist_key="x", monitor_device="m")
        assert len(app.build_sinks(both)) == 2

    def test_run_stops_with_instructions_when_permission_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def no_models(*_args: Any) -> Any:
            raise AssertionError("models must not load")

        monkeypatch.setattr(app.sys, "platform", "darwin")
        monkeypatch.setattr(app, "request_macos_permissions", lambda: ["Accessibility"])
        monkeypatch.setattr(app, "build_pipeline", no_models)
        with pytest.raises(SystemExit):
            app.run(Config())
        assert "macOS needs Accessibility" in capsys.readouterr().err

    def test_session_flow_and_no_audio_saved(self, stub_vad: StubVad, isolated_data: Path) -> None:
        delivered: list[str] = []
        sink = SimpleNamespace(deliver=delivered.append)
        stream = io.StringIO()
        session = app.Session(
            Config(), self.fake_pipeline(stub_vad, [sink]), StatusIndicator(stream)
        )
        session.worker.start()
        session.submit(tone_1k())
        session.close()
        assert delivered == ["hello"]
        assert not (isolated_data / "recordings").exists()
        assert "hello" not in stream.getvalue()  # dictated text never reaches the log

    def test_save_audio_opt_in(self, stub_vad: StubVad, isolated_data: Path) -> None:
        cfg = Config()
        cfg.privacy.save_audio = True
        session = app.Session(cfg, self.fake_pipeline(stub_vad, []), StatusIndicator(io.StringIO()))
        session.process(np.full(16000, 0.1, np.float32))
        assert len(list((isolated_data / "recordings").glob("*.wav"))) == 1

    def test_injection_failure_is_reported(self, stub_vad: StubVad) -> None:
        def fail(_text: str) -> None:
            raise ts.InjectionError("denied")

        stream = io.StringIO()
        session = app.Session(
            Config(),
            self.fake_pipeline(stub_vad, [SimpleNamespace(deliver=fail)]),
            StatusIndicator(stream),
        )
        session.process(tone_1k())
        assert "error: denied" in stream.getvalue()


class TestDoctor:
    def test_mic(self) -> None:
        class Stream:
            def __enter__(self) -> Stream:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

        assert doctor.check_mic(SimpleNamespace(InputStream=lambda **_k: Stream())).ok

        def denied(**_kwargs: Any) -> Any:
            raise OSError("denied")

        result = doctor.check_mic(SimpleNamespace(InputStream=denied))
        assert not result.ok and "privacy" in result.detail

    def test_virtual_device_fix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(*_args: Any) -> Any:
            raise LookupError("none")

        monkeypatch.setattr(doctor, "find_device", missing)
        result = doctor.check_virtual_device(Config())
        assert not result.ok and "setup-virtual-mic" in result.detail

    def test_models_missing_then_good_then_corrupt(self) -> None:
        cfg = Config()
        assert not any(c.ok for c in doctor.check_models(cfg))
        for path in (whisper_dir(cfg.asr.model), piper_voice_dir(cfg.tts.voice)):
            path.mkdir(parents=True)
            (path / MANIFEST).write_text("{}")
        assert all(c.ok for c in doctor.check_models(cfg))
        (whisper_dir(cfg.asr.model) / MANIFEST).write_text(json.dumps({"gone.bin": "0"}))
        assert not doctor.check_models(cfg)[0].ok

    def test_wayland(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/usr/bin/x")
        monkeypatch.setattr(doctor.os, "access", lambda *_a: False)
        monkeypatch.setattr(doctor, "select_backends", lambda *_a: [object()])
        checks = doctor.check_wayland({"XDG_SESSION_TYPE": "wayland"})
        assert [c.name for c in checks if not c.ok] == ["/dev/uinput access"]

    def test_macos_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "ax_trusted", lambda: False)
        monkeypatch.setattr(doctor, "input_monitoring_granted", lambda: True)
        accessibility, monitoring = doctor.check_macos()
        assert not accessibility.ok and "Accessibility" in accessibility.detail and monitoring.ok

    def test_request_reports_only_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "prompt_accessibility", lambda: True)
        monkeypatch.setattr(doctor, "prompt_input_monitoring", lambda: False)
        assert doctor.request_macos_permissions() == ["Input Monitoring"]

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS frameworks")
    def test_macos_symbols_resolve_without_prompting(self) -> None:
        import ctypes

        ax = ctypes.cdll.LoadLibrary(ts._AX_LIB)
        cf = ctypes.cdll.LoadLibrary(ts._CF_LIB)
        assert ax.AXIsProcessTrustedWithOptions and cf.CFDictionaryCreate and cf.CFRelease
        assert ctypes.c_void_p.in_dll(ax, "kAXTrustedCheckOptionPrompt").value
        assert ctypes.c_void_p.in_dll(cf, "kCFBooleanTrue").value
        assert ctypes.cdll.LoadLibrary(doctor._CORE_GRAPHICS).CGRequestListenEventAccess

    def test_gpu_absence_is_not_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "cuda_available", lambda: False)
        assert doctor.check_gpu().ok

    def test_real_cuda_probe_runs(self) -> None:
        assert isinstance(wh.cuda_available(), bool)

    def test_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        failures = doctor.report([doctor.Check("a", True), doctor.Check("b", False, "fix it")])
        output = capsys.readouterr().out
        assert failures == 1 and "[ok] a" in output and "[FAIL] b - fix it" in output

    def test_run_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        good = doctor.Check("x", True)
        monkeypatch.setattr(doctor, "check_mic", lambda: good)
        monkeypatch.setattr(doctor, "check_virtual_device", lambda cfg: good)
        monkeypatch.setattr(doctor, "check_macos", lambda: [good])
        monkeypatch.setattr(doctor, "list_devices", lambda: [1])
        assert len(doctor.run_checks(Config())) >= 5


def write_clip(path: Path, value: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, np.full(16000, value, np.float32), 16000)


class EchoPipeline:
    def __init__(self, text: str = "hello world") -> None:
        self.text = text

    def process(self, _audio: Any) -> Any:
        return SimpleNamespace(text=self.text, timings={"asr": 10.0, "guard": 1.0})


class TestBench:
    def test_wer_is_real(self) -> None:
        assert wer(["Hello, World."], ["hello world"]) == 0
        assert wer(["the quick brown fox"], ["a slow brown cat"]) == pytest.approx(0.75)
        assert wer(["two words"], [""]) == 1

    def test_load_set_and_run(self, tmp_path: Path) -> None:
        write_clip(tmp_path / "a.wav")
        (tmp_path / "transcripts.json").write_text(json.dumps({"a.wav": "hello world"}))
        result = run_set("x", load_set(tmp_path), EchoPipeline())  # type: ignore[arg-type]
        assert result.wer == 0 and result.output_chars == 11

    def test_set_without_transcripts_has_no_wer(self, tmp_path: Path) -> None:
        write_clip(tmp_path / "a.wav")
        assert run_set("x", load_set(tmp_path), EchoPipeline()).wer is None  # type: ignore[arg-type]

    def test_format(self) -> None:
        text = format_results(
            [SetResult("normal", 2, 0.05, 10, {"asr": [10.0, 20.0]}), SetResult("silence", 3)]
        )
        assert "5.00%" in text and "n/a" in text and "p95" in text and "asr" in text

    def test_missing_fixtures(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="make_fixtures"):
            bench_fixtures(tmp_path, EchoPipeline())  # type: ignore[arg-type]

    def test_personal_ab(self) -> None:
        for i in range(10):
            clip = fdata.enroll_dir() / f"{i}.wav"
            write_clip(clip)
            fdata.append_sample(fdata.Sample(clip, "hello world"))
        ftrain.finetuned_dir().mkdir(parents=True)
        models_seen: list[str] = []

        def make(cfg: Config) -> EchoPipeline:
            models_seen.append(cfg.asr.model)
            return EchoPipeline()

        results = bench_personal(Config(), make)
        assert [r.name.split(":")[0] for r in results] == ["base", "fine-tuned"]
        assert results[0].clips == 2  # the held-out fifth, never training data
        assert models_seen[1] == str(ftrain.finetuned_dir())

    def test_personal_without_recordings(self) -> None:
        with pytest.raises(FileNotFoundError, match="enroll"):
            bench_personal(Config(), lambda _c: None)


class TestFinetune:
    def test_prompts(self) -> None:
        assert len(PROMPTS) >= 100 and len(set(PROMPTS)) == len(PROMPTS)
        assert len(SENTENCES) >= 100 and "pytest" in CODING_VOCABULARY

    def recorder(self, levels: list[float]) -> Recorder:
        pending = list(levels)
        recorder = Recorder(stream_factory=FakeStream)
        recorder.start = lambda: None  # type: ignore[method-assign]
        recorder.stop = lambda: np.full(16000, pending.pop(0), np.float32)  # type: ignore[method-assign]
        return recorder

    def test_enroll_saves_and_resumes(self) -> None:
        answers = iter(["", "", "q"])
        saved = enroll(
            self.recorder([0.01]), ["one", "two"], lambda _p: next(answers), lambda _s: None
        )
        assert saved == 1
        assert [s.text for s in fdata.load_enrollment()] == ["one"]
        answers = iter(["", ""])
        enroll(self.recorder([0.01]), ["one", "two"], lambda _p: next(answers), lambda _s: None)
        assert [s.text for s in fdata.load_enrollment()] == ["one", "two"]
        assert (fdata.enroll_dir() / "0001.wav").exists()

    def test_level_check_prompts_retake(self) -> None:
        Calibration(-70, -40).save()
        said: list[str] = []
        recorder = self.recorder([0.9, 0.01])
        enroll(recorder, ["one"], lambda _p: "", said.append)
        assert any("too loud" in s for s in said)
        assert len(fdata.load_enrollment()) == 1

    def test_scan_corpus(self, tmp_path: Path) -> None:
        write_clip(tmp_path / "s1" / "a.wav")
        (tmp_path / "s1" / "a.txt").write_text(" hello \n")
        write_clip(tmp_path / "s1" / "no_transcript.wav")
        assert [s.text for s in fdata.scan_corpus(tmp_path)] == ["hello"]

    def test_split_never_overlaps(self) -> None:
        samples = [fdata.Sample(Path(f"{i}.wav"), str(i)) for i in range(20)]
        train, test = fdata.split(samples)
        assert len(test) == 4 and len(train) == 16
        assert not {s.text for s in train} & {s.text for s in test}

    def test_missing_audio_skipped(self) -> None:
        fdata.enroll_dir().mkdir(parents=True)
        fdata.append_sample(fdata.Sample(fdata.enroll_dir() / "gone.wav", "x"))
        assert fdata.load_enrollment() == []

    def test_lora_targets(self) -> None:
        settings = ftrain.lora_settings(ftrain.TrainConfig())
        assert settings["target_modules"] == ["q_proj", "v_proj"] and settings["r"] == 8

    def test_train_requires_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("torch", "peft", "transformers"):
            monkeypatch.setitem(sys.modules, name, SimpleNamespace(
                LoraConfig=None, get_peft_model=None,
                WhisperForConditionalGeneration=None, WhisperProcessor=None,
            ))  # fmt: skip
        with pytest.raises(ValueError, match="enroll"):
            ftrain.train([], ftrain.TrainConfig())

    @pytest.mark.parametrize(
        ("cuda", "mps", "device"),
        [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
    )
    def test_device(self, cuda: bool, mps: bool, device: str) -> None:
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        )
        assert ftrain.pick_device(torch) == device

    def test_export_converts_and_cleans_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        class Converter:
            def __init__(self, path: str, copy_files: list[str]) -> None:
                seen["files"] = copy_files
                assert Path(path).is_dir()
                assert seen["feature_extractor_saved"]

            def convert(self, out: str, quantization: str, force: bool) -> None:
                seen["quantization"] = quantization

        converters = SimpleNamespace(TransformersConverter=Converter)
        monkeypatch.setitem(sys.modules, "ctranslate2.converters", converters)

        def save(path: Any) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

        def save_extractor(_path: Any) -> None:
            seen["feature_extractor_saved"] = True

        saver = SimpleNamespace(
            save_pretrained=save,
            feature_extractor=SimpleNamespace(save_pretrained=save_extractor),
        )
        ftrain.export_ct2(saver, saver)
        assert seen["quantization"] == "int8" and "tokenizer.json" in seen["files"]
        assert not (data_dir() / "finetune-merged").exists()


class TestPlugins:
    def test_vsr_stub_refuses(self) -> None:
        with pytest.raises(NotImplementedError):
            WebcamVsrStub().start()
        with pytest.raises(NotImplementedError):
            WebcamVsrStub().stop()

    def test_sonar_is_readme_only(self) -> None:
        import silentdictate.plugins

        base = Path(silentdictate.plugins.__file__).parent / "sonar_experimental"
        assert [p.name for p in base.iterdir()] == ["README.md"]


SPEC_COMMANDS = (
    "run devices calibrate enroll finetune bench setup-virtual-mic doctor toggle ptt-start "
    "ptt-stop purge-data"
).split()


class TestCli:
    def test_help_lists_every_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        output = capsys.readouterr().out
        for command in SPEC_COMMANDS + ["download-models"]:
            assert command in output

    def test_devices(self, monkeypatch: pytest.MonkeyPatch, fake_sd: FakeSoundDevice) -> None:
        monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
        assert cli.main(["devices"]) == 0

    def test_bad_override(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["--set-config", "mode=never", "devices"]) == 2
        assert "mode must be" in capsys.readouterr().err

    def test_control_without_server(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["toggle"]) == 1
        assert "silentdictate run" in capsys.readouterr().err

    def test_control_with_server(self) -> None:
        events: list[str] = []
        server = SocketServer(Controller(lambda: events.append("start"), lambda: None))
        server.start()
        try:
            assert cli.main(["ptt-start"]) == 0
        finally:
            server.stop()
        assert events == ["start"]

    def test_purge_keeps_models_unless_asked(self, isolated_data: Path) -> None:
        write_clip(isolated_data / "enroll" / "a.wav")
        write_clip(isolated_data / "recordings" / "b.wav")
        (isolated_data / "calibration.json").write_text("{}")
        (isolated_data / "models").mkdir()
        assert cli.main(["purge-data", "--yes"]) == 0
        assert [p.name for p in isolated_data.iterdir()] == ["models"]
        assert cli.main(["purge-data", "--yes", "--models"]) == 0
        assert list(isolated_data.iterdir()) == []

    def test_purge_needs_confirmation(
        self, isolated_data: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_clip(isolated_data / "enroll" / "a.wav")
        monkeypatch.setattr("builtins.input", lambda _p: "no")
        assert cli.main(["purge-data"]) == 1
        assert (isolated_data / "enroll" / "a.wav").exists()

    def test_purge_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["purge-data", "--yes"]) == 0
        assert "Nothing" in capsys.readouterr().out

    def test_calibrate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        levels = iter([0.001, 0.01])

        def capture(_self: Recorder, seconds: float, _sleep: Any) -> Any:
            return np.full(1000, next(levels), np.float32)

        monkeypatch.setattr(Recorder, "open", lambda _s: None)
        monkeypatch.setattr(Recorder, "capture", capture)
        monkeypatch.setattr("builtins.input", lambda _p: "")
        assert cli.main(["calibrate"]) == 0
        saved = Calibration.load()
        assert saved is not None and saved.whisper_db > saved.noise_db

    def test_calibrate_rejects_no_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Recorder, "open", lambda _s: None)
        monkeypatch.setattr(Recorder, "capture", lambda _s, _t, _f: np.full(1000, 0.01, np.float32))
        monkeypatch.setattr("builtins.input", lambda _p: "")
        assert cli.main(["calibrate"]) == 1
        assert Calibration.load() is None

    def test_setup_virtual_mic_prints_only_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["setup-virtual-mic"]) == 0
        assert "setup_virtual_mic" in capsys.readouterr().out

    def test_bench_fixtures_uses_no_sinks(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        seen: dict[str, Any] = {}

        def build(cfg: Config, sinks: Any = None) -> EchoPipeline:
            seen.update(model=cfg.asr.model, device=cfg.asr.device, sinks=sinks)
            return EchoPipeline()

        monkeypatch.setattr(app, "build_pipeline", build)
        for name in ("normal", "quiet", "silence"):
            write_clip(tmp_path / name / "a.wav")
        assert cli.main(["bench", "--fixtures-dir", str(tmp_path)]) == 0
        assert seen == {"model": "small", "device": "cpu", "sinks": []}
        assert "WER" in capsys.readouterr().out

    def test_config_init(self, tmp_path: Path) -> None:
        assert cli.main(["config-init"]) == 0
        assert (tmp_path / "config.toml").exists()

    def test_doctor_shows_permission_help(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.setattr(doctor, "run_checks", lambda cfg: [])
        monkeypatch.setattr(doctor, "request_macos_permissions", lambda: ["Input Monitoring"])
        assert cli.main(["doctor"]) == 0
        assert "macOS needs Input Monitoring" in capsys.readouterr().out

    def test_doctor_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "request_macos_permissions", lambda: [])
        monkeypatch.setattr(doctor, "run_checks", lambda cfg: [doctor.Check("x", False)])
        assert cli.main(["doctor"]) == 1


def test_recorder_threads_do_not_leak() -> None:
    before = threading.active_count()
    server = SocketServer(Controller(lambda: None, lambda: None))
    server.start()
    server.stop()
    assert threading.active_count() <= before + 1
