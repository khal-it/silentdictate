"""Unit tests for config, audio, preprocessing, VAD, models, guard, postprocess, pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from conftest import FakeSoundDevice, FakeStream, StubVad

from silentdictate import audio_in, models
from silentdictate.asr.base import Transcript
from silentdictate.asr.guard import HallucinationGuard
from silentdictate.audio_in import (
    Calibration,
    Recorder,
    find_device,
    list_devices,
    meter,
    rms_db,
    volume_feedback,
)
from silentdictate.config import (
    Config,
    ConfigError,
    PostprocessConfig,
    config_path,
    data_dir,
    default_virtual_mic,
    load_config,
    save_config,
)
from silentdictate.pipeline import Pipeline
from silentdictate.postprocess import (
    DefaultPostProcessor,
    LlmCleanup,
    apply_replacements,
    apply_spoken_punctuation,
)
from silentdictate.preprocess import preprocess
from silentdictate.vad import VadSegmenter, longest_speech_ms, speech_probs


class TestConfig:
    def test_defaults_match_spec(self) -> None:
        cfg = load_config()
        assert cfg.mode == "hold" and cfg.output_mode == "type" and cfg.auto_submit is False
        assert cfg.postprocess.llm_cleanup is False and cfg.privacy.save_audio is False
        assert cfg.asr.backend == "whisper" and cfg.tts.backend == "piper"

    def test_round_trip_and_overrides(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        cfg = Config()
        cfg.asr.hotwords = ["pytest"]
        save_config(cfg, path)
        loaded = load_config(path, ["asr.model=small", "asr.beam_size=2", "auto_submit=true"])
        assert loaded.asr.hotwords == ["pytest"]
        assert (loaded.asr.model, loaded.asr.beam_size, loaded.auto_submit) == ("small", 2, True)

    @pytest.mark.parametrize(
        "bad", ["nope=1", "asr.nope=1", "mode=never", "asr.device=tpu", "broken"]
    )
    def test_invalid_rejected(self, bad: str) -> None:
        with pytest.raises(ConfigError):
            load_config(overrides=[bad])

    def test_bad_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text("= broken")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_paths_use_overrides(self, isolated_data: Path) -> None:
        assert data_dir() == isolated_data
        assert config_path().name == "config.toml"

    @pytest.mark.parametrize(
        ("platform", "name"),
        [("darwin", "BlackHole 2ch"), ("win32", "CABLE Input"), ("linux", "silentdict-sink")],
    )
    def test_virtual_mic_defaults(self, platform: str, name: str) -> None:
        assert default_virtual_mic(platform) == name


class TestAudioIn:
    def test_device_listing_and_selection(self, fake_sd: FakeSoundDevice) -> None:
        assert [d.name for d in list_devices(fake_sd)] == ["Built-in Mic", "BlackHole 2ch"]
        assert find_device("built-in", "input", fake_sd).index == 0
        assert find_device("1", "output", fake_sd).name == "BlackHole 2ch"

    def test_device_missing_or_wrong_kind(self, fake_sd: FakeSoundDevice) -> None:
        with pytest.raises(LookupError, match="BlackHole"):
            find_device("Built-in", "output", fake_sd)
        with pytest.raises(LookupError):
            find_device("9", "input", fake_sd)

    def test_levels(self) -> None:
        assert rms_db(np.ones(10, np.float32)) == pytest.approx(0)
        assert rms_db(np.zeros(10, np.float32)) == float("-inf")
        assert rms_db(np.zeros(0, np.float32)) == float("-inf")
        assert meter(0.0, 10) == "#" * 10 and meter(float("-inf"), 10) == "-" * 10

    @pytest.mark.parametrize(
        ("level", "verdict"), [(-40, "too loud"), (-50, "ok"), (-65, "too quiet")]
    )
    def test_feedback(self, level: float, verdict: str) -> None:
        assert volume_feedback(level, Calibration(-70, -50)) == verdict

    def test_feedback_uncalibrated(self) -> None:
        assert volume_feedback(-10, None) == "ok"

    def test_calibration_round_trip(self) -> None:
        assert Calibration.load() is None
        Calibration(-70, -50).save()
        assert Calibration.load() == Calibration(-70, -50)

    def test_recorder_keeps_only_preroll(self) -> None:
        streams: list[FakeStream] = []

        def factory(**kwargs: Any) -> FakeStream:
            streams.append(FakeStream(**kwargs))
            return streams[-1]

        recorder = Recorder("mic", preroll_ms=500, stream_factory=factory)
        recorder.open()
        stream = streams[0]
        assert stream.started and stream.kwargs["samplerate"] == 16000
        stream.push(0.1, blocks=100)  # far more than the 16-block ring
        recorder.start()
        stream.push(0.5, blocks=2)
        audio = recorder.stop()
        assert audio.size == 512 * (16 + 2)
        assert audio.dtype == np.float32
        stream.push(0.9)  # after stop this must not land in the utterance
        recorder.start()
        assert recorder.stop().max() == pytest.approx(0.9)
        recorder.close()
        assert stream.closed

    def test_recorder_block_tap_and_capture(self) -> None:
        seen: list[Any] = []
        streams: list[FakeStream] = []

        def factory(**kwargs: Any) -> FakeStream:
            streams.append(FakeStream(**kwargs))
            return streams[-1]

        recorder = Recorder(stream_factory=factory, on_block=seen.append)
        recorder.open()
        audio = recorder.capture(1, lambda _s: streams[0].push(0.2))
        assert len(seen) == 1 and audio.size == 512

    def test_no_audio_written(self, isolated_data: Path) -> None:
        recorder = Recorder(stream_factory=FakeStream)
        recorder.open()
        recorder.close()
        assert not isolated_data.exists()

    def test_sample_rate_is_16k(self) -> None:
        assert audio_in.SAMPLE_RATE == 16000


class TestPreprocess:
    def test_normalizes_peak(self) -> None:
        t = np.arange(16000) / 16000
        audio = (0.001 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = preprocess(audio)
        assert 20 * np.log10(np.abs(out).max()) == pytest.approx(-3, abs=0.1)

    def test_removes_rumble(self) -> None:
        t = np.arange(16000) / 16000
        rumble = (0.5 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)
        tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        mixed = preprocess(rumble + tone)
        spectrum = np.abs(np.fft.rfft(mixed[4000:]))
        freqs = np.fft.rfftfreq(mixed[4000:].size, 1 / 16000)
        assert spectrum[np.argmin(abs(freqs - 20))] < spectrum[np.argmin(abs(freqs - 1000))] / 50

    def test_gain_is_capped_and_empty_safe(self) -> None:
        assert preprocess(np.zeros(0, np.float32)).size == 0
        assert np.all(preprocess(np.zeros(100, np.float32)) == 0)
        tiny = np.full(1000, 1e-9, np.float32)
        tiny[::2] *= -1
        assert np.abs(preprocess(tiny)).max() < 1e-5


class TestVad:
    def test_probs_ignore_partial_frame(self, stub_vad: StubVad) -> None:
        assert speech_probs(np.zeros(700, np.float32), stub_vad).size == 1
        assert speech_probs(np.zeros(100, np.float32), stub_vad).size == 0

    def test_longest_run(self, stub_vad: StubVad) -> None:
        audio = np.zeros(512 * 10, np.float32)
        audio[512 : 512 * 4] = 0.5
        assert longest_speech_ms(audio, stub_vad, 0.5) == pytest.approx(96)

    def test_segmenter_keeps_preroll_and_ends_on_silence(self, stub_vad: StubVad) -> None:
        seg = VadSegmenter(stub_vad, preroll_ms=64, min_silence_ms=96, min_speech_ms=64)
        quiet = np.zeros(512 * 5, np.float32)
        speech = np.full(512 * 5, 0.5, np.float32)
        assert seg.feed(quiet) == []
        assert seg.feed(speech) == []
        finished = seg.feed(quiet)
        assert len(finished) == 1
        assert finished[0].size == 512 * (2 + 5 + 3)

    def test_segmenter_drops_blips(self, stub_vad: StubVad) -> None:
        seg = VadSegmenter(stub_vad, min_silence_ms=64, min_speech_ms=300)
        audio = np.zeros(512 * 8, np.float32)
        audio[:512] = 0.5
        assert seg.feed(audio) == []

    def test_segmenter_caps_length(self, stub_vad: StubVad) -> None:
        seg = VadSegmenter(stub_vad, max_utterance_s=0.5)
        finished = seg.feed(np.full(512 * 40, 0.5, np.float32))
        assert [f.size for f in finished] == [512 * 14, 512 * 28]


def _lfs(digest: str) -> SimpleNamespace:
    return SimpleNamespace(sha256=digest)


class FakeApi:
    def __init__(self, siblings: list[SimpleNamespace]) -> None:
        self.siblings = siblings

    def model_info(self, _repo: str, files_metadata: bool) -> SimpleNamespace:
        assert files_metadata
        return SimpleNamespace(sha="abc", siblings=self.siblings)


class TestModels:
    @pytest.fixture
    def repo(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
        files = {"model.bin": b"weights", "config.json": b"{}"}

        def snapshot(_repo: str, revision: str, local_dir: Path, allow_patterns: Any) -> None:
            assert revision == "abc"
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            for name, content in files.items():
                (Path(local_dir) / name).write_bytes(content)

        monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
        return files

    def _api(self, weights_digest: str | None = None) -> FakeApi:
        blob = hashlib.sha1(b"blob 2\0{}", usedforsecurity=False).hexdigest()
        digest = weights_digest or hashlib.sha256(b"weights").hexdigest()
        return FakeApi(
            [
                SimpleNamespace(rfilename="model.bin", lfs=_lfs(digest), blob_id=""),
                SimpleNamespace(rfilename="config.json", lfs=None, blob_id=blob),
                SimpleNamespace(rfilename="absent.txt", lfs=None, blob_id="x"),
            ]
        )

    def test_fetch_and_verify(self, tmp_path: Path, repo: dict[str, bytes]) -> None:
        dest = tmp_path / "m"
        models.fetch("r", dest, api=self._api())
        assert models.is_cached(dest) and models.verify(dest) == []
        (dest / "model.bin").write_bytes(b"tampered")
        assert models.verify(dest) == ["model.bin"]

    def test_checksum_mismatch_deletes_file(self, tmp_path: Path, repo: dict[str, bytes]) -> None:
        dest = tmp_path / "m"
        with pytest.raises(models.ModelError, match="checksum"):
            models.fetch("r", dest, api=self._api("0" * 64))
        assert not (dest / "model.bin").exists() and not models.is_cached(dest)

    def test_unknown_model_and_local_path(self, tmp_path: Path) -> None:
        with pytest.raises(models.ModelError):
            models.ensure_whisper("gpt-99")
        assert models.ensure_whisper(str(tmp_path)) == tmp_path

    def test_cached_model_needs_no_network(self, block_network: None) -> None:
        dest = models.whisper_dir("tiny")
        dest.mkdir(parents=True)
        (dest / models.MANIFEST).write_text("{}")
        assert models.ensure_whisper("tiny") == dest

    def test_parakeet_repo_and_patterns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_fetch(repo: str, dest: Path, patterns: list[str]) -> None:
            seen.update(repo=repo, patterns=patterns)

        monkeypatch.setattr(models, "fetch", fake_fetch)
        models.ensure_parakeet("nemo-parakeet-tdt-0.6b-v3")
        assert seen["repo"] == "istupakov/parakeet-tdt-0.6b-v3-onnx"
        assert "*.int8.onnx" in seen["patterns"]

    def test_verify_uncached(self, tmp_path: Path) -> None:
        assert models.verify(tmp_path) == [models.MANIFEST]

    def test_piper_paths(self) -> None:
        assert models.piper_patterns("en_US-lessac-medium")[0] == (
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        )

    def test_all_spec_models_known(self) -> None:
        assert {"tiny", "base", "small", "large-v3", "large-v3-turbo", "distil-large-v3.5"} <= set(
            models.WHISPER_REPOS
        )


class TestGuard:
    def guard(self, stub_vad: StubVad) -> HallucinationGuard:
        return HallucinationGuard(stub_vad, 0.5)

    def test_rejects_digital_silence(self, stub_vad: StubVad) -> None:
        silent = np.zeros(16000, np.float32)
        assert not self.guard(stub_vad).has_speech(silent, silent)

    def test_rejects_no_vad_speech(self, stub_vad: StubVad) -> None:
        raw = np.full(16000, 1e-3, np.float32)
        assert not self.guard(stub_vad).has_speech(raw, raw)

    def test_faint_speech_passes_on_normalized_audio(self, stub_vad: StubVad) -> None:
        raw = np.full(16000, 1e-3, np.float32)
        assert self.guard(stub_vad).has_speech(raw, raw * 500)

    @pytest.mark.parametrize(
        "text",
        ["Thank you.", "Thanks for watching!", "you", "[Music]", "(silence)", "♪♪", "  ", "BYE."],
    )
    def test_known_phantoms(self, stub_vad: StubVad, text: str) -> None:
        assert self.guard(stub_vad).clean(text) == ""

    def test_real_text_survives(self, stub_vad: StubVad) -> None:
        assert self.guard(stub_vad).clean("Thank you for the review [Music]") == (
            "Thank you for the review"
        )

    def test_no_speech_probability(self, stub_vad: StubVad) -> None:
        assert self.guard(stub_vad).clean("real words", 0.95) == ""


class TestPostprocess:
    @pytest.mark.parametrize(
        ("spoken", "expected"),
        [
            ("hello comma world period", "hello, world."),
            ("is it done question mark", "is it done?"),
            ("first line new line second line", "first line\nsecond line"),
            ("one new paragraph two", "one\n\ntwo"),
            ("stop exclamation point", "stop!"),
            ("call open paren x close paren", "call(x)"),
            ("Hello, comma, world. Period.", "Hello, world."),
        ],
    )
    def test_spoken_punctuation(self, spoken: str, expected: str) -> None:
        assert apply_spoken_punctuation(spoken) == expected

    @pytest.mark.parametrize(
        "text", ["a period of time passed", "the colon is an organ", "version 3.11 works"]
    )
    def test_ordinary_words_kept(self, text: str) -> None:
        assert apply_spoken_punctuation(text) == text

    def test_replacements_whole_phrase(self) -> None:
        table = {"pie test": "pytest", "c": "C++"}
        assert apply_replacements("run Pie Test on c code", table) == "run pytest on C++ code"
        assert apply_replacements("cat", table) == "cat"
        assert apply_replacements("x", {"x": r"\1"}) == r"\1"

    def test_default_has_no_network(self, block_network: None) -> None:
        processor = DefaultPostProcessor(PostprocessConfig(replacements={"foo": "bar"}))
        assert processor.cleanup is None
        assert processor.process("foo period") == "bar."

    def _client(self, handler: Any) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_llm_cleanup(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path == "/v1/chat/completions"
            assert "never add" in body["messages"][0]["content"].lower()
            reply = {"choices": [{"message": {"content": "Hello, world."}}]}
            return httpx.Response(200, json=reply)

        clean = LlmCleanup("http://127.0.0.1:11434/v1", "m", self._client(handler))
        assert clean("hello world") == "Hello, world."

    def test_llm_added_content_rejected(self) -> None:
        reply = {"choices": [{"message": {"content": "Sure! Here is a long answer for you."}}]}
        clean = LlmCleanup(
            "http://localhost:1/v1", "m", self._client(lambda _r: httpx.Response(200, json=reply))
        )
        assert clean("hello world") == "hello world"

    @pytest.mark.parametrize("status", [500, 200])
    def test_llm_failures_keep_text(self, status: int) -> None:
        clean = LlmCleanup(
            "http://localhost:1/v1",
            "m",
            self._client(lambda _r: httpx.Response(status, json={})),
        )
        assert clean("hello") == "hello"

    def test_llm_refused_connection_keeps_text(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        assert LlmCleanup("http://[::1]:1/v1", "m", self._client(handler))("hi") == "hi"

    def test_llm_must_be_local(self) -> None:
        with pytest.raises(ValueError, match="local"):
            LlmCleanup("https://api.example.com/v1", "m")

    def test_llm_enabled_through_config(self) -> None:
        assert DefaultPostProcessor(PostprocessConfig(llm_cleanup=True)).cleanup is not None


class FakeAsr:
    name = "fake"

    def __init__(self, text: str = "hello world period", no_speech: float = 0.0) -> None:
        self.text = text
        self.no_speech = no_speech
        self.calls = 0

    def transcribe(self, audio: Any) -> Transcript:
        self.calls += 1
        return Transcript(self.text, self.no_speech)


class FakeSink:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def deliver(self, text: str) -> None:
        self.texts.append(text)


def speech_tone() -> Any:
    """A 1 kHz tone; constant values would be removed by the high-pass filter."""
    return (0.1 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)).astype(np.float32)


class TestPipeline:
    def build(self, stub_vad: StubVad, asr: FakeAsr, sink: FakeSink) -> Pipeline:
        return Pipeline(
            asr,
            HallucinationGuard(stub_vad, 0.5),
            DefaultPostProcessor(PostprocessConfig()),
            [sink],
        )

    def test_flow(self, stub_vad: StubVad) -> None:
        sink = FakeSink()
        result = self.build(stub_vad, FakeAsr(), sink).process(speech_tone())
        assert result.text == "hello world." and sink.texts == ["hello world."]
        assert set(result.timings) == {"preprocess", "guard", "asr", "postprocess", "output"}

    def test_silence_never_reaches_asr(self, stub_vad: StubVad) -> None:
        asr, sink = FakeAsr(), FakeSink()
        result = self.build(stub_vad, asr, sink).process(np.zeros(16000, np.float32))
        assert result.text == "" and asr.calls == 0 and sink.texts == []

    @pytest.mark.parametrize(
        ("text", "outcome"),
        [
            ("hello", "5 characters sent"),
            ("", "recognizer returned nothing"),
            ("Thank you.", "dropped as phantom"),
        ],
    )
    def test_outcome_after_speech(self, stub_vad: StubVad, text: str, outcome: str) -> None:
        result = self.build(stub_vad, FakeAsr(text), FakeSink()).process(speech_tone())
        assert result.outcome == outcome and result.speech_ms > 0

    def test_outcome_without_speech(self, stub_vad: StubVad) -> None:
        pipeline = self.build(stub_vad, FakeAsr(), FakeSink())
        result = pipeline.process(np.zeros(16000, np.float32))
        assert result.outcome == "no speech detected" and result.speech_ms == 0

    def test_phantom_text_is_not_delivered(self, stub_vad: StubVad) -> None:
        sink = FakeSink()
        asr = FakeAsr("Thank you.")
        assert self.build(stub_vad, asr, sink).process(speech_tone()).text == ""
        assert asr.calls == 1 and sink.texts == []
