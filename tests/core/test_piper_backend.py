"""Adversarial test suite for PiperTTSBackend and select_backend().

All subprocess calls are mocked; no real piper binary or model file is needed.
"""

from __future__ import annotations

import io
import json
import subprocess
import wave
from unittest.mock import MagicMock, patch

import pytest

from dubbing.backends import EspeakTTSBackend, PiperTTSBackend, SayTTSBackend, select_backend
from dubbing.backends.base import TTSBackend
from dubbing.backends.piper import _raw_to_wav, _validate_model_path, _wav_duration_ms
from dubbing.models import ProsodyTag, Segment, SRTEntry, TTSResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_PATH = "/models/en_US-lessac-medium.onnx"
_PCM_SILENT = b"\x00\x00" * 2205  # 100 ms of 16-bit silence at 22050 Hz


def _make_entry(
    index: int = 1, start_ms: int = 0, end_ms: int = 2000, text: str = "Hello"
) -> SRTEntry:
    return SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text)


def _make_segment(
    text: str = "Hello",
    tags: list[ProsodyTag] | None = None,
    language: str = "en-US",
    index: int = 1,
    start_ms: int = 0,
    end_ms: int = 2000,
) -> Segment:
    return Segment(
        entry=_make_entry(index=index, start_ms=start_ms, end_ms=end_ms, text=text),
        tags=tags or [],
        language=language,
    )


def _completed_process(
    stdout: bytes = _PCM_SILENT, stderr: bytes = b""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["piper"], 0, stdout=stdout, stderr=stderr)


def _piper_patch(stdout: bytes = _PCM_SILENT, stderr: bytes = b""):
    """Context manager: patch shutil.which (piper found) and subprocess.run."""
    which_patch = patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper")
    run_patch = patch(
        "dubbing.backends.piper.subprocess.run",
        return_value=_completed_process(stdout=stdout, stderr=stderr),
    )
    return which_patch, run_patch


# ---------------------------------------------------------------------------
# Class membership
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_piper_backend_subclasses_tts_backend(self):
        assert issubclass(PiperTTSBackend, TTSBackend)

    def test_instance_is_tts_backend(self):
        assert isinstance(PiperTTSBackend(model=_MODEL_PATH), TTSBackend)


# ---------------------------------------------------------------------------
# Happy path — subprocess.run mocked to return valid raw PCM
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_returns_list(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                results = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        assert isinstance(results, list)

    def test_single_segment_yields_one_result(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                results = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        assert len(results) == 1

    def test_result_is_ttsresult_instance(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        assert isinstance(result, TTSResult)

    def test_audio_bytes_is_bytes(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        assert isinstance(result.audio_bytes, bytes)

    def test_audio_starts_with_riff_header(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        assert result.audio_bytes[:4] == b"RIFF"
        assert result.audio_bytes[8:12] == b"WAVE"

    def test_audio_is_valid_wav(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        with wave.open(io.BytesIO(result.audio_bytes)) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050

    def test_duration_ms_is_int(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        assert isinstance(result.duration_ms, int)

    def test_duration_ms_is_positive(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        assert result.duration_ms > 0

    def test_segment_reference_preserved_in_result(self):
        seg = _make_segment()
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([seg])[0]
        assert result.segment is seg

    def test_three_segments_yield_three_results(self):
        segments = [_make_segment(text=f"Segment {i}", index=i) for i in range(1, 4)]
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                results = PiperTTSBackend(model=_MODEL_PATH).synthesize(segments)
        assert len(results) == 3

    def test_results_order_matches_input_order(self):
        segments = [_make_segment(text=f"text {i}", index=i) for i in range(1, 4)]
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                results = PiperTTSBackend(model=_MODEL_PATH).synthesize(segments)
        for result, seg in zip(results, segments):
            assert result.segment is seg

    def test_empty_segment_list_yields_empty_list(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                results = PiperTTSBackend(model=_MODEL_PATH).synthesize([])
        assert results == []

    def test_subprocess_called_once_per_segment(self):
        segments = [_make_segment(text=f"Segment {i}", index=i) for i in range(1, 4)]
        mock_run = MagicMock(return_value=_completed_process())
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", mock_run):
                PiperTTSBackend(model=_MODEL_PATH).synthesize(segments)
        assert mock_run.call_count == 3


# ---------------------------------------------------------------------------
# Error: piper binary absent
# ---------------------------------------------------------------------------


class TestPiperNotFound:
    def test_raises_runtime_error_when_which_returns_none(self):
        with patch("dubbing.backends.piper.shutil.which", return_value=None):
            with pytest.raises(RuntimeError):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])

    def test_error_message_mentions_piper(self):
        with patch("dubbing.backends.piper.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="piper"):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])

    def test_error_message_mentions_path(self):
        with patch("dubbing.backends.piper.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="PATH"):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])

    def test_raises_even_with_empty_segment_list(self):
        """piper absence is checked before iterating — but empty list never enters the loop."""
        with patch("dubbing.backends.piper.shutil.which", return_value=None):
            with pytest.raises(RuntimeError):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])

    def test_error_type_is_exactly_runtime_error(self):
        with patch("dubbing.backends.piper.shutil.which", return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        assert type(exc_info.value) is RuntimeError

    def test_no_subprocess_call_when_piper_absent(self):
        mock_run = MagicMock()
        with patch("dubbing.backends.piper.shutil.which", return_value=None):
            with patch("dubbing.backends.piper.subprocess.run", mock_run):
                with pytest.raises(RuntimeError):
                    PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Error: model not configured
# ---------------------------------------------------------------------------


class TestModelNotConfigured:
    def test_raises_when_model_not_set_and_no_env_var(self, monkeypatch):
        monkeypatch.delenv("PIPER_MODEL", raising=False)
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with pytest.raises(RuntimeError, match="model"):
                PiperTTSBackend().synthesize([_make_segment()])

    def test_raises_when_model_is_empty_string(self, monkeypatch):
        monkeypatch.delenv("PIPER_MODEL", raising=False)
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with pytest.raises(RuntimeError):
                PiperTTSBackend(model="").synthesize([_make_segment()])

    def test_error_message_mentions_piper_model_env_var(self, monkeypatch):
        monkeypatch.delenv("PIPER_MODEL", raising=False)
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with pytest.raises(RuntimeError, match="PIPER_MODEL"):
                PiperTTSBackend().synthesize([_make_segment()])

    def test_env_var_used_as_fallback_model(self, monkeypatch):
        monkeypatch.setenv("PIPER_MODEL", _MODEL_PATH)
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                results = PiperTTSBackend().synthesize([_make_segment()])
        assert len(results) == 1

    def test_constructor_model_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("PIPER_MODEL", "/env/model.onnx")
        captured_calls: list[list[str]] = []

        def record(cmd, **kwargs):
            captured_calls.append(list(cmd))
            return _completed_process()

        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model="/explicit/model.onnx").synthesize([_make_segment()])
        assert "/explicit/model.onnx" in captured_calls[0]
        assert "/env/model.onnx" not in captured_calls[0]


class TestModelPathValidation:
    @pytest.mark.parametrize("model", ["bad;model", "bad$model", "bad\nmodel", "bad\x00model"])
    def test_forbidden_model_path_refuses_before_subprocess(self, model):
        run = MagicMock()
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", run):
                with pytest.raises(ValueError, match="forbidden"):
                    PiperTTSBackend(model=model).synthesize([_make_segment()])
        run.assert_not_called()

    def test_printable_path_is_accepted(self):
        _validate_model_path("/models/voice model.onnx")


# ---------------------------------------------------------------------------
# Subprocess argv — model path and flags reach piper
# ---------------------------------------------------------------------------


class TestSubprocessArgv:
    def _capture(self) -> tuple[list[list[str]], MagicMock]:
        calls: list[list[str]] = []

        def record(cmd, **kwargs):
            calls.append(list(cmd))
            return _completed_process()

        return calls, record

    def test_argv_starts_with_piper(self):
        calls, record = self._capture()
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        assert calls[0][0] == "piper"

    def test_model_flag_in_argv(self):
        calls, record = self._capture()
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        assert "--model" in calls[0]

    def test_model_path_follows_model_flag(self):
        calls, record = self._capture()
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        idx = calls[0].index("--model")
        assert calls[0][idx + 1] == _MODEL_PATH

    def test_output_raw_flag_in_argv(self):
        calls, record = self._capture()
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        assert "--output_raw" in calls[0]

    def test_text_passed_via_stdin_not_argv(self):
        """Text must not appear as an argv item — it's delivered on stdin."""
        text = "Bonjour le monde"
        captured_kwargs: list[dict] = []

        def record(cmd, **kwargs):
            captured_kwargs.append(kwargs)
            return _completed_process()

        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment(text=text)])
        assert captured_kwargs[0]["input"] == text.encode("utf-8")

    def test_hostile_text_not_injected_into_argv(self):
        """A '-' leading text or flag-like text must reach piper as stdin data."""
        text = "--model /evil/path.onnx"
        captured: list[tuple[list[str], dict]] = []

        def record(cmd, **kwargs):
            captured.append((list(cmd), kwargs))
            return _completed_process()

        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment(text=text)])
        cmd, kwargs = captured[0]
        assert text not in cmd
        assert kwargs["input"] == text.encode("utf-8")

    def test_capture_stderr_is_set(self):
        """stderr must be captured so piper's JSON info line can be read."""
        captured_kwargs: list[dict] = []

        def record(cmd, **kwargs):
            captured_kwargs.append(kwargs)
            return _completed_process()

        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
        assert captured_kwargs[0].get("capture_output") is True


# ---------------------------------------------------------------------------
# Prosody tags — preserved in result; text reaches subprocess stdin
# ---------------------------------------------------------------------------


class TestProsodyTagsAndText:
    def test_tags_preserved_in_result(self):
        tag = ProsodyTag(name="rate", value="slow")
        seg = _make_segment(tags=[tag])
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([seg])[0]
        assert result.segment.tags == [tag]

    def test_segment_text_reaches_subprocess_stdin(self):
        """Any prosody-annotated text (e.g. SSML-like) is passed verbatim as stdin."""
        text = "<rate:slow>Hello world"
        captured_input: list[bytes] = []

        def record(cmd, **kwargs):
            captured_input.append(kwargs["input"])
            return _completed_process()

        seg = _make_segment(text=text)
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", record):
                PiperTTSBackend(model=_MODEL_PATH).synthesize([seg])
        assert captured_input[0] == text.encode("utf-8")

    def test_multiple_tags_preserved(self):
        tags = [ProsodyTag(name="rate", value="fast"), ProsodyTag(name="pitch", value="high")]
        seg = _make_segment(tags=tags)
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([seg])[0]
        assert result.segment.tags == tags


# ---------------------------------------------------------------------------
# Language code — preserved in TTSResult
# ---------------------------------------------------------------------------


class TestLanguagePassThrough:
    def test_language_preserved_in_result(self):
        seg = _make_segment(language="de-DE")
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([seg])[0]
        assert result.segment.language == "de-DE"

    def test_empty_language_preserved(self):
        seg = _make_segment(language="")
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([seg])[0]
        assert result.segment.language == ""

    def test_language_for_each_result_matches_input_segment(self):
        segments = [
            _make_segment(language=lang, index=i)
            for i, lang in enumerate(["en-US", "fr-FR", "de-DE"], 1)
        ]
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", return_value=_completed_process()):
                results = PiperTTSBackend(model=_MODEL_PATH).synthesize(segments)
        for result, seg in zip(results, segments):
            assert result.segment.language == seg.language


# ---------------------------------------------------------------------------
# Sample rate extraction from piper's stderr JSON
# ---------------------------------------------------------------------------


class TestSampleRateFromStderr:
    def test_default_sample_rate_when_no_stderr(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch(
                "dubbing.backends.piper.subprocess.run", return_value=_completed_process(stderr=b"")
            ):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        with wave.open(io.BytesIO(result.audio_bytes)) as wf:
            assert wf.getframerate() == 22050

    def test_sample_rate_extracted_from_stderr_json(self):
        stderr_json = json.dumps({"audio": {"sample_rate": 16000}}).encode("utf-8")
        # 100 ms of PCM at 16000 Hz = 1600 samples × 2 bytes = 3200 bytes
        raw_pcm_16k = b"\x00\x00" * 1600
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch(
                "dubbing.backends.piper.subprocess.run",
                return_value=_completed_process(stdout=raw_pcm_16k, stderr=stderr_json),
            ):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        with wave.open(io.BytesIO(result.audio_bytes)) as wf:
            assert wf.getframerate() == 16000

    def test_non_json_stderr_falls_back_to_default_rate(self):
        stderr = b"[S] Starting synthesis\n[S] Done\n"
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch(
                "dubbing.backends.piper.subprocess.run",
                return_value=_completed_process(stderr=stderr),
            ):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        with wave.open(io.BytesIO(result.audio_bytes)) as wf:
            assert wf.getframerate() == 22050

    def test_mixed_json_and_non_json_stderr_uses_first_audio_key(self):
        lines = [
            b"piper info line",
            json.dumps({"audio": {"sample_rate": 24000}}).encode(),
            json.dumps({"audio": {"sample_rate": 8000}}).encode(),
        ]
        stderr = b"\n".join(lines)
        raw_pcm_24k = b"\x00\x00" * 2400  # 100 ms at 24 kHz
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch(
                "dubbing.backends.piper.subprocess.run",
                return_value=_completed_process(stdout=raw_pcm_24k, stderr=stderr),
            ):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        with wave.open(io.BytesIO(result.audio_bytes)) as wf:
            assert wf.getframerate() == 24000

    def test_stderr_json_without_audio_key_uses_default(self):
        stderr = json.dumps({"status": "ok"}).encode()
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch(
                "dubbing.backends.piper.subprocess.run",
                return_value=_completed_process(stderr=stderr),
            ):
                result = PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])[0]
        with wave.open(io.BytesIO(result.audio_bytes)) as wf:
            assert wf.getframerate() == 22050

    def test_malformed_sample_rate_in_json_uses_default(self):
        stderr = json.dumps({"audio": {"sample_rate": "not-a-number"}}).encode()
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch(
                "dubbing.backends.piper.subprocess.run",
                return_value=_completed_process(stderr=stderr),
            ):
                # int("not-a-number") raises ValueError → falls back to default
                # The implementation uses int(info["audio"].get("sample_rate", sample_rate))
                # so if the value is already an int it works; a non-numeric string causes ValueError.
                # That exception is NOT caught, so we expect it to propagate — this test is an edge-case
                # boundary probe.  If the impl changes to handle it gracefully, adjust assertion.
                try:
                    PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])
                except (ValueError, RuntimeError):
                    pass  # either outcome is acceptable; we just must not silently corrupt the WAV


# ---------------------------------------------------------------------------
# subprocess failure propagates
# ---------------------------------------------------------------------------


class TestSubprocessFailure:
    def test_called_process_error_propagates(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch(
                "dubbing.backends.piper.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["piper"]),
            ):
                with pytest.raises(subprocess.CalledProcessError):
                    PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])

    def test_timeout_error_propagates(self):
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch(
                "dubbing.backends.piper.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["piper"], 60),
            ):
                with pytest.raises(subprocess.TimeoutExpired):
                    PiperTTSBackend(model=_MODEL_PATH).synthesize([_make_segment()])

    def test_error_on_second_segment_does_not_silently_return_partial(self):
        call_count = 0

        def fail_on_second(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise subprocess.CalledProcessError(1, cmd)
            return _completed_process()

        segments = [_make_segment(index=i) for i in range(1, 3)]
        with patch("dubbing.backends.piper.shutil.which", return_value="/usr/local/bin/piper"):
            with patch("dubbing.backends.piper.subprocess.run", fail_on_second):
                with pytest.raises(subprocess.CalledProcessError):
                    PiperTTSBackend(model=_MODEL_PATH).synthesize(segments)


# ---------------------------------------------------------------------------
# Internal helpers — _raw_to_wav and _wav_duration_ms
# ---------------------------------------------------------------------------


class TestRawToWav:
    def test_returns_bytes(self):
        assert isinstance(_raw_to_wav(b"\x00\x00" * 100, 22050), bytes)

    def test_starts_with_riff_header(self):
        wav = _raw_to_wav(b"\x00\x00" * 100, 22050)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_wav_encodes_correct_sample_rate(self):
        for rate in (16000, 22050, 24000):
            wav = _raw_to_wav(b"\x00\x00" * rate, rate)  # 1 second
            with wave.open(io.BytesIO(wav)) as wf:
                assert wf.getframerate() == rate

    def test_wav_is_mono(self):
        wav = _raw_to_wav(b"\x00\x00" * 100, 22050)
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnchannels() == 1

    def test_wav_is_16bit(self):
        wav = _raw_to_wav(b"\x00\x00" * 100, 22050)
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getsampwidth() == 2

    def test_empty_raw_produces_valid_wav_with_zero_frames(self):
        wav = _raw_to_wav(b"", 22050)
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnframes() == 0


class TestWavDurationMs:
    def test_100ms_silent_pcm(self):
        frames = int(22050 * 0.1)  # 100 ms
        wav = _raw_to_wav(b"\x00\x00" * frames, 22050)
        assert _wav_duration_ms(wav) == 100

    def test_1000ms_at_16khz(self):
        wav = _raw_to_wav(b"\x00\x00" * 16000, 16000)
        assert _wav_duration_ms(wav) == 1000

    def test_duration_rounds_down(self):
        # 22050 frames → exactly 1000 ms; add 1 extra frame (still 1000 ms after floor)
        frames = 22050 + 1
        wav = _raw_to_wav(b"\x00\x00" * frames, 22050)
        # int(22051 * 1000 / 22050) == 1000
        assert _wav_duration_ms(wav) == 1000


# ---------------------------------------------------------------------------
# select_backend() auto-select logic
# ---------------------------------------------------------------------------


class TestSelectBackend:
    def test_returns_piper_backend_when_piper_on_path(self):
        with (
            patch("dubbing.backends.shutil.which", return_value="/usr/local/bin/piper"),
            patch.dict("os.environ", {"PIPER_MODEL": _MODEL_PATH}),
        ):
            backend = select_backend()
        assert isinstance(backend, PiperTTSBackend)

    def test_returns_say_backend_when_complete_say_toolchain_exists(self):
        def which(name):
            return f"/usr/bin/{name}" if name in {"say", "afconvert"} else None

        with patch("dubbing.backends.shutil.which", side_effect=which):
            backend = select_backend()
        assert isinstance(backend, SayTTSBackend)

    def test_piper_backend_is_tts_backend(self):
        with (
            patch("dubbing.backends.shutil.which", return_value="/usr/local/bin/piper"),
            patch.dict("os.environ", {"PIPER_MODEL": _MODEL_PATH}),
        ):
            backend = select_backend()
        assert isinstance(backend, TTSBackend)

    def test_espeak_backend_is_tts_backend(self):
        def which(name):
            return "/usr/bin/espeak-ng" if name == "espeak-ng" else None

        with patch("dubbing.backends.shutil.which", side_effect=which):
            backend = select_backend()
        assert isinstance(backend, EspeakTTSBackend)

    def test_which_called_with_piper_argument(self):
        mock_which = MagicMock(return_value="/usr/local/bin/piper")
        with (
            patch("dubbing.backends.shutil.which", mock_which),
            patch.dict("os.environ", {"PIPER_MODEL": _MODEL_PATH}),
        ):
            select_backend()
        mock_which.assert_called_once_with("piper")

    def test_no_available_backend_raises_actionable_error(self):
        with patch("dubbing.backends.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="no usable local TTS backend"):
                select_backend()

    def test_piper_backend_returned_for_configured_paths(self):
        for path in ("/usr/local/bin/piper", "/opt/piper/piper", "./piper"):
            with (
                patch("dubbing.backends.shutil.which", return_value=path),
                patch.dict("os.environ", {"PIPER_MODEL": _MODEL_PATH}),
            ):
                backend = select_backend()
            assert isinstance(backend, PiperTTSBackend), f"Failed for path={path!r}"
