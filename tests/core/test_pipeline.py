from __future__ import annotations


import pytest

from dubbing.aligner import TimedSegment
from dubbing.backends.base import TTSBackend
from dubbing.models import (
    JobConfig,
    Segment,
    SegmentLimitExceeded,
    SynthesisTimeBudgetExceeded,
    TTSResult,
    _env_float,
    _env_int,
)
from dubbing.pipeline import DubbingPipeline


# ---------------------------------------------------------------------------
# Controllable test double — NOT the implementation's MockTTSBackend
# ---------------------------------------------------------------------------


class _FixedDurationBackend(TTSBackend):
    """Returns a fixed duration_ms for every segment; records all calls."""

    def __init__(self, duration_ms: int = 1000) -> None:
        self.duration_ms = duration_ms
        self.calls: list[list[Segment]] = []

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        self.calls.append(list(segments))
        return [
            TTSResult(segment=seg, audio_bytes=b"", duration_ms=self.duration_ms)
            for seg in segments
        ]


# ---------------------------------------------------------------------------
# Sample SRT strings
# ---------------------------------------------------------------------------

_SRT_SINGLE = """\
1
00:00:01,000 --> 00:00:03,000
Hello world

"""

_SRT_MULTI = """\
1
00:00:00,000 --> 00:00:02,000
First subtitle

2
00:00:03,000 --> 00:00:05,000
Second subtitle

3
00:00:06,000 --> 00:00:08,000
Third subtitle

"""

_SRT_PROSODY = """\
1
00:00:00,000 --> 00:00:02,000
<emotion:happy>Hello world

"""

_SRT_MULTI_TAG = """\
1
00:00:00,000 --> 00:00:03,000
<rate:slow><emotion:sad>Goodbye

"""


# ---------------------------------------------------------------------------
# Basic return type and count
# ---------------------------------------------------------------------------


class TestPipelineReturnType:
    def test_run_returns_list(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_SINGLE)
        assert isinstance(result, list)

    def test_single_entry_returns_one_item(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_SINGLE)
        assert len(result) == 1

    def test_multi_entry_returns_correct_count(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_MULTI)
        assert len(result) == 3

    def test_items_are_timed_segments(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_SINGLE)
        assert all(isinstance(ts, TimedSegment) for ts in result)


# ---------------------------------------------------------------------------
# Segment plan and timing
# ---------------------------------------------------------------------------


class TestPipelineTiming:
    def test_single_segment_start_ms(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_SINGLE)
        assert result[0].start_ms == 1000

    def test_single_segment_end_ms(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_SINGLE)
        assert result[0].end_ms == 3000

    def test_multi_segment_timings_sequential(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_MULTI)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 2000
        assert result[1].start_ms == 3000
        assert result[1].end_ms == 5000
        assert result[2].start_ms == 6000
        assert result[2].end_ms == 8000

    def test_timing_independent_of_tts_duration(self):
        # Whether TTS says 100ms or 10_000ms, aligned bounds = SRT window
        short_backend = _FixedDurationBackend(duration_ms=50)
        long_backend = _FixedDurationBackend(duration_ms=50_000)
        r_short = DubbingPipeline(short_backend).run(_SRT_MULTI)
        r_long = DubbingPipeline(long_backend).run(_SRT_MULTI)
        for ts_s, ts_l in zip(r_short, r_long):
            assert ts_s.start_ms == ts_l.start_ms
            assert ts_s.end_ms == ts_l.end_ms


# ---------------------------------------------------------------------------
# Prosody tag stripping
# ---------------------------------------------------------------------------


class TestPipelineProsody:
    def test_prosody_tag_stripped_from_segment_text(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_PROSODY)
        assert "<emotion:happy>" not in result[0].segment.entry.text

    def test_text_content_preserved_after_stripping(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_PROSODY)
        assert "Hello world" in result[0].segment.entry.text

    def test_prosody_tag_stored_on_segment(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_PROSODY)
        tags = result[0].segment.tags
        assert len(tags) == 1
        assert tags[0].name == "emotion"
        assert tags[0].value == "happy"

    def test_multiple_prosody_tags_all_stripped(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_MULTI_TAG)
        text = result[0].segment.entry.text
        assert "<rate:slow>" not in text
        assert "<emotion:sad>" not in text

    def test_multiple_prosody_tags_all_stored(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_MULTI_TAG)
        tag_names = {t.name for t in result[0].segment.tags}
        assert "rate" in tag_names
        assert "emotion" in tag_names

    def test_segment_without_tags_has_empty_tags_list(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_SINGLE)
        assert result[0].segment.tags == []


# ---------------------------------------------------------------------------
# Backend interaction — correct calls, no audio files
# ---------------------------------------------------------------------------


class TestPipelineBackendInteraction:
    def test_backend_called_once_per_segment(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_MULTI)
        assert len(backend.calls) == 3

    def test_backend_receives_all_segments(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_MULTI)
        total = sum(len(c) for c in backend.calls)
        assert total == 3

    def test_backend_receives_cleaned_text(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_PROSODY)
        all_segs = [s for c in backend.calls for s in c]
        assert all("<emotion:happy>" not in seg.entry.text for seg in all_segs)

    def test_no_audio_files_written(self, tmp_path):
        before = set(tmp_path.rglob("*"))
        DubbingPipeline(_FixedDurationBackend()).run(_SRT_MULTI)
        after = set(tmp_path.rglob("*"))
        assert before == after, "pipeline.run() must not write any files"


# ---------------------------------------------------------------------------
# Plain-text fallback (non-SRT string input)
# ---------------------------------------------------------------------------


class TestPipelinePlainTextFallback:
    def test_plain_text_produces_one_segment(self):
        result = DubbingPipeline(_FixedDurationBackend()).run("Just some words")
        assert len(result) == 1

    def test_plain_text_segment_has_input_text(self):
        result = DubbingPipeline(_FixedDurationBackend()).run("Just some words")
        assert result[0].segment.entry.text == "Just some words"

    def test_plain_text_segment_start_ms_is_zero(self):
        result = DubbingPipeline(_FixedDurationBackend()).run("Hello")
        assert result[0].start_ms == 0

    def test_plain_text_segment_end_ms_is_zero(self):
        result = DubbingPipeline(_FixedDurationBackend()).run("Hello")
        assert result[0].end_ms == 0

    def test_empty_string_input_produces_one_segment(self):
        result = DubbingPipeline(_FixedDurationBackend()).run("")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# File (Path) input
# ---------------------------------------------------------------------------


class TestPipelineFileInput:
    def test_path_input_reads_srt_file(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = DubbingPipeline(_FixedDurationBackend()).run(srt)
        assert len(result) == 1

    def test_path_input_timing_correct(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = DubbingPipeline(_FixedDurationBackend()).run(srt)
        assert result[0].start_ms == 1000
        assert result[0].end_ms == 3000

    def test_path_input_multi_segment(self, tmp_path):
        srt = tmp_path / "multi.srt"
        srt.write_text(_SRT_MULTI, encoding="utf-8")
        result = DubbingPipeline(_FixedDurationBackend()).run(srt)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Backend injected directly — acceptance criteria coverage
# ---------------------------------------------------------------------------


class TestBackendInjected:
    def test_pipeline_accepts_backend(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_SINGLE)
        assert isinstance(result, list)

    def test_single_srt_count(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_SINGLE)
        assert len(result) == 1

    def test_multi_srt_count(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_MULTI)
        assert len(result) == 3

    def test_len_results_equals_len_srt_entries(self):
        backend = _FixedDurationBackend()
        result = DubbingPipeline(backend).run(_SRT_MULTI)
        total_segs = sum(len(c) for c in backend.calls)
        assert len(result) == total_segs

    def test_len_results_equals_input_segment_count_single(self):
        backend = _FixedDurationBackend()
        result = DubbingPipeline(backend).run(_SRT_SINGLE)
        assert len(result) == len(backend.calls[0])

    def test_backend_items_are_timed_segments(self):
        result = DubbingPipeline(_FixedDurationBackend()).run(_SRT_MULTI)
        assert all(isinstance(ts, TimedSegment) for ts in result)


# ---------------------------------------------------------------------------
# Language field forwarded to backend
# ---------------------------------------------------------------------------


class TestLanguageFieldForwarded:
    def test_language_default_is_empty_string(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_SINGLE)
        seg = backend.calls[0][0]
        assert seg.language == ""

    def test_language_forwarded_to_all_segments(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_MULTI)
        all_segs = [s for c in backend.calls for s in c]
        for seg in all_segs:
            assert seg.language == ""

    def test_language_forwarded_with_prosody_tags(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_PROSODY)
        seg = backend.calls[0][0]
        assert seg.language == ""

    def test_language_field_is_str_type(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_SINGLE)
        seg = backend.calls[0][0]
        assert isinstance(seg.language, str)

    def test_explicit_language_propagated_to_segment(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_SINGLE, language="es")
        assert backend.calls[0][0].language == "es"

    def test_explicit_language_propagated_to_all_segments(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_MULTI, language="fr")
        all_segs = [s for c in backend.calls for s in c]
        assert all(seg.language == "fr" for seg in all_segs)

    def test_run_full_propagates_language(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run_full(_SRT_SINGLE, language="ja")
        assert backend.calls[0][0].language == "ja"

    def test_language_propagated_for_plain_text_fallback(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run("plain words", language="de")
        assert backend.calls[0][0].language == "de"


# ---------------------------------------------------------------------------
# ProsodyTag list forwarded to backend.synthesize
# ---------------------------------------------------------------------------


class TestProsodyTagsForwardedToBackend:
    def test_single_tag_forwarded_to_backend(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_PROSODY)
        seg = backend.calls[0][0]
        assert len(seg.tags) == 1
        assert seg.tags[0].name == "emotion"
        assert seg.tags[0].value == "happy"

    def test_multiple_tags_forwarded_to_backend(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_MULTI_TAG)
        seg = backend.calls[0][0]
        assert len(seg.tags) == 2

    def test_tag_names_forwarded_correctly(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_MULTI_TAG)
        seg = backend.calls[0][0]
        names = {t.name for t in seg.tags}
        assert "rate" in names
        assert "emotion" in names

    def test_no_tags_forwarded_when_plain_text(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_SINGLE)
        seg = backend.calls[0][0]
        assert seg.tags == []

    def test_tags_stripped_from_segment_text_before_backend(self):
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(_SRT_PROSODY)
        seg = backend.calls[0][0]
        # text passed to backend has NO raw tag markup
        assert "<emotion:happy>" not in seg.entry.text
        assert "Hello world" in seg.entry.text

    def test_tags_forwarded_independently_per_segment(self):
        srt = """\
1
00:00:00,000 --> 00:00:02,000
<emotion:happy>First

2
00:00:03,000 --> 00:00:05,000
No tags here

"""
        backend = _FixedDurationBackend()
        DubbingPipeline(backend).run(srt)
        all_segs = sorted(
            [s for c in backend.calls for s in c],
            key=lambda s: s.entry.index,
        )
        assert len(all_segs[0].tags) == 1
        assert all_segs[1].tags == []


class TestJobConfigLimits:
    def test_segment_limit_refuses_before_backend(self):
        backend = _FixedDurationBackend()
        with pytest.raises(SegmentLimitExceeded) as exc_info:
            DubbingPipeline(backend, JobConfig(max_segments=2)).run(_SRT_MULTI)
        assert backend.calls == []
        assert exc_info.value.error_code == "segment_count_exceeded"
        assert exc_info.value.limit == 2
        assert exc_info.value.requested == 3

    def test_estimated_time_limit_refuses_before_backend(self):
        backend = _FixedDurationBackend()
        with pytest.raises(SynthesisTimeBudgetExceeded) as exc_info:
            DubbingPipeline(backend, JobConfig(max_synthesis_seconds=5)).run(_SRT_MULTI)
        assert backend.calls == []
        assert exc_info.value.error_code == "time_budget_exceeded"
        assert exc_info.value.limit == 5
        assert exc_info.value.requested == 6

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_segments": -1}, "max_segments"),
            ({"max_segments": True}, "max_segments"),
            ({"max_synthesis_seconds": -1}, "max_synthesis_seconds"),
            ({"max_synthesis_seconds": float("inf")}, "max_synthesis_seconds"),
            ({"max_synthesis_seconds": float("nan")}, "max_synthesis_seconds"),
            ({"max_synthesis_seconds": True}, "max_synthesis_seconds"),
        ],
    )
    def test_job_config_rejects_invalid_values(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            JobConfig(**kwargs)

    def test_none_limits_preserve_direct_api_behavior(self):
        result = DubbingPipeline(_FixedDurationBackend(), JobConfig()).run(_SRT_MULTI)
        assert len(result) == 3


class TestEnvironmentLimitParsing:
    @pytest.mark.parametrize("value", ["", "garbage", "0", "-1"])
    def test_integer_env_drift_falls_back(self, monkeypatch, value):
        monkeypatch.setenv("DUBBING_TEST_LIMIT", value)
        assert _env_int("DUBBING_TEST_LIMIT", 17) == 17

    @pytest.mark.parametrize("value", ["", "garbage", "0", "-1", "nan", "inf", "1e999"])
    def test_float_env_drift_falls_back(self, monkeypatch, value):
        monkeypatch.setenv("DUBBING_TEST_LIMIT", value)
        assert _env_float("DUBBING_TEST_LIMIT", 2.5) == 2.5

    def test_valid_environment_values_are_retained(self, monkeypatch):
        monkeypatch.setenv("DUBBING_TEST_INT", "9")
        monkeypatch.setenv("DUBBING_TEST_FLOAT", "1.25")
        assert _env_int("DUBBING_TEST_INT", 17) == 9
        assert _env_float("DUBBING_TEST_FLOAT", 2.5) == 1.25


class TestStrictFileInputs:
    @pytest.mark.parametrize("payload", ["", "  \n", "1\n00:00:01,000 -->"])
    def test_no_valid_srt_entries_refuse_before_backend(self, tmp_path, payload):
        source = tmp_path / "invalid.srt"
        source.write_text(payload, encoding="utf-8")
        backend = _FixedDurationBackend()
        with pytest.raises(ValueError, match="no valid subtitle entries"):
            DubbingPipeline(backend).run(source)
        assert backend.calls == []

    def test_nul_srt_refuses_before_backend(self, tmp_path):
        source = tmp_path / "invalid.srt"
        source.write_text(_SRT_SINGLE.replace("Hello", "Hello\x00"), encoding="utf-8")
        backend = _FixedDurationBackend()
        with pytest.raises(ValueError, match="NUL"):
            DubbingPipeline(backend).run(source)
        assert backend.calls == []

    @pytest.mark.parametrize("language", ["x" * 21, "한국어", "en\n"])
    def test_invalid_language_refuses_before_backend(self, language):
        backend = _FixedDurationBackend()
        with pytest.raises(ValueError, match="language"):
            DubbingPipeline(backend).run(_SRT_SINGLE, language=language)
        assert backend.calls == []
