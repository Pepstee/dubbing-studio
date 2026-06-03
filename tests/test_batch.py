from __future__ import annotations

from pathlib import Path

import pytest

from dubbing.aligner import TimedSegment
from dubbing.backends.base import TTSBackend
from dubbing.batch import batch_dub
from dubbing.models import Segment, TTSResult


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------

class _MockBackend(TTSBackend):
    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        return [
            TTSResult(segment=seg, audio_bytes=b"", duration_ms=1000)
            for seg in segments
        ]


# ---------------------------------------------------------------------------
# SRT fixtures
# ---------------------------------------------------------------------------

_SRT_ONE_SEGMENT = """\
1
00:00:00,000 --> 00:00:02,000
Alpha segment

"""

_SRT_TWO_SEGMENTS = """\
1
00:00:00,000 --> 00:00:01,000
Beta one

2
00:00:02,000 --> 00:00:04,000
Beta two

"""

_SRT_THREE_SEGMENTS = """\
1
00:00:00,000 --> 00:00:01,000
Line one

2
00:00:02,000 --> 00:00:03,000
Line two

3
00:00:04,000 --> 00:00:05,000
Line three

"""


@pytest.fixture()
def two_srt_files(tmp_path: Path):
    a = tmp_path / "alpha.srt"
    b = tmp_path / "beta.srt"
    a.write_text(_SRT_ONE_SEGMENT, encoding="utf-8")
    b.write_text(_SRT_TWO_SEGMENTS, encoding="utf-8")
    return a, b


# ---------------------------------------------------------------------------
# Return value shape — two mock SRT inputs
# ---------------------------------------------------------------------------

class TestBatchDubOutputShape:
    def test_returns_dict(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert isinstance(results, dict)

    def test_dict_has_two_entries(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert len(results) == 2

    def test_keys_are_path_objects(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert all(isinstance(k, Path) for k in results)

    def test_input_paths_are_in_keys(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert a in results
        assert b in results


# ---------------------------------------------------------------------------
# Segment integrity — counts
# ---------------------------------------------------------------------------

class TestBatchDubSegmentCounts:
    def test_first_file_has_one_segment(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert len(results[a]) == 1

    def test_second_file_has_two_segments(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert len(results[b]) == 2

    def test_three_segment_file_count(self, tmp_path):
        c = tmp_path / "gamma.srt"
        c.write_text(_SRT_THREE_SEGMENTS, encoding="utf-8")
        results = batch_dub([c], _MockBackend(), tmp_path / "out")
        assert len(results[c]) == 3


# ---------------------------------------------------------------------------
# Segment integrity — timing and text
# ---------------------------------------------------------------------------

class TestBatchDubSegmentContent:
    def test_segments_are_timed_segment_instances(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        for segs in results.values():
            for ts in segs:
                assert isinstance(ts, TimedSegment)

    def test_first_file_segment_start_ms(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert results[a][0].start_ms == 0

    def test_first_file_segment_end_ms(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert results[a][0].end_ms == 2000

    def test_first_file_segment_text(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert "Alpha segment" in results[a][0].segment.entry.text

    def test_second_file_segments_order(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        segs = results[b]
        assert segs[0].start_ms == 0
        assert segs[0].end_ms == 1000
        assert segs[1].start_ms == 2000
        assert segs[1].end_ms == 4000

    def test_second_file_segment_texts(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        texts = [ts.segment.entry.text for ts in results[b]]
        assert any("Beta one" in t for t in texts)
        assert any("Beta two" in t for t in texts)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

class TestBatchDubOutputDir:
    def test_creates_output_directory(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        out_dir = tmp_path / "created_by_batch"
        assert not out_dir.exists()
        batch_dub([a, b], _MockBackend(), out_dir)
        assert out_dir.exists()

    def test_creates_nested_output_directory(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        out_dir = tmp_path / "nested" / "deep" / "output"
        batch_dub([a, b], _MockBackend(), out_dir)
        assert out_dir.exists()

    def test_accepts_string_output_dir(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        out_dir = str(tmp_path / "str_dir")
        results = batch_dub([a, b], _MockBackend(), out_dir)
        assert len(results) == 2

    def test_no_audio_files_written_to_output_dir(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        out_dir = tmp_path / "out"
        batch_dub([a, b], _MockBackend(), out_dir)
        audio = list(out_dir.glob("*.wav")) + list(out_dir.glob("*.mp3"))
        assert audio == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBatchDubEdgeCases:
    def test_empty_input_list_returns_empty_dict(self, tmp_path):
        results = batch_dub([], _MockBackend(), tmp_path / "out")
        assert results == {}

    def test_single_input(self, tmp_path):
        srt = tmp_path / "only.srt"
        srt.write_text(_SRT_ONE_SEGMENT, encoding="utf-8")
        results = batch_dub([srt], _MockBackend(), tmp_path / "out")
        assert len(results) == 1

    def test_accepts_string_paths_as_inputs(self, two_srt_files, tmp_path):
        a, b = two_srt_files
        results = batch_dub([str(a), str(b)], _MockBackend(), tmp_path / "out")
        assert len(results) == 2

    def test_duplicate_input_path_processed_once_per_occurrence(self, tmp_path):
        srt = tmp_path / "dup.srt"
        srt.write_text(_SRT_ONE_SEGMENT, encoding="utf-8")
        # passing same file twice: dict will have one key (Path deduped), list has 2 run results
        results = batch_dub([srt, srt], _MockBackend(), tmp_path / "out")
        # both entries map to same Path key; the second run overwrites the first in the dict
        assert srt in results


# ---------------------------------------------------------------------------
# Two-SRT batch acceptance: explicit coverage of acceptance criteria
# ---------------------------------------------------------------------------

class TestBatchDubTwoSRTs:
    def test_batch_two_srt_files_returns_two_results(self, tmp_path):
        a = tmp_path / "first.srt"
        b = tmp_path / "second.srt"
        a.write_text(_SRT_ONE_SEGMENT, encoding="utf-8")
        b.write_text(_SRT_TWO_SEGMENTS, encoding="utf-8")
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert len(results) == 2

    def test_batch_two_srt_each_key_is_its_input_path(self, tmp_path):
        a = tmp_path / "first.srt"
        b = tmp_path / "second.srt"
        a.write_text(_SRT_ONE_SEGMENT, encoding="utf-8")
        b.write_text(_SRT_TWO_SEGMENTS, encoding="utf-8")
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        assert a in results
        assert b in results

    def test_batch_two_srt_values_are_lists_of_timed_segments(self, tmp_path):
        a = tmp_path / "first.srt"
        b = tmp_path / "second.srt"
        a.write_text(_SRT_ONE_SEGMENT, encoding="utf-8")
        b.write_text(_SRT_TWO_SEGMENTS, encoding="utf-8")
        results = batch_dub([a, b], _MockBackend(), tmp_path / "out")
        for segs in results.values():
            assert isinstance(segs, list)
            assert all(isinstance(ts, TimedSegment) for ts in segs)

    def test_batch_two_srt_no_audio_files_written(self, tmp_path):
        a = tmp_path / "first.srt"
        b = tmp_path / "second.srt"
        a.write_text(_SRT_ONE_SEGMENT, encoding="utf-8")
        b.write_text(_SRT_TWO_SEGMENTS, encoding="utf-8")
        out_dir = tmp_path / "out"
        batch_dub([a, b], _MockBackend(), out_dir)
        written_files = [f for f in out_dir.rglob("*") if f.is_file()]
        assert written_files == [], f"Unexpected files written: {written_files}"

    def test_batch_no_srt_files_written_to_output(self, tmp_path):
        a = tmp_path / "a.srt"
        b = tmp_path / "b.srt"
        a.write_text(_SRT_TWO_SEGMENTS, encoding="utf-8")
        b.write_text(_SRT_THREE_SEGMENTS, encoding="utf-8")
        out_dir = tmp_path / "out"
        batch_dub([a, b], _MockBackend(), out_dir)
        srt_files = list(out_dir.rglob("*.srt"))
        assert srt_files == []

    def test_batch_only_creates_output_directory_nothing_else(self, tmp_path):
        a = tmp_path / "a.srt"
        b = tmp_path / "b.srt"
        a.write_text(_SRT_ONE_SEGMENT, encoding="utf-8")
        b.write_text(_SRT_TWO_SEGMENTS, encoding="utf-8")
        out_dir = tmp_path / "only_dir"
        batch_dub([a, b], _MockBackend(), out_dir)
        all_items = list(out_dir.rglob("*"))
        # directory created; no files inside it
        assert out_dir.is_dir()
        assert all(not p.is_file() for p in all_items)
