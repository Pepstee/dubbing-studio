from __future__ import annotations

import pytest

from dubbing.aligner import TimedSegment, TimelineAligner
from dubbing.models import Segment, SRTEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    index: int = 1,
    start_ms: int = 0,
    end_ms: int = 1000,
    text: str = "hello",
) -> Segment:
    entry = SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text)
    return Segment(entry=entry, tags=[], language="")


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestAlignEmpty:
    def test_empty_returns_empty_list(self):
        result = TimelineAligner().align([], [])
        assert result == []

    def test_empty_return_type_is_list(self):
        result = TimelineAligner().align([], [])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Single segment
# ---------------------------------------------------------------------------

class TestAlignSingleSegment:
    def test_start_ms_matches_srt_entry(self):
        seg = _make_segment(start_ms=1000, end_ms=3000)
        result = TimelineAligner().align([seg], [500])
        assert result[0].start_ms == 1000

    def test_end_ms_matches_srt_window(self):
        seg = _make_segment(start_ms=1000, end_ms=3000)
        result = TimelineAligner().align([seg], [500])
        assert result[0].end_ms == 3000

    def test_segment_reference_preserved(self):
        seg = _make_segment(start_ms=0, end_ms=2000)
        result = TimelineAligner().align([seg], [900])
        assert result[0].segment is seg

    def test_returns_timed_segment_instance(self):
        seg = _make_segment()
        result = TimelineAligner().align([seg], [200])
        assert isinstance(result[0], TimedSegment)

    def test_tts_shorter_than_window_still_fills_it(self):
        seg = _make_segment(start_ms=0, end_ms=5000)
        result = TimelineAligner().align([seg], [100])  # 100ms << 5000ms window
        assert result[0].start_ms == 0
        assert result[0].end_ms == 5000

    def test_tts_longer_than_window_is_compressed(self):
        seg = _make_segment(start_ms=0, end_ms=1000)
        result = TimelineAligner().align([seg], [9999])  # 9999ms >> 1000ms window
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1000

    def test_tts_exactly_matching_window(self):
        seg = _make_segment(start_ms=500, end_ms=2500)
        result = TimelineAligner().align([seg], [2000])
        assert result[0].start_ms == 500
        assert result[0].end_ms == 2500


# ---------------------------------------------------------------------------
# Multi-segment with duration mismatch
# ---------------------------------------------------------------------------

class TestAlignMultiSegment:
    def test_returns_correct_count(self):
        segs = [_make_segment(index=i, start_ms=i * 2000, end_ms=(i + 1) * 2000) for i in range(4)]
        result = TimelineAligner().align(segs, [100, 5000, 2000, 1])
        assert len(result) == 4

    def test_each_segment_maps_to_own_srt_window(self):
        segs = [
            _make_segment(index=1, start_ms=0, end_ms=2000, text="first"),
            _make_segment(index=2, start_ms=3000, end_ms=5000, text="second"),
            _make_segment(index=3, start_ms=6000, end_ms=8000, text="third"),
        ]
        durations = [500, 9000, 2000]  # short, long, exact — all mismatched
        result = TimelineAligner().align(segs, durations)

        assert result[0].start_ms == 0
        assert result[0].end_ms == 2000
        assert result[1].start_ms == 3000
        assert result[1].end_ms == 5000
        assert result[2].start_ms == 6000
        assert result[2].end_ms == 8000

    def test_segment_references_preserved_in_order(self):
        segs = [_make_segment(index=i, start_ms=i * 1000, end_ms=(i + 1) * 1000) for i in range(3)]
        result = TimelineAligner().align(segs, [100, 200, 300])
        for i, ts in enumerate(result):
            assert ts.segment is segs[i]

    def test_gaps_between_srt_windows_dont_interfere(self):
        segs = [
            _make_segment(index=1, start_ms=0, end_ms=100),
            _make_segment(index=2, start_ms=50_000, end_ms=51_000),  # huge gap
        ]
        result = TimelineAligner().align(segs, [500, 500])
        assert result[0].end_ms == 100
        assert result[1].start_ms == 50_000

    def test_non_zero_srt_start_offset(self):
        seg = _make_segment(start_ms=3_600_000, end_ms=3_601_000)  # starts at 1h
        result = TimelineAligner().align([seg], [500])
        assert result[0].start_ms == 3_600_000
        assert result[0].end_ms == 3_601_000


# ---------------------------------------------------------------------------
# Length mismatch raises ValueError
# ---------------------------------------------------------------------------

class TestAlignLengthMismatch:
    def test_more_segments_than_durations_raises(self):
        segs = [_make_segment(), _make_segment()]
        with pytest.raises(ValueError):
            TimelineAligner().align(segs, [1000])

    def test_more_durations_than_segments_raises(self):
        segs = [_make_segment()]
        with pytest.raises(ValueError):
            TimelineAligner().align(segs, [500, 500])

    def test_error_message_mentions_lengths(self):
        segs = [_make_segment(), _make_segment(), _make_segment()]
        with pytest.raises(ValueError, match="same length"):
            TimelineAligner().align(segs, [100])


# ---------------------------------------------------------------------------
# Degenerate / edge inputs
# ---------------------------------------------------------------------------

class TestAlignDegenerateCases:
    def test_zero_duration_window_passes_through(self):
        seg = _make_segment(start_ms=1000, end_ms=1000)  # zero-width window
        result = TimelineAligner().align([seg], [500])
        assert result[0].start_ms == 1000
        assert result[0].end_ms == 1000

    def test_negative_window_passes_through(self):
        seg = _make_segment(start_ms=2000, end_ms=1000)  # end < start
        result = TimelineAligner().align([seg], [500])
        assert result[0].start_ms == 2000
        assert result[0].end_ms == 1000

    def test_zero_tts_duration_passes_through(self):
        seg = _make_segment(start_ms=0, end_ms=3000)
        result = TimelineAligner().align([seg], [0])
        assert result[0].start_ms == 0
        assert result[0].end_ms == 3000

    def test_negative_tts_duration_passes_through(self):
        seg = _make_segment(start_ms=0, end_ms=3000)
        result = TimelineAligner().align([seg], [-1])
        assert result[0].start_ms == 0
        assert result[0].end_ms == 3000

    def test_degenerate_window_segment_reference_still_preserved(self):
        seg = _make_segment(start_ms=500, end_ms=500)
        result = TimelineAligner().align([seg], [1000])
        assert result[0].segment is seg

    def test_zero_tts_duration_segment_reference_still_preserved(self):
        seg = _make_segment(start_ms=0, end_ms=1000)
        result = TimelineAligner().align([seg], [0])
        assert result[0].segment is seg


# ---------------------------------------------------------------------------
# Stretch ratio: effective ratio == 1.0 when tts_ms matches the SRT window
# ---------------------------------------------------------------------------

class TestAlignStretchRatio:
    def test_exact_fit_effective_stretch_ratio_is_one(self):
        window_ms = 2000
        seg = _make_segment(start_ms=500, end_ms=2500)  # window = 2000ms
        result = TimelineAligner().align([seg], [window_ms])
        actual_window = result[0].end_ms - result[0].start_ms
        assert actual_window / window_ms == 1.0

    def test_exact_fit_start_unchanged(self):
        seg = _make_segment(start_ms=1000, end_ms=3000)
        result = TimelineAligner().align([seg], [2000])
        assert result[0].start_ms == 1000

    def test_exact_fit_end_unchanged(self):
        seg = _make_segment(start_ms=1000, end_ms=3000)
        result = TimelineAligner().align([seg], [2000])
        assert result[0].end_ms == 3000

    def test_exact_fit_window_preserved(self):
        seg = _make_segment(start_ms=200, end_ms=1200)  # 1000ms window
        result = TimelineAligner().align([seg], [1000])
        assert (result[0].end_ms - result[0].start_ms) == 1000

    def test_short_tts_fills_full_srt_window(self):
        # effective ratio > 1.0 — audio stretched to fill the window
        seg = _make_segment(start_ms=0, end_ms=3000)
        result = TimelineAligner().align([seg], [500])
        assert result[0].end_ms - result[0].start_ms == 3000


# ---------------------------------------------------------------------------
# Over-budget: tts_ms > srt_window — output stays within SRT bounds
# ---------------------------------------------------------------------------

class TestAlignOverBudget:
    def test_tts_over_budget_11_percent_clipped_to_window(self):
        srt_window = 1000
        tts_ms = int(srt_window * 1.11)  # 11% over budget
        seg = _make_segment(start_ms=0, end_ms=srt_window)
        result = TimelineAligner().align([seg], [tts_ms])
        assert result[0].end_ms == srt_window

    def test_tts_over_budget_50_percent_clipped_to_window(self):
        srt_window = 2000
        tts_ms = int(srt_window * 1.50)
        seg = _make_segment(start_ms=500, end_ms=2500)
        result = TimelineAligner().align([seg], [tts_ms])
        assert result[0].start_ms == 500
        assert result[0].end_ms == 2500

    def test_tts_over_budget_does_not_bleed_into_next_slot(self):
        segs = [
            _make_segment(index=1, start_ms=0, end_ms=1000),
            _make_segment(index=2, start_ms=2000, end_ms=3000),
        ]
        # first segment: tts 3× over-budget
        result = TimelineAligner().align(segs, [3000, 1000])
        assert result[0].end_ms <= 1000
        assert result[1].start_ms == 2000

    def test_tts_exactly_at_window_is_not_over_budget(self):
        srt_window = 1500
        seg = _make_segment(start_ms=100, end_ms=1600)
        result = TimelineAligner().align([seg], [srt_window])
        assert result[0].end_ms - result[0].start_ms == srt_window

    def test_tts_over_budget_output_window_never_exceeds_srt_window(self):
        srt_window = 800
        for over_factor in [1.11, 1.25, 2.0, 5.0]:
            tts_ms = int(srt_window * over_factor)
            seg = _make_segment(start_ms=0, end_ms=srt_window)
            result = TimelineAligner().align([seg], [tts_ms])
            output_window = result[0].end_ms - result[0].start_ms
            assert output_window == srt_window, (
                f"over_factor={over_factor}: expected window={srt_window}, got {output_window}"
            )
