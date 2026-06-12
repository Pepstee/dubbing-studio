"""Tests for the extracted acceptance helpers: verify_wav_header and verify_segments.

These functions are imported directly from acceptance.py and exercised with
synthetic fixtures — no CLI subprocess, no live server.
"""
from __future__ import annotations

import io
import wave

import pytest

from acceptance import verify_segments, verify_wav_header


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(duration_ms: int, rate: int = 22050) -> bytes:
    """Return real 16-bit mono WAV bytes of exactly *duration_ms* milliseconds."""
    frames = int(rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        # write silent frames so the duration is exact
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# verify_wav_header — happy paths
# ---------------------------------------------------------------------------

class TestVerifyWavHeaderHappyPath:
    def test_exact_match_returns_duration(self):
        data = _make_wav(1000)
        result = verify_wav_header(data, expected_end_ms=1000, tolerance_ms=0)
        assert result == 1000

    def test_within_positive_tolerance_returns_duration(self):
        # WAV is 50ms shorter than the expected end — within the 100ms default.
        data = _make_wav(950)
        result = verify_wav_header(data, expected_end_ms=1000, tolerance_ms=100)
        assert abs(result - 950) <= 2  # allow ±2ms rounding from frame count

    def test_within_negative_tolerance_returns_duration(self):
        # WAV is 50ms longer than the expected end — still within 100ms.
        data = _make_wav(1050)
        result = verify_wav_header(data, expected_end_ms=1000, tolerance_ms=100)
        assert abs(result - 1050) <= 2

    def test_returns_int(self):
        data = _make_wav(2000)
        result = verify_wav_header(data, expected_end_ms=2000)
        assert isinstance(result, int)

    def test_large_timeline_exact(self):
        # 17 seconds — mirrors the sample.srt timeline end used in check_cli().
        data = _make_wav(17_000)
        result = verify_wav_header(data, expected_end_ms=17_000, tolerance_ms=100)
        assert abs(result - 17_000) <= 2

    def test_custom_zero_tolerance_exact_match(self):
        data = _make_wav(500)
        result = verify_wav_header(data, expected_end_ms=500, tolerance_ms=0)
        assert result == 500

    def test_non_default_sample_rate(self):
        # 44100 Hz WAV — the function must still report the correct duration.
        data = _make_wav(2000, rate=44100)
        result = verify_wav_header(data, expected_end_ms=2000, tolerance_ms=5)
        assert abs(result - 2000) <= 5


# ---------------------------------------------------------------------------
# verify_wav_header — failure paths
# ---------------------------------------------------------------------------

class TestVerifyWavHeaderFailures:
    def test_empty_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            verify_wav_header(b"", expected_end_ms=1000)

    def test_invalid_wav_bytes_raises(self):
        with pytest.raises(Exception):  # wave.Error is an OSError subclass
            verify_wav_header(b"not a wav file at all", expected_end_ms=1000)

    def test_too_short_beyond_tolerance_raises(self):
        # 500ms WAV, expected 2000ms, tolerance 100ms → delta = 1500ms > 100ms.
        data = _make_wav(500)
        with pytest.raises(ValueError, match="not timeline-true"):
            verify_wav_header(data, expected_end_ms=2000, tolerance_ms=100)

    def test_too_long_beyond_tolerance_raises(self):
        # 3000ms WAV, expected 1000ms, tolerance 100ms → delta = 2000ms > 100ms.
        data = _make_wav(3000)
        with pytest.raises(ValueError, match="not timeline-true"):
            verify_wav_header(data, expected_end_ms=1000, tolerance_ms=100)

    def test_just_outside_tolerance_raises(self):
        # rate=1000 Hz gives exactly 1ms/frame — no quantisation rounding.
        # 1101ms WAV with 100ms tolerance: delta=101 > 100 → must raise.
        data = _make_wav(1101, rate=1000)
        with pytest.raises(ValueError, match="not timeline-true"):
            verify_wav_header(data, expected_end_ms=1000, tolerance_ms=100)

    def test_error_message_contains_actual_duration(self):
        data = _make_wav(500)
        with pytest.raises(ValueError) as exc_info:
            verify_wav_header(data, expected_end_ms=3000, tolerance_ms=0)
        msg = str(exc_info.value)
        assert "3000" in msg

    def test_zero_tolerance_rejects_off_by_one_ms(self):
        # rate=1000 Hz: 1001ms = exactly 1001 frames, no rounding.
        # At tolerance_ms=0, delta=1 > 0 → must raise.
        data = _make_wav(1001, rate=1000)
        with pytest.raises(ValueError):
            verify_wav_header(data, expected_end_ms=1000, tolerance_ms=0)

    def test_truncated_wav_header_raises(self):
        # Take a valid WAV and cut it to just the RIFF header (first 12 bytes).
        full = _make_wav(1000)
        with pytest.raises(Exception):
            verify_wav_header(full[:12], expected_end_ms=1000)


# ---------------------------------------------------------------------------
# verify_segments — happy paths
# ---------------------------------------------------------------------------

class TestVerifySegmentsHappyPath:
    def test_empty_plan_with_zero_count(self):
        verify_segments([], expected_count=0)  # must not raise

    def test_single_segment_exact(self):
        verify_segments([{"text": "Hello"}], expected_count=1)

    def test_five_segments_exact(self):
        plan = [{"index": i} for i in range(5)]
        verify_segments(plan, expected_count=5)

    def test_large_plan_exact(self):
        plan = list(range(100))
        verify_segments(plan, expected_count=100)

    def test_returns_none_on_success(self):
        result = verify_segments(["a", "b"], expected_count=2)
        assert result is None


# ---------------------------------------------------------------------------
# verify_segments — failure paths
# ---------------------------------------------------------------------------

class TestVerifySegmentsFailures:
    def test_fewer_segments_raises(self):
        with pytest.raises(ValueError, match="expected 5 segments in plan, got 3"):
            verify_segments([1, 2, 3], expected_count=5)

    def test_more_segments_raises(self):
        with pytest.raises(ValueError, match="expected 2 segments in plan, got 4"):
            verify_segments([1, 2, 3, 4], expected_count=2)

    def test_empty_plan_nonzero_expected_raises(self):
        with pytest.raises(ValueError, match="got 0"):
            verify_segments([], expected_count=1)

    def test_nonempty_plan_zero_expected_raises(self):
        with pytest.raises(ValueError, match="expected 0 segments"):
            verify_segments([{"text": "extra"}], expected_count=0)

    def test_error_message_contains_expected_count(self):
        with pytest.raises(ValueError) as exc_info:
            verify_segments(["a"], expected_count=99)
        assert "99" in str(exc_info.value)

    def test_error_message_contains_actual_count(self):
        with pytest.raises(ValueError) as exc_info:
            verify_segments(["a", "b", "c"], expected_count=1)
        assert "3" in str(exc_info.value)

    def test_off_by_one_raises(self):
        plan = list(range(4))
        with pytest.raises(ValueError):
            verify_segments(plan, expected_count=5)

    def test_heterogeneous_items_counted_correctly(self):
        # verify_segments counts len(plan), not by type — mixed items are fine.
        plan = [None, 42, "x", {}, []]
        verify_segments(plan, expected_count=5)  # should not raise
        with pytest.raises(ValueError):
            verify_segments(plan, expected_count=4)
