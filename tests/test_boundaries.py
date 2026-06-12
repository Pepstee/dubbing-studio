from __future__ import annotations

import pytest

from dubbing.aligner import TimelineAligner
from dubbing.prosody import parse_prosody
from dubbing.srt_parser import _ms, parse_srt_string

from tests.support.assertions import (
    assert_prosody_tags,
    assert_timed_segment,
    assert_wav_duration,
    assert_wav_format,
)
from tests.support.builders import make_segment, make_wav
from tests.support.params import (
    MS_BOUNDARY_CASES,
    OVER_BUDGET_FACTORS,
    PROSODY_TAG_CASES,
    SRT_ENTRY_CASES,
    STRETCH_BOUNDARY_CASES,
)


# ---------------------------------------------------------------------------
# _ms() — timecode component boundaries (parametrised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h,m,s,ms_str,expected", MS_BOUNDARY_CASES)
def test_ms_boundary(h, m, s, ms_str, expected):
    assert _ms(h, m, s, ms_str) == expected


# ---------------------------------------------------------------------------
# parse_srt_string — round-trip field values (parametrised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("srt,entry_idx,start_ms,end_ms,text", SRT_ENTRY_CASES)
def test_srt_entry_fields(srt, entry_idx, start_ms, end_ms, text):
    entries = parse_srt_string(srt)
    e = entries[entry_idx]
    assert e.start_ms == start_ms
    assert e.end_ms == end_ms
    assert e.text == text


# ---------------------------------------------------------------------------
# TimelineAligner — stretch ratio at key boundaries (parametrised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tts_ms,window_ms,expected_ratio", STRETCH_BOUNDARY_CASES)
def test_aligner_stretch_ratio(tts_ms, window_ms, expected_ratio):
    seg = make_segment(start_ms=0, end_ms=window_ms)
    result = TimelineAligner().align([seg], [tts_ms])
    assert result[0].stretch_ratio == pytest.approx(expected_ratio)


# ---------------------------------------------------------------------------
# TimelineAligner — output window never exceeds SRT window (parametrised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factor", OVER_BUDGET_FACTORS)
def test_over_budget_window_stays_within_srt(factor):
    srt_window = 1000
    tts_ms = int(srt_window * factor)
    seg = make_segment(start_ms=0, end_ms=srt_window)
    result = TimelineAligner().align([seg], [tts_ms])
    assert_timed_segment(result[0], start_ms=0, end_ms=srt_window)


# ---------------------------------------------------------------------------
# TimelineAligner — assert_timed_segment helper with non-zero origin
# ---------------------------------------------------------------------------

def test_timed_segment_non_zero_origin():
    seg = make_segment(start_ms=5000, end_ms=8000)
    result = TimelineAligner().align([seg], [3000])
    assert_timed_segment(result[0], start_ms=5000, end_ms=8000, stretch_ratio=1.0)


def test_timed_segment_compressed_stretch_ratio():
    seg = make_segment(start_ms=0, end_ms=1000)
    result = TimelineAligner().align([seg], [2000])
    assert_timed_segment(result[0], start_ms=0, end_ms=1000, stretch_ratio=0.5)


# ---------------------------------------------------------------------------
# parse_prosody — single-tag boundaries (parametrised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_name,expected_value", PROSODY_TAG_CASES)
def test_prosody_single_tag(text, expected_name, expected_value):
    clean, tags = parse_prosody(text)
    assert_prosody_tags(tags, names=[expected_name], values={expected_name: expected_value})
    assert f"<{expected_name}:{expected_value}>" not in clean


# ---------------------------------------------------------------------------
# assert_wav_duration / assert_wav_format helpers — self-checks
# ---------------------------------------------------------------------------

def test_assert_wav_duration_exact():
    data = make_wav(500)
    assert_wav_duration(data, 500)


def test_assert_wav_duration_within_tolerance():
    data = make_wav(1000)
    assert_wav_duration(data, 1001, tolerance_ms=2)


def test_assert_wav_format_default():
    data = make_wav(200)
    assert_wav_format(data, channels=1, sampwidth=2, rate=22050)


def test_assert_wav_duration_fails_outside_tolerance():
    data = make_wav(1000)
    with pytest.raises(AssertionError):
        assert_wav_duration(data, 500, tolerance_ms=2)


def test_assert_wav_format_fails_wrong_rate():
    data = make_wav(200, rate=44100)
    with pytest.raises(AssertionError):
        assert_wav_format(data, rate=22050)
