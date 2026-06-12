from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _ms() — timecode component boundary cases
# (h, m, s, ms_str, expected_total_ms)
# ---------------------------------------------------------------------------

MS_BOUNDARY_CASES = [
    pytest.param("00", "00", "00", "000", 0, id="zero"),
    pytest.param("01", "00", "00", "000", 3_600_000, id="one-hour"),
    pytest.param("00", "01", "00", "000", 60_000, id="one-minute"),
    pytest.param("00", "00", "01", "000", 1_000, id="one-second"),
    pytest.param("00", "00", "00", "001", 1, id="one-ms"),
    pytest.param("00", "00", "00", "999", 999, id="max-ms-field"),
    pytest.param("23", "59", "59", "999", 86_399_999, id="near-max"),
    pytest.param("01", "02", "03", "456", 3_723_456, id="combined"),
]

# ---------------------------------------------------------------------------
# TimelineAligner stretch ratio — key threshold boundaries
# (tts_ms, srt_window_ms, expected_stretch_ratio)
# ---------------------------------------------------------------------------

STRETCH_BOUNDARY_CASES = [
    pytest.param(1000, 1000, 1.0, id="exact-fit"),
    pytest.param(500, 2000, 1.0, id="short-tts-natural-speed"),
    pytest.param(1110, 1000, 1000 / 1110, id="over-11pct"),
    pytest.param(3000, 2000, 2000 / 3000, id="over-50pct"),
    pytest.param(5000, 1000, 1000 / 5000, id="over-5x"),
]

# ---------------------------------------------------------------------------
# Over-budget factors: output window must equal SRT window for all of these
# ---------------------------------------------------------------------------

OVER_BUDGET_FACTORS = [
    pytest.param(1.01, id="1pct-over"),
    pytest.param(1.11, id="11pct-over"),
    pytest.param(1.25, id="25pct-over"),
    pytest.param(2.0, id="2x-over"),
    pytest.param(5.0, id="5x-over"),
    pytest.param(10.0, id="10x-over"),
]

# ---------------------------------------------------------------------------
# Prosody single-tag parsing — (text, expected_name, expected_value)
# ---------------------------------------------------------------------------

PROSODY_TAG_CASES = [
    pytest.param("<emotion:happy>text", "emotion", "happy", id="emotion-happy"),
    pytest.param("<emotion:sad>text", "emotion", "sad", id="emotion-sad"),
    pytest.param("<rate:slow>text", "rate", "slow", id="rate-slow"),
    pytest.param("<rate:fast>text", "rate", "fast", id="rate-fast"),
    pytest.param("<pitch:high>text", "pitch", "high", id="pitch-high"),
    pytest.param("<pitch:low>text", "pitch", "low", id="pitch-low"),
    pytest.param("<rate:100>text", "rate", "100", id="numeric-value"),
]

# ---------------------------------------------------------------------------
# SRT parse round-trip cases — (srt_text, idx, start_ms, end_ms, text)
# ---------------------------------------------------------------------------

SRT_ENTRY_CASES = [
    pytest.param(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        0, 0, 1_000, "Hello",
        id="zero-start",
    ),
    pytest.param(
        "1\n00:01:00,000 --> 00:02:00,000\nLate entry\n",
        0, 60_000, 120_000, "Late entry",
        id="minute-offset",
    ),
    pytest.param(
        "1\n01:00:00,000 --> 01:00:05,750\nHour mark\n",
        0, 3_600_000, 3_605_750, "Hour mark",
        id="hour-offset",
    ),
]
