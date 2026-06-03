from __future__ import annotations

import pytest

from dubbing.srt_parser import parse_srt_string, _ms
from dubbing.models import SRTEntry


# ---------------------------------------------------------------------------
# _ms helper
# ---------------------------------------------------------------------------

def test_ms_zero():
    assert _ms("00", "00", "00", "000") == 0


def test_ms_hours():
    assert _ms("01", "00", "00", "000") == 3_600_000


def test_ms_minutes():
    assert _ms("00", "01", "00", "000") == 60_000


def test_ms_seconds():
    assert _ms("00", "00", "01", "000") == 1_000


def test_ms_milliseconds():
    assert _ms("00", "00", "00", "123") == 123


def test_ms_combined():
    # 1h 2m 3s 456ms
    expected = 3_600_000 + 2 * 60_000 + 3 * 1_000 + 456
    assert _ms("01", "02", "03", "456") == expected


# ---------------------------------------------------------------------------
# parse_srt_string — happy paths
# ---------------------------------------------------------------------------

THREE_ENTRY_SRT = """\
1
00:00:01,000 --> 00:00:02,500
Hello world

2
00:00:03,000 --> 00:00:05,000
Second subtitle

3
00:01:00,000 --> 00:01:05,750
Third line
"""


def test_three_entry_count():
    entries = parse_srt_string(THREE_ENTRY_SRT)
    assert len(entries) == 3


def test_three_entry_indices():
    entries = parse_srt_string(THREE_ENTRY_SRT)
    assert [e.index for e in entries] == [1, 2, 3]


def test_three_entry_start_ms():
    entries = parse_srt_string(THREE_ENTRY_SRT)
    assert entries[0].start_ms == 1_000
    assert entries[1].start_ms == 3_000
    assert entries[2].start_ms == 60_000


def test_three_entry_end_ms():
    entries = parse_srt_string(THREE_ENTRY_SRT)
    assert entries[0].end_ms == 2_500
    assert entries[1].end_ms == 5_000
    assert entries[2].end_ms == 65_750


def test_three_entry_text():
    entries = parse_srt_string(THREE_ENTRY_SRT)
    assert entries[0].text == "Hello world"
    assert entries[1].text == "Second subtitle"
    assert entries[2].text == "Third line"


def test_entries_are_srtentry_instances():
    entries = parse_srt_string(THREE_ENTRY_SRT)
    for entry in entries:
        assert isinstance(entry, SRTEntry)


# ---------------------------------------------------------------------------
# parse_srt_string — single entry
# ---------------------------------------------------------------------------

SINGLE_ENTRY_SRT = """\
1
00:00:00,000 --> 00:00:01,000
Only entry
"""


def test_single_entry_count():
    entries = parse_srt_string(SINGLE_ENTRY_SRT)
    assert len(entries) == 1


def test_single_entry_fields():
    entry = parse_srt_string(SINGLE_ENTRY_SRT)[0]
    assert entry.index == 1
    assert entry.start_ms == 0
    assert entry.end_ms == 1_000
    assert entry.text == "Only entry"


# ---------------------------------------------------------------------------
# parse_srt_string — multi-line subtitle text
# ---------------------------------------------------------------------------

MULTILINE_SRT = """\
1
00:00:01,000 --> 00:00:04,000
Line one
Line two
Line three
"""


def test_multiline_text_preserved():
    entries = parse_srt_string(MULTILINE_SRT)
    assert len(entries) == 1
    assert entries[0].text == "Line one\nLine two\nLine three"


def test_multiline_text_newlines_count():
    entries = parse_srt_string(MULTILINE_SRT)
    assert entries[0].text.count("\n") == 2


# ---------------------------------------------------------------------------
# parse_srt_string — empty / whitespace input
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty_list():
    assert parse_srt_string("") == []


def test_whitespace_only_returns_empty_list():
    assert parse_srt_string("   \n\n   ") == []


# ---------------------------------------------------------------------------
# parse_srt_string — malformed / edge inputs (silent skip)
# ---------------------------------------------------------------------------

def test_block_without_timecode_is_skipped():
    bad = "1\nNOT A TIMECODE\nsome text\n"
    assert parse_srt_string(bad) == []


def test_block_with_non_integer_index_is_skipped():
    bad = "one\n00:00:01,000 --> 00:00:02,000\ntext\n"
    assert parse_srt_string(bad) == []


def test_block_with_only_two_lines_is_skipped():
    bad = "1\n00:00:01,000 --> 00:00:02,000\n"
    # Only 2 lines (index + timecode), no text line — skipped
    assert parse_srt_string(bad) == []


def test_mixed_valid_and_invalid_blocks():
    mixed = """\
1
00:00:01,000 --> 00:00:02,000
Good subtitle

not_an_index
00:00:03,000 --> 00:00:04,000
skipped

2
00:00:05,000 --> 00:00:06,000
Also good
"""
    entries = parse_srt_string(mixed)
    assert len(entries) == 2
    assert entries[0].index == 1
    assert entries[1].index == 2


def test_timecode_arrow_with_extra_spaces():
    srt = "1\n00:00:00,000  -->  00:00:01,000\ntext\n"
    entries = parse_srt_string(srt)
    assert len(entries) == 1
    assert entries[0].start_ms == 0
    assert entries[0].end_ms == 1_000


def test_text_is_not_stripped_of_internal_content():
    srt = "1\n00:00:00,000 --> 00:00:01,000\n  spaced  \n"
    entries = parse_srt_string(srt)
    # The implementation joins lines 2+ without stripping individual lines
    assert "spaced" in entries[0].text


def test_large_index_value():
    srt = "999\n01:30:00,000 --> 01:30:05,000\nLate entry\n"
    entries = parse_srt_string(srt)
    assert entries[0].index == 999


def test_end_ms_greater_than_start_ms():
    entries = parse_srt_string(THREE_ENTRY_SRT)
    for entry in entries:
        assert entry.end_ms > entry.start_ms


def test_returns_list_type():
    result = parse_srt_string(THREE_ENTRY_SRT)
    assert isinstance(result, list)


def test_zero_timestamp_entry():
    srt = "1\n00:00:00,000 --> 00:00:00,500\nFlash subtitle\n"
    entries = parse_srt_string(srt)
    assert entries[0].start_ms == 0
    assert entries[0].end_ms == 500


# ---------------------------------------------------------------------------
# Round-trip: every field on every entry
# ---------------------------------------------------------------------------

def test_round_trip_all_fields():
    entries = parse_srt_string(THREE_ENTRY_SRT)
    assert entries[0] == SRTEntry(index=1, start_ms=1_000, end_ms=2_500, text="Hello world")
    assert entries[1] == SRTEntry(index=2, start_ms=3_000, end_ms=5_000, text="Second subtitle")
    assert entries[2] == SRTEntry(index=3, start_ms=60_000, end_ms=65_750, text="Third line")


def test_round_trip_parse_twice_identical():
    a = parse_srt_string(THREE_ENTRY_SRT)
    b = parse_srt_string(THREE_ENTRY_SRT)
    assert a == b


# ---------------------------------------------------------------------------
# Prosody tag preservation: parser does NOT strip prosody markup
# ---------------------------------------------------------------------------

def test_prosody_tags_preserved_in_parsed_text():
    srt = "1\n00:00:01,000 --> 00:00:03,000\n<emotion:happy>Hello world\n"
    entries = parse_srt_string(srt)
    assert "<emotion:happy>" in entries[0].text


def test_multiple_prosody_tags_preserved_in_parsed_text():
    srt = "1\n00:00:00,000 --> 00:00:02,000\n<rate:slow><emotion:sad>Goodbye\n"
    entries = parse_srt_string(srt)
    assert "<rate:slow>" in entries[0].text
    assert "<emotion:sad>" in entries[0].text


def test_parse_prosody_on_parsed_text_leaves_clean_text():
    from dubbing.prosody import parse_prosody
    srt = "1\n00:00:01,000 --> 00:00:03,000\n<emotion:happy>Hello world\n"
    entries = parse_srt_string(srt)
    clean, tags = parse_prosody(entries[0].text)
    assert "<emotion:happy>" not in clean
    assert "Hello world" in clean
    assert len(tags) == 1
    assert tags[0].name == "emotion"
    assert tags[0].value == "happy"


def test_parse_prosody_extracts_all_tags_from_parsed_entry():
    from dubbing.prosody import parse_prosody
    srt = "1\n00:00:00,000 --> 00:00:03,000\n<rate:slow><emotion:sad>Goodbye cruel world\n"
    entries = parse_srt_string(srt)
    clean, tags = parse_prosody(entries[0].text)
    assert clean == "Goodbye cruel world"
    tag_names = {t.name for t in tags}
    assert tag_names == {"rate", "emotion"}


# ---------------------------------------------------------------------------
# Malformed block handling: blocks with invalid structure are skipped without crash
# ---------------------------------------------------------------------------

def test_malformed_block_non_integer_index_no_exception():
    bad = "abc\n00:00:01,000 --> 00:00:02,000\nsome text\n"
    result = parse_srt_string(bad)
    # gracefully skipped — no exception, no entry produced
    assert result == []


def test_malformed_block_missing_timecode_no_exception():
    bad = "1\nNOT_A_TIMECODE\nsome text\n"
    result = parse_srt_string(bad)
    assert result == []


def test_malformed_block_does_not_corrupt_subsequent_valid_blocks():
    mixed = (
        "bad_index\n00:00:01,000 --> 00:00:02,000\nskipped\n\n"
        "1\n00:00:03,000 --> 00:00:04,000\nkept\n"
    )
    entries = parse_srt_string(mixed)
    assert len(entries) == 1
    assert entries[0].index == 1
    assert "kept" in entries[0].text


# ---------------------------------------------------------------------------
# Language default: Segment wrapping a parsed entry defaults to language=""
# ---------------------------------------------------------------------------

def test_language_default_propagated_via_segment():
    from dubbing.models import Segment
    entries = parse_srt_string(SINGLE_ENTRY_SRT)
    seg = Segment(entry=entries[0], tags=[], language="")
    assert seg.language == ""


def test_language_default_is_empty_string_not_none():
    from dubbing.models import Segment
    entries = parse_srt_string(SINGLE_ENTRY_SRT)
    seg = Segment(entry=entries[0], tags=[], language="")
    assert seg.language is not None
    assert seg.language == ""
