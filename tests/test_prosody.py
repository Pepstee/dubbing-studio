from __future__ import annotations

import pytest

from dubbing.prosody import parse_prosody
from dubbing.models import ProsodyTag


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------

def test_returns_tuple():
    result = parse_prosody("hello")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_clean_text_is_str():
    clean, _ = parse_prosody("hello")
    assert isinstance(clean, str)


def test_tags_is_list():
    _, tags = parse_prosody("hello")
    assert isinstance(tags, list)


# ---------------------------------------------------------------------------
# No tags present
# ---------------------------------------------------------------------------

def test_plain_text_unchanged():
    clean, tags = parse_prosody("Just plain text.")
    assert clean == "Just plain text."
    assert tags == []


def test_empty_string():
    clean, tags = parse_prosody("")
    assert clean == ""
    assert tags == []


def test_whitespace_only():
    clean, tags = parse_prosody("   ")
    assert clean == "   "
    assert tags == []


# ---------------------------------------------------------------------------
# Single tag removal
# ---------------------------------------------------------------------------

def test_emotion_tag_removed_from_text():
    clean, _ = parse_prosody("<emotion:happy>Hello there")
    assert "<emotion:happy>" not in clean


def test_emotion_tag_clean_text():
    clean, _ = parse_prosody("<emotion:happy>Hello there")
    assert clean == "Hello there"


def test_emotion_tag_captured():
    _, tags = parse_prosody("<emotion:happy>Hello there")
    assert len(tags) == 1
    assert tags[0].name == "emotion"
    assert tags[0].value == "happy"


def test_rate_tag_removed_from_text():
    clean, _ = parse_prosody("<rate:slow>Speak slowly")
    assert "<rate:slow>" not in clean


def test_rate_tag_clean_text():
    clean, _ = parse_prosody("<rate:slow>Speak slowly")
    assert clean == "Speak slowly"


def test_rate_tag_captured():
    _, tags = parse_prosody("<rate:slow>Speak slowly")
    assert len(tags) == 1
    assert tags[0].name == "rate"
    assert tags[0].value == "slow"


# ---------------------------------------------------------------------------
# Multiple tags
# ---------------------------------------------------------------------------

def test_two_tags_both_removed():
    clean, tags = parse_prosody("<emotion:happy><rate:slow>Good morning")
    assert "<emotion:happy>" not in clean
    assert "<rate:slow>" not in clean
    assert clean == "Good morning"


def test_two_tags_both_captured():
    _, tags = parse_prosody("<emotion:happy><rate:slow>Good morning")
    assert len(tags) == 2


def test_two_tags_correct_names():
    _, tags = parse_prosody("<emotion:happy><rate:slow>Good morning")
    names = {t.name for t in tags}
    assert names == {"emotion", "rate"}


def test_two_tags_correct_values():
    _, tags = parse_prosody("<emotion:happy><rate:slow>Good morning")
    by_name = {t.name: t.value for t in tags}
    assert by_name["emotion"] == "happy"
    assert by_name["rate"] == "slow"


def test_tag_in_middle_of_text():
    clean, tags = parse_prosody("Say <emotion:sad>goodbye now")
    assert clean == "Say goodbye now"
    assert tags[0].name == "emotion"
    assert tags[0].value == "sad"


def test_tag_at_end_of_text():
    clean, tags = parse_prosody("The end<rate:fast>")
    assert clean == "The end"
    assert tags[0].name == "rate"
    assert tags[0].value == "fast"


def test_tags_order_matches_appearance():
    _, tags = parse_prosody("<emotion:happy><rate:slow><pitch:high>text")
    assert [t.name for t in tags] == ["emotion", "rate", "pitch"]


def test_three_tags_all_removed():
    clean, tags = parse_prosody("<emotion:happy><rate:slow><pitch:high>text")
    assert clean == "text"
    assert len(tags) == 3


# ---------------------------------------------------------------------------
# ProsodyTag type check
# ---------------------------------------------------------------------------

def test_tags_are_prosodytag_instances():
    _, tags = parse_prosody("<emotion:happy>hello")
    assert all(isinstance(t, ProsodyTag) for t in tags)


# ---------------------------------------------------------------------------
# Non-matching angle-bracket constructs left intact
# ---------------------------------------------------------------------------

def test_html_like_tag_without_colon_not_consumed():
    # <b> has no colon so the regex won't match it
    clean, tags = parse_prosody("<b>bold</b>")
    assert tags == []
    assert "<b>" in clean


def test_numeric_value_tag_not_consumed():
    # <rate:100> — value is \w+ which includes digits, so this WILL match
    clean, tags = parse_prosody("<rate:100>text")
    assert len(tags) == 1
    assert tags[0].name == "rate"
    assert tags[0].value == "100"


def test_tag_with_spaces_not_consumed():
    # spaces aren't \w, so <emotion: happy> won't match
    clean, tags = parse_prosody("<emotion: happy>text")
    assert tags == []
    assert "<emotion: happy>" in clean


# ---------------------------------------------------------------------------
# Surrounding text is fully preserved
# ---------------------------------------------------------------------------

def test_text_before_tag_preserved():
    clean, _ = parse_prosody("Before <emotion:happy> after")
    assert "Before" in clean
    assert "after" in clean


def test_no_extra_whitespace_introduced():
    # The regex just removes the tag; it does not collapse whitespace
    clean, _ = parse_prosody("Hello<emotion:happy>world")
    assert clean == "Helloworld"
