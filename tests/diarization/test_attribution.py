from __future__ import annotations

import pytest

from dubbing.diarization import (
    SpeakerConstraints,
    SpeakerTurn,
    attribute_window,
)


def test_single_speaker_is_attributed_without_changing_timing():
    result = attribute_window(1000, 3000, [SpeakerTurn(500, 2500, "SPEAKER_00")])

    assert (result.start_ms, result.end_ms) == (1000, 3000)
    assert result.speaker == "SPEAKER_00"
    assert result.speakers == ("SPEAKER_00",)
    assert result.status == "attributed"
    assert result.speech_coverage_ms == 1500
    assert result.speech_coverage_ratio == 0.75


def test_silence_is_explicit_unknown():
    result = attribute_window(1000, 2000, [])

    assert result.speaker is None
    assert result.speakers == ()
    assert result.status == "no_speech"
    assert result.speech_coverage_ratio == 0.0


def test_boundary_between_speakers_is_not_silently_assigned():
    turns = [
        SpeakerTurn(0, 1000, "SPEAKER_00"),
        SpeakerTurn(1000, 2000, "SPEAKER_01"),
    ]

    result = attribute_window(500, 1500, turns)

    assert result.speaker is None
    assert result.speakers == ("SPEAKER_00", "SPEAKER_01")
    assert result.status == "speaker_boundary"


def test_simultaneous_speech_is_explicit_overlap():
    turns = [
        SpeakerTurn(0, 1500, "SPEAKER_00"),
        SpeakerTurn(1000, 2000, "SPEAKER_01"),
    ]

    result = attribute_window(500, 1800, turns)

    assert result.speaker is None
    assert result.status == "overlap"
    assert [item.overlap_ms for item in result.contributions] == [1000, 800]
    assert result.speech_coverage_ms == 1300
    assert result.speech_coverage_ratio == 1.0


def test_adjacent_half_open_turn_does_not_leak_across_boundary():
    result = attribute_window(
        0,
        1000,
        [
            SpeakerTurn(0, 1000, "SPEAKER_00"),
            SpeakerTurn(1000, 2000, "SPEAKER_01"),
        ],
    )

    assert result.speaker == "SPEAKER_00"
    assert result.speakers == ("SPEAKER_00",)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"num_speakers": 0}, "at least 1"),
        ({"min_speakers": 3, "max_speakers": 2}, "cannot exceed"),
        ({"num_speakers": 2, "min_speakers": 3}, "cannot be below"),
    ],
)
def test_invalid_speaker_constraints_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SpeakerConstraints(**kwargs)
