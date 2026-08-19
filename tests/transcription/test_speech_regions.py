from unittest.mock import patch

import pytest

from dubbing.transcription.speech_regions import (
    FasterWhisperSileroSpeechRegionDetector,
    SpeechRegion,
    select_complementary_regions,
)


def test_complementary_sensitive_pass_cannot_swallow_precise_regions():
    strict = (
        SpeechRegion(3000, 5000, "strict"),
        SpeechRegion(7000, 8000, "strict"),
    )
    sensitive = (
        SpeechRegion(1000, 12_000, "sensitive"),
        SpeechRegion(16_000, 17_000, "sensitive"),
    )

    assert select_complementary_regions(strict, sensitive) == (
        *strict,
        SpeechRegion(16_000, 17_000, "sensitive"),
    )


def test_detector_records_both_passes_and_converts_samples_to_milliseconds(tmp_path):
    audio = tmp_path / "span.wav"
    audio.write_bytes(b"audio")
    detector = FasterWhisperSileroSpeechRegionDetector(
        strict_threshold=0.2,
        sensitive_threshold=0.1,
    )
    decoded = [0.0] * 320_000

    def timestamps(_, threshold):
        if threshold == 0.2:
            return [{"start": 16_000, "end": 32_000}]
        return [
            {"start": 8_000, "end": 48_000},
            {"start": 160_000, "end": 176_000},
        ]

    with patch.object(detector, "_decode_audio", return_value=decoded), patch.object(
        detector, "_timestamps", side_effect=timestamps
    ):
        plan = detector.detect(audio)

    assert plan.duration_ms == 20_000
    assert plan.strict_regions == (SpeechRegion(1000, 2000, "strict"),)
    assert plan.sensitive_regions == (
        SpeechRegion(500, 3000, "sensitive"),
        SpeechRegion(10_000, 11_000, "sensitive"),
    )
    assert plan.selected_regions == (
        SpeechRegion(1000, 2000, "strict"),
        SpeechRegion(10_000, 11_000, "sensitive"),
    )


@pytest.mark.parametrize(
    ("strict", "sensitive"),
    [(0.1, 0.2), (0.2, 0.2), (1.0, 0.1)],
)
def test_invalid_threshold_order_is_rejected(strict, sensitive):
    with pytest.raises(ValueError, match="thresholds"):
        FasterWhisperSileroSpeechRegionDetector(
            strict_threshold=strict,
            sensitive_threshold=sensitive,
        )
