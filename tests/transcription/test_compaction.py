import wave

import pytest

from dubbing.media import ffmpeg_executable
from dubbing.transcription.compaction import (
    build_speech_compaction_plan,
    extract_compacted_flac,
)
from dubbing.transcription.job import media_duration_ms
from dubbing.transcription.models import TranscriptSegment, TranscriptionError, TranscriptionResult
from dubbing.transcription.speech_regions import SpeechRegion, SpeechRegionPlan


def _local(duration_ms, segments=()):
    return TranscriptionResult(
        segments=tuple(segments),
        text=" ".join(item.text for item in segments),
        backend="local",
        model="fixture",
        device="cpu",
        language=None,
        duration_ms=duration_ms,
        confidence_available=False,
        source_sha256="a" * 64,
    )


def _regions(duration_ms, *, strict=(), sensitive=()):
    return SpeechRegionPlan(
        "fixture-vad",
        duration_ms,
        tuple(strict),
        tuple(sensitive),
        tuple(strict),
    )


def test_compaction_uses_sensitive_vad_and_restores_original_timestamps():
    local = _local(20_000, (TranscriptSegment(5_200, 6_800, "three spoken words"),))
    regions = _regions(
        20_000,
        strict=(SpeechRegion(5_200, 6_800, "strict"),),
        sensitive=(SpeechRegion(5_000, 7_000, "sensitive"),),
    )
    plan = build_speech_compaction_plan(
        regions,
        local,
        padding_ms=1_000,
        merge_gap_ms=500,
        separator_ms=500,
        maximum_packet_ms=60_000,
    )
    assert [(item.start_ms, item.end_ms) for item in plan.retained_intervals] == [
        (4_000, 8_000)
    ]
    assert plan.uploaded_ms == 4_000
    assert plan.removed_ms == 16_000
    assert plan.packets[0].restore_interval(1_000, 1_500) == (5_000, 5_500)


def test_inserted_separator_has_no_original_time_and_cannot_be_admitted():
    local = _local(20_000)
    regions = _regions(
        20_000,
        sensitive=(
            SpeechRegion(1_000, 2_000, "sensitive"),
            SpeechRegion(10_000, 11_000, "sensitive"),
        ),
    )
    plan = build_speech_compaction_plan(
        regions,
        local,
        padding_ms=0,
        merge_gap_ms=0,
        separator_ms=500,
        maximum_packet_ms=60_000,
        preserve_diarization_context=False,
    )
    packet = plan.packets[0]
    assert packet.compact_duration_ms == 2_500
    assert packet.restore_interval(900, 1_100) is None
    assert packet.restore_interval(1_100, 1_200) is None
    assert packet.restore_interval(1_500, 2_000) == (10_000, 10_500)


def test_no_vad_or_local_evidence_falls_back_to_full_audio():
    plan = build_speech_compaction_plan(
        _regions(30_000),
        _local(30_000),
        maximum_packet_ms=60_000,
    )
    assert plan.fallback_reason == "NO_SPEECH_EVIDENCE_FULL_AUDIO_RETAINED"
    assert plan.uploaded_ms == 30_000
    assert plan.removed_ms == 0


def test_vad_duration_mismatch_fails_before_compaction():
    with pytest.raises(TranscriptionError, match="duration does not match"):
        build_speech_compaction_plan(
            _regions(10_000),
            _local(20_000),
            maximum_packet_ms=60_000,
        )


def test_long_retained_audio_is_split_into_bounded_cloud_packets():
    local = _local(130_000)
    regions = _regions(
        130_000,
        sensitive=(SpeechRegion(0, 130_000, "sensitive"),),
    )
    plan = build_speech_compaction_plan(
        regions,
        local,
        padding_ms=0,
        merge_gap_ms=0,
        separator_ms=500,
        maximum_packet_ms=60_000,
    )
    assert [item.compact_duration_ms for item in plan.packets] == [60_000, 60_000, 10_000]
    assert plan.uploaded_ms == 130_000


def test_packet_slice_count_is_bounded_for_highly_fragmented_speech():
    regions = tuple(
        SpeechRegion(index * 2_000, index * 2_000 + 500, "sensitive")
        for index in range(65)
    )
    plan = build_speech_compaction_plan(
        _regions(130_000, sensitive=regions),
        _local(130_000),
        padding_ms=0,
        merge_gap_ms=0,
        separator_ms=100,
        maximum_packet_ms=60_000,
        maximum_slices_per_packet=64,
        preserve_diarization_context=False,
    )
    assert [len(item.slices) for item in plan.packets] == [64, 1]
    assert [item.compact_duration_ms for item in plan.packets] == [38_300, 500]


def test_diarization_safe_compaction_never_joins_discontiguous_source_audio():
    regions = _regions(
        30_000,
        sensitive=(
            SpeechRegion(1_000, 4_000, "sensitive"),
            SpeechRegion(20_000, 23_000, "sensitive"),
        ),
    )
    plan = build_speech_compaction_plan(
        regions,
        _local(30_000),
        padding_ms=0,
        merge_gap_ms=5_000,
        separator_ms=500,
        maximum_packet_ms=60_000,
    )
    assert plan.preserve_diarization_context is True
    assert [len(item.slices) for item in plan.packets] == [1, 1]
    assert [item.original_retained_ms for item in plan.packets] == [3_000, 3_000]
    assert all(item.compact_duration_ms == item.original_retained_ms for item in plan.packets)


def test_diarization_context_retains_natural_pause_as_contiguous_audio():
    regions = _regions(
        20_000,
        sensitive=(
            SpeechRegion(1_000, 3_000, "sensitive"),
            SpeechRegion(10_000, 12_000, "sensitive"),
        ),
    )
    plan = build_speech_compaction_plan(
        regions,
        _local(20_000),
        padding_ms=0,
        merge_gap_ms=15_000,
        separator_ms=500,
        maximum_packet_ms=60_000,
    )
    assert [(item.start_ms, item.end_ms) for item in plan.retained_intervals] == [
        (1_000, 12_000)
    ]
    assert len(plan.packets) == 1
    assert plan.packets[0].slices[0].original_start_ms == 1_000
    assert plan.packets[0].slices[0].original_end_ms == 12_000


def test_padding_does_not_expand_the_diarization_merge_threshold():
    regions = _regions(
        48_346,
        sensitive=(
            SpeechRegion(0, 5_956, "sensitive"),
            SpeechRegion(23_983, 48_346, "sensitive"),
        ),
    )
    plan = build_speech_compaction_plan(
        regions,
        _local(48_346),
        padding_ms=1_500,
        merge_gap_ms=15_000,
        separator_ms=500,
        maximum_packet_ms=60_000,
    )
    assert 23_983 - 5_956 == 18_027
    assert [(item.start_ms, item.end_ms) for item in plan.retained_intervals] == [
        (0, 7_456),
        (22_483, 48_346),
    ]
    assert [len(item.slices) for item in plan.packets] == [1, 1]
    assert plan.removed_ms == 15_027


def test_real_ffmpeg_compaction_matches_the_mapping_duration(tmp_path):
    if ffmpeg_executable() is None:
        pytest.skip("ffmpeg is not installed")
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 16_000 * 4)
    regions = _regions(
        4_000,
        sensitive=(
            SpeechRegion(0, 1_000, "sensitive"),
            SpeechRegion(3_000, 4_000, "sensitive"),
        ),
    )
    plan = build_speech_compaction_plan(
        regions,
        _local(4_000),
        padding_ms=0,
        merge_gap_ms=0,
        separator_ms=500,
        maximum_packet_ms=60_000,
        preserve_diarization_context=False,
    )
    output = tmp_path / "compact.flac"
    extract_compacted_flac(source, plan.packets[0], output)
    assert output.is_file()
    assert abs(media_duration_ms(output) - 2_500) <= 50
