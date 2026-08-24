from __future__ import annotations

from collections import defaultdict

from dubbing.aligner import TimedSegment
from dubbing.diarization.models import (
    SegmentAttribution,
    SpeakerContribution,
    SpeakerTurn,
)


def _merged_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _cross_speaker_overlap(
    intervals: dict[str, list[tuple[int, int]]],
) -> bool:
    speakers = sorted(intervals)
    for index, left_speaker in enumerate(speakers):
        for right_speaker in speakers[index + 1 :]:
            for left_start, left_end in intervals[left_speaker]:
                for right_start, right_end in intervals[right_speaker]:
                    if max(left_start, right_start) < min(left_end, right_end):
                        return True
    return False


def attribute_window(
    start_ms: int,
    end_ms: int,
    turns: tuple[SpeakerTurn, ...] | list[SpeakerTurn],
) -> SegmentAttribution:
    """Attribute one half-open segment window without changing its boundaries.

    A single overlapping speaker is assigned directly. Multiple speakers remain
    explicitly unassigned: ``status`` distinguishes simultaneous ``overlap``
    from a ``speaker_boundary`` crossed by the subtitle. Silence is ``no_speech``.
    """
    if start_ms < 0 or end_ms < start_ms:
        raise ValueError("segment window must satisfy 0 <= start_ms <= end_ms")

    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    all_intervals: list[tuple[int, int]] = []
    for turn in turns:
        overlap_start = max(start_ms, turn.start_ms)
        overlap_end = min(end_ms, turn.end_ms)
        if overlap_start < overlap_end:
            interval = (overlap_start, overlap_end)
            intervals[turn.speaker].append(interval)
            all_intervals.append(interval)

    contributions = tuple(
        SpeakerContribution(speaker=speaker, overlap_ms=_merged_duration(speaker_intervals))
        for speaker, speaker_intervals in sorted(intervals.items())
    )
    speakers = tuple(item.speaker for item in contributions)
    coverage_ms = _merged_duration(all_intervals)
    window_ms = end_ms - start_ms
    coverage_ratio = min(1.0, coverage_ms / window_ms) if window_ms else 0.0

    if not speakers:
        speaker = None
        status = "no_speech"
    elif len(speakers) == 1:
        speaker = speakers[0]
        status = "attributed"
    else:
        speaker = None
        status = "overlap" if _cross_speaker_overlap(intervals) else "speaker_boundary"

    return SegmentAttribution(
        start_ms=start_ms,
        end_ms=end_ms,
        speaker=speaker,
        speakers=speakers,
        contributions=contributions,
        speech_coverage_ms=coverage_ms,
        speech_coverage_ratio=coverage_ratio,
        status=status,
    )


def attribute_timed_segments(
    timed: list[TimedSegment],
    turns: tuple[SpeakerTurn, ...] | list[SpeakerTurn],
) -> list[SegmentAttribution]:
    """Map diarization onto existing aligned segments, preserving all timing."""
    return [attribute_window(item.start_ms, item.end_ms, turns) for item in timed]


def attributed_segment_plan(
    timed: list[TimedSegment],
    attributions: list[SegmentAttribution],
) -> list[dict]:
    if len(timed) != len(attributions):
        raise ValueError(
            "timed segments and attributions must have the same length "
            f"({len(timed)} vs {len(attributions)})"
        )
    result: list[dict] = []
    for segment, attribution in zip(timed, attributions):
        item = {
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "stretch_ratio": segment.stretch_ratio,
            "text": segment.segment.entry.text,
        }
        item.update(
            {
                "speaker": attribution.speaker,
                "speakers": list(attribution.speakers),
                "speaker_status": attribution.status,
                "speaker_contributions": [
                    contribution.to_dict() for contribution in attribution.contributions
                ],
                "speech_coverage_ms": attribution.speech_coverage_ms,
                "speech_coverage_ratio": attribution.speech_coverage_ratio,
            }
        )
        result.append(item)
    return result
