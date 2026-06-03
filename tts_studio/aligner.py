from __future__ import annotations

from dataclasses import dataclass

from tts_studio.models import Segment, SpeechResult


@dataclass
class AlignedEntry:
    start_ms: int
    end_ms: int
    result: SpeechResult
    stretch_ratio: float
    over_budget: bool


def align_timeline(
    segments: list[Segment], results: list[SpeechResult]
) -> list[AlignedEntry]:
    by_index = {r.segment_index: r for r in results}
    entries: list[AlignedEntry] = []
    for segment in segments:
        result = by_index.get(segment.index)
        if result is None:
            continue
        allotted_ms = segment.end_ms - segment.start_ms
        actual_ms = result.duration_ms
        stretch_ratio = actual_ms / allotted_ms if allotted_ms > 0 else float("inf")
        entries.append(
            AlignedEntry(
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                result=result,
                stretch_ratio=stretch_ratio,
                over_budget=stretch_ratio > 1.0,
            )
        )
    return entries
