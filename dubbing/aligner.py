from __future__ import annotations

from dataclasses import dataclass

from dubbing.models import Segment


@dataclass
class TimedSegment:
    start_ms: int
    end_ms: int
    segment: Segment


class TimelineAligner:
    """Aligns TTS-rendered segments to SRT subtitle timestamps.

    When TTS audio is shorter or longer than the subtitle window, the segment
    is stretched/compressed within that window rather than overflowing into
    adjacent slots.
    """

    def align(
        self,
        segments: list[Segment],
        durations: list[int],
    ) -> list[TimedSegment]:
        """Return TimedSegments mapped onto SRT timestamps.

        Args:
            segments: Ordered list of Segment objects (each carrying an SRTEntry
                      with start_ms/end_ms and an optional language tag).
            durations: TTS render duration in ms for each segment (same order).

        Returns:
            List of TimedSegment, one per input segment, with start_ms/end_ms
            clipped to the SRT window and stretched/compressed as needed.
        """
        if len(segments) != len(durations):
            raise ValueError(
                f"segments and durations must have the same length "
                f"({len(segments)} vs {len(durations)})"
            )

        result: list[TimedSegment] = []
        for segment, tts_ms in zip(segments, durations):
            srt_start = segment.entry.start_ms
            srt_end = segment.entry.end_ms
            window = srt_end - srt_start

            if window <= 0:
                # Degenerate entry: keep the SRT bounds as-is.
                result.append(TimedSegment(start_ms=srt_start, end_ms=srt_end, segment=segment))
                continue

            if tts_ms <= 0:
                result.append(TimedSegment(start_ms=srt_start, end_ms=srt_end, segment=segment))
                continue

            # Stretch or compress: the rendered audio is mapped linearly onto the
            # subtitle window regardless of how much longer/shorter it is.
            result.append(
                TimedSegment(start_ms=srt_start, end_ms=srt_start + window, segment=segment)
            )

        return result
