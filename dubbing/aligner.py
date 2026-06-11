from __future__ import annotations

from dataclasses import dataclass

from dubbing.models import Segment


@dataclass
class TimedSegment:
    start_ms: int
    end_ms: int
    segment: Segment
    #: Factor applied to the rendered audio's duration so it fits the SRT
    #: window: < 1.0 compresses overlong audio; 1.0 keeps natural speed
    #: (short audio is padded with silence by the assembler).
    stretch_ratio: float = 1.0


def segment_plan(timed: list[TimedSegment]) -> list[dict]:
    """Serialisable plan for a list of aligned segments (CLI/batch JSON output)."""
    return [
        {
            "start_ms": ts.start_ms,
            "end_ms": ts.end_ms,
            "stretch_ratio": ts.stretch_ratio,
            "text": ts.segment.entry.text,
        }
        for ts in timed
    ]


class TimelineAligner:
    """Aligns TTS-rendered segments to SRT subtitle timestamps.

    Output bounds always equal the SRT window, so no segment overflows into
    adjacent slots. The aligner also computes a per-segment `stretch_ratio`
    from the real synthesis duration: audio longer than its window gets a
    ratio < 1.0 (time-compress to fit); audio that fits gets 1.0 (play at
    natural speed, pad the remainder with silence).
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
            clipped to the SRT window and a stretch_ratio derived from the
            real TTS duration.
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

            # Compress overlong audio to fit the window; audio that already
            # fits plays at natural speed (ratio 1.0) and the assembler pads
            # the remainder of the window with silence.
            ratio = window / tts_ms if tts_ms > window else 1.0
            result.append(
                TimedSegment(
                    start_ms=srt_start,
                    end_ms=srt_start + window,
                    segment=segment,
                    stretch_ratio=ratio,
                )
            )

        return result
