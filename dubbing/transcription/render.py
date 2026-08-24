from __future__ import annotations

from dubbing.transcription.models import TranscriptionResult


def _srt_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def transcript_to_srt(result: TranscriptionResult, *, speaker_labels: bool = True) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(result.segments, start=1):
        text = segment.text
        if speaker_labels and segment.speaker:
            text = f"[{segment.speaker}] {text}"
        blocks.append(
            "\n".join(
                (
                    str(index),
                    f"{_srt_timestamp(segment.start_ms)} --> "
                    f"{_srt_timestamp(segment.end_ms)}",
                    text,
                )
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def transcript_to_text(result: TranscriptionResult, *, speaker_labels: bool = True) -> str:
    lines: list[str] = []
    for segment in result.segments:
        prefix = f"{segment.speaker}: " if speaker_labels and segment.speaker else ""
        lines.append(f"{prefix}{segment.text}")
    return "\n".join(lines) + ("\n" if lines else "")
