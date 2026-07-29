from __future__ import annotations

from dataclasses import replace

from dubbing.diarization.attribution import attribute_window
from dubbing.diarization.models import DiarizationResult, SpeakerTurn
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionResult,
)


def _word_speaker(word: TranscriptWord, turns: tuple[SpeakerTurn, ...]) -> str | None:
    midpoint = word.start_ms + (word.end_ms - word.start_ms) // 2
    speakers = {
        turn.speaker
        for turn in turns
        if turn.start_ms <= midpoint < turn.end_ms
    }
    return next(iter(speakers)) if len(speakers) == 1 else None


def attribute_transcript(
    transcript: TranscriptionResult,
    diarization: DiarizationResult,
) -> TranscriptionResult:
    """Attach speaker labels without changing ASR timing or text."""
    segments: list[TranscriptSegment] = []
    for segment in transcript.segments:
        attribution = attribute_window(
            segment.start_ms,
            segment.end_ms,
            diarization.turns,
        )
        words = tuple(
            replace(word, speaker=_word_speaker(word, diarization.turns))
            for word in segment.words
        )
        word_speakers = {word.speaker for word in words if word.speaker is not None}
        speaker = attribution.speaker
        status = attribution.status
        if words and len(word_speakers) == 1 and all(word.speaker for word in words):
            speaker = next(iter(word_speakers))
            status = "word_attributed"
        segments.append(
            replace(
                segment,
                words=words,
                speaker=speaker,
                speakers=attribution.speakers,
                speaker_status=status,
            )
        )
    return replace(
        transcript,
        segments=tuple(segments),
        diarization=diarization.to_dict(),
    )
