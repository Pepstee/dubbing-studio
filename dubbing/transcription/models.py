from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


class TranscriptionError(RuntimeError):
    """Base error for actionable speech-to-text failures."""


@dataclass(frozen=True)
class TranscriptionOptions:
    """Provider-neutral controls understood by transcription backends."""

    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    initial_prompt: str | None = None
    word_timestamps: bool = True

    def __post_init__(self) -> None:
        if self.language is not None and not self.language.strip():
            raise ValueError("language cannot be blank")
        if self.initial_prompt is not None and not self.initial_prompt.strip():
            raise ValueError("initial_prompt cannot be blank")


@dataclass(frozen=True)
class TranscriptWord:
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None
    speaker: str | None = None

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("word timing must satisfy 0 <= start_ms < end_ms")
        if not self.text:
            raise ValueError("word text cannot be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("word confidence must be between 0 and 1")

    def shifted(self, offset_ms: int) -> TranscriptWord:
        if offset_ms < 0:
            raise ValueError("offset_ms cannot be negative")
        return replace(
            self,
            start_ms=self.start_ms + offset_ms,
            end_ms=self.end_ms + offset_ms,
        )

    def to_dict(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str
    words: tuple[TranscriptWord, ...] = ()
    confidence: float | None = None
    speaker: str | None = None
    speakers: tuple[str, ...] = ()
    speaker_status: Literal[
        "not_requested",
        "attributed",
        "no_speech",
        "speaker_boundary",
        "overlap",
        "word_attributed",
    ] = "not_requested"

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("segment timing must satisfy 0 <= start_ms < end_ms")
        if not self.text.strip():
            raise ValueError("segment text cannot be blank")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("segment confidence must be between 0 and 1")
        ordered = tuple(sorted(self.words, key=lambda item: (item.start_ms, item.end_ms)))
        if ordered != self.words:
            raise ValueError("segment words must be sorted")

    def shifted(self, offset_ms: int) -> TranscriptSegment:
        if offset_ms < 0:
            raise ValueError("offset_ms cannot be negative")
        return replace(
            self,
            start_ms=self.start_ms + offset_ms,
            end_ms=self.end_ms + offset_ms,
            words=tuple(word.shifted(offset_ms) for word in self.words),
        )

    def to_dict(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "confidence": self.confidence,
            "speaker": self.speaker,
            "speakers": list(self.speakers),
            "speaker_status": self.speaker_status,
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(frozen=True)
class TranscriptionResult:
    segments: tuple[TranscriptSegment, ...]
    text: str
    backend: str
    model: str
    device: str
    language: str | None
    duration_ms: int | None
    confidence_available: bool
    source_sha256: str | None = None
    diarization: dict | None = None

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(self.segments, key=lambda item: (item.start_ms, item.end_ms))
        )
        if ordered != self.segments:
            raise ValueError("transcript segments must be sorted")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
        if self.source_sha256 is not None and len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 hex digest")

    def shifted(self, offset_ms: int) -> TranscriptionResult:
        return replace(
            self,
            segments=tuple(segment.shifted(offset_ms) for segment in self.segments),
            duration_ms=(
                None if self.duration_ms is None else self.duration_ms + offset_ms
            ),
        )

    def to_dict(self) -> dict:
        result = {
            "schema_version": "dubbing.transcription.v1",
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "confidence_available": self.confidence_available,
            "source_sha256": self.source_sha256,
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        if self.diarization is not None:
            result["diarization"] = self.diarization
        return result


def transcription_result_from_dict(document: dict) -> TranscriptionResult:
    """Rebuild and validate a transcription result from its JSON contract."""
    if document.get("schema_version") != "dubbing.transcription.v1":
        raise ValueError("unsupported transcription schema")
    segments = []
    for item in document.get("segments", []):
        words = tuple(
            TranscriptWord(
                start_ms=word["start_ms"],
                end_ms=word["end_ms"],
                text=word["text"],
                confidence=word.get("confidence"),
                speaker=word.get("speaker"),
            )
            for word in item.get("words", [])
        )
        segments.append(
            TranscriptSegment(
                start_ms=item["start_ms"],
                end_ms=item["end_ms"],
                text=item["text"],
                words=words,
                confidence=item.get("confidence"),
                speaker=item.get("speaker"),
                speakers=tuple(item.get("speakers", [])),
                speaker_status=item.get("speaker_status", "not_requested"),
            )
        )
    return TranscriptionResult(
        segments=tuple(segments),
        text=document.get("text", ""),
        backend=document["backend"],
        model=document["model"],
        device=document["device"],
        language=document.get("language"),
        duration_ms=document.get("duration_ms"),
        confidence_available=document.get("confidence_available", False),
        source_sha256=document.get("source_sha256"),
        diarization=document.get("diarization"),
    )
