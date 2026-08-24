from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class DiarizationError(RuntimeError):
    """Base error for actionable diarization failures."""


class UnsupportedSpeakerConstraintError(DiarizationError):
    """Raised when a backend cannot honour a requested speaker constraint."""


@dataclass(frozen=True)
class SpeakerConstraints:
    """Optional constraints supplied to a diarization backend."""

    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("num_speakers", self.num_speakers),
            ("min_speakers", self.min_speakers),
            ("max_speakers", self.max_speakers),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot exceed max_speakers")
        if self.num_speakers is not None:
            if self.min_speakers is not None and self.num_speakers < self.min_speakers:
                raise ValueError("num_speakers cannot be below min_speakers")
            if self.max_speakers is not None and self.num_speakers > self.max_speakers:
                raise ValueError("num_speakers cannot exceed max_speakers")


@dataclass(frozen=True)
class SpeakerTurn:
    """A deterministic half-open speaker interval: ``[start_ms, end_ms)``."""

    start_ms: int
    end_ms: int
    speaker: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError("speaker turn start_ms cannot be negative")
        if self.end_ms <= self.start_ms:
            raise ValueError("speaker turn end_ms must be greater than start_ms")
        if not self.speaker:
            raise ValueError("speaker label cannot be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("speaker confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, int | str | float | None]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "speaker": self.speaker,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class DiarizationResult:
    """Typed output from a real diarization backend."""

    turns: tuple[SpeakerTurn, ...]
    backend: str
    model: str
    device: str
    confidence_available: bool = False
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker)))
        if ordered != self.turns:
            raise ValueError("speaker turns must be sorted deterministically")

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(turn.speaker for turn in self.turns))

    def to_dict(self) -> dict:
        document = {
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            "confidence_available": self.confidence_available,
            "speakers": list(self.speakers),
            "turns": [turn.to_dict() for turn in self.turns],
        }
        if self.provenance is not None:
            document["provenance"] = self.provenance
        return document


@dataclass(frozen=True)
class SpeakerContribution:
    speaker: str
    overlap_ms: int

    def to_dict(self) -> dict[str, str | int]:
        return {"speaker": self.speaker, "overlap_ms": self.overlap_ms}


@dataclass(frozen=True)
class SegmentAttribution:
    """Speaker attribution for an existing segment without changing its timing."""

    start_ms: int
    end_ms: int
    speaker: str | None
    speakers: tuple[str, ...]
    contributions: tuple[SpeakerContribution, ...]
    speech_coverage_ms: int
    speech_coverage_ratio: float
    status: Literal["attributed", "no_speech", "speaker_boundary", "overlap"]

    def to_dict(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "speaker": self.speaker,
            "speakers": list(self.speakers),
            "contributions": [item.to_dict() for item in self.contributions],
            "speech_coverage_ms": self.speech_coverage_ms,
            "speech_coverage_ratio": self.speech_coverage_ratio,
            "status": self.status,
        }
