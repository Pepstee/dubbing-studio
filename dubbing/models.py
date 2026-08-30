from __future__ import annotations

import math
import os
from dataclasses import dataclass

_FALLBACK_MAX_SEGMENTS = 500
_FALLBACK_MAX_SYNTHESIS_SECONDS = 120.0
MAX_LANGUAGE_LENGTH = 20


def _env_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return value if value > 0 else fallback


def _env_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        value = float(raw)
    except ValueError:
        return fallback
    return value if math.isfinite(value) and value > 0 else fallback


DEFAULT_MAX_SEGMENTS = _env_int("DUBBING_MAX_SEGMENTS", _FALLBACK_MAX_SEGMENTS)
DEFAULT_MAX_SYNTHESIS_SECONDS = _env_float(
    "DUBBING_MAX_SYNTHESIS_SECONDS", _FALLBACK_MAX_SYNTHESIS_SECONDS
)


class SegmentLimitExceeded(Exception):
    error_code = "segment_count_exceeded"

    def __init__(
        self, message: str, *, limit: int | None = None, requested: int | None = None
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.requested = requested


class SynthesisTimeBudgetExceeded(Exception):
    error_code = "time_budget_exceeded"

    def __init__(
        self, message: str, *, limit: float | None = None, requested: float | None = None
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.requested = requested


def validate_language(value: str) -> str:
    """Return the canonical optional language token or reject unsafe input."""
    if not isinstance(value, str):
        raise TypeError("language must be a string")
    language = value.strip()
    if language != value:
        raise ValueError("language must not contain leading or trailing whitespace")
    if len(language) > MAX_LANGUAGE_LENGTH or not all(
        character.isascii() and character.isprintable() for character in language
    ):
        raise ValueError(
            f"language must be at most {MAX_LANGUAGE_LENGTH} printable ASCII characters"
        )
    return language


@dataclass(frozen=True)
class JobConfig:
    max_segments: int | None = None
    max_synthesis_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_segments is not None and (
            type(self.max_segments) is not int or self.max_segments < 0
        ):
            raise ValueError("max_segments must be a non-negative integer or None")
        if self.max_synthesis_seconds is not None:
            value = self.max_synthesis_seconds
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "max_synthesis_seconds must be a non-negative finite number or None"
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "max_synthesis_seconds must be a non-negative finite number or None"
                )


@dataclass
class SRTEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass
class ProsodyTag:
    name: str
    value: str


@dataclass
class Segment:
    entry: SRTEntry
    tags: list[ProsodyTag]
    language: str


@dataclass
class TTSResult:
    segment: Segment
    audio_bytes: bytes
    duration_ms: int
