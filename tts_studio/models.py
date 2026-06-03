from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ProsodicHint:
    tag: str
    value: str


@dataclass
class Segment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    language: str
    hints: list[ProsodicHint] = field(default_factory=list)


@dataclass
class SpeechResult:
    segment_index: int
    audio_bytes: bytes
    duration_ms: int


class TTSBackend(Protocol):
    def synthesize(self, segment: Segment) -> SpeechResult:
        ...
