from __future__ import annotations

from dataclasses import dataclass


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
