from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentTranslation:
    start_ms: int
    end_ms: int
    source_text: str
    source_language: str | None
    language_confidence: float | None
    target_text: str | None
    target_language: str
    status: str
    speaker: str | None = None
    source_segment_id: str | None = None
    source_segment_sha256: str | None = None

    def to_dict(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "speaker": self.speaker,
            "source_text": self.source_text,
            "source_language": self.source_language,
            "language_confidence": self.language_confidence,
            "target_text": self.target_text,
            "target_language": self.target_language,
            "status": self.status,
            "source_segment_id": self.source_segment_id,
            "source_segment_sha256": self.source_segment_sha256,
            "original_language_authoritative": True,
        }


def source_segment_identity(start_ms: int, end_ms: int, text: str) -> tuple[str, str]:
    encoded = f"{start_ms}:{end_ms}:{text}".encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"segment:{start_ms}-{end_ms}:{digest[:16]}", digest


@dataclass(frozen=True)
class TranslationResult:
    target_language: str
    backend: str
    supported_source_languages: tuple[str, ...]
    segments: tuple[SegmentTranslation, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": "dubbing.translation.v1",
            "target_language": self.target_language,
            "backend": self.backend,
            "supported_source_languages": list(self.supported_source_languages),
            "segments": [segment.to_dict() for segment in self.segments],
        }
