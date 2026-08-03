from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dubbing.transcription.models import TranscriptionResult
from dubbing.translation.base import LanguageDetector, TranslationBackend
from dubbing.translation.models import SegmentTranslation, TranslationResult
from dubbing.translation.pipeline import translate_segment

_CHECKPOINT_SCHEMA = "dubbing.translation-checkpoint.v1"


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _segment_from_dict(document: dict) -> SegmentTranslation:
    return SegmentTranslation(
        start_ms=document["start_ms"],
        end_ms=document["end_ms"],
        speaker=document.get("speaker"),
        source_text=document["source_text"],
        source_language=document.get("source_language"),
        language_confidence=document.get("language_confidence"),
        target_text=document.get("target_text"),
        target_language=document["target_language"],
        status=document["status"],
        source_segment_id=document.get("source_segment_id"),
        source_segment_sha256=document.get("source_segment_sha256"),
    )


class ResumableTranslationJob:
    """Translate each transcript segment exactly once with source-bound checkpoints."""

    def __init__(
        self,
        detector: LanguageDetector,
        backend: TranslationBackend,
        checkpoint_dir: str | Path,
        *,
        target_language: str = "en",
        supported_source_languages: tuple[str, ...] = ("en", "ko", "ro", "ru"),
    ) -> None:
        self.detector = detector
        self.backend = backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.target_language = target_language
        self.supported_source_languages = supported_source_languages

    def _manifest(self, transcript: TranscriptionResult) -> dict:
        encoded = json.dumps(
            transcript.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return {
            "schema_version": _CHECKPOINT_SCHEMA,
            "transcript_sha256": hashlib.sha256(encoded).hexdigest(),
            "detector_identity": self.detector.identity,
            "backend_identity": self.backend.identity,
            "target_language": self.target_language,
            "supported_source_languages": list(self.supported_source_languages),
            "segment_count": len(transcript.segments),
        }

    def _admit_manifest(self, expected: dict) -> None:
        path = self.checkpoint_dir / "manifest.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("translation checkpoint manifest is malformed") from exc
            if existing != expected:
                raise RuntimeError(
                    "translation checkpoint does not match transcript or configuration"
                )
        else:
            _atomic_json(path, expected)

    def run(self, transcript: TranscriptionResult) -> TranslationResult:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._admit_manifest(self._manifest(transcript))
        translated: list[SegmentTranslation] = []
        total = len(transcript.segments)
        for index, segment in enumerate(transcript.segments):
            checkpoint = self.checkpoint_dir / "segments" / f"{index:06d}.json"
            if checkpoint.exists():
                try:
                    item = _segment_from_dict(
                        json.loads(checkpoint.read_text(encoding="utf-8"))
                    )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"translation checkpoint is malformed: {checkpoint.name}"
                    ) from exc
            else:
                item = translate_segment(
                    segment,
                    detector=self.detector,
                    backend=self.backend,
                    target_language=self.target_language,
                    supported_source_languages=self.supported_source_languages,
                )
                _atomic_json(checkpoint, item.to_dict())
            translated.append(item)
            _atomic_json(
                self.checkpoint_dir / "progress.json",
                {
                    "schema_version": "dubbing.translation-progress.v1",
                    "completed_segments": index + 1,
                    "total_segments": total,
                },
            )
        result = TranslationResult(
            target_language=self.target_language,
            backend=self.backend.identity,
            supported_source_languages=self.supported_source_languages,
            segments=tuple(translated),
        )
        _atomic_json(self.checkpoint_dir / "result.json", result.to_dict())
        return result
