from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from dubbing.diarization.base import DiarizationBackend
from dubbing.diarization.models import (
    DiarizationError,
    DiarizationResult,
    SpeakerConstraints,
    SpeakerTurn,
)
from dubbing.media import ffmpeg_executable
from dubbing.transcription.job import media_duration_ms, source_sha256

_CHECKPOINT_SCHEMA = "dubbing.diarization-checkpoint.v1"


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _result_from_dict(document: dict) -> DiarizationResult:
    return DiarizationResult(
        turns=tuple(
            SpeakerTurn(
                start_ms=item["start_ms"],
                end_ms=item["end_ms"],
                speaker=item["speaker"],
                confidence=item.get("confidence"),
            )
            for item in document.get("turns", [])
        ),
        backend=document["backend"],
        model=document["model"],
        device=document["device"],
        confidence_available=document.get("confidence_available", False),
    )


class ResumableDiarizationJob:
    """Checkpoint long diarisation without asserting cross-chunk voice identity."""

    def __init__(
        self,
        backend: DiarizationBackend,
        checkpoint_dir: str | Path,
        *,
        chunk_seconds: int = 2 * 60 * 60,
    ) -> None:
        if not 60 <= chunk_seconds <= 3 * 60 * 60:
            raise ValueError("diarization chunk_seconds must be between 60 and 10800")
        self.backend = backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.chunk_seconds = chunk_seconds

    @property
    def backend_identity(self) -> str:
        identity = getattr(self.backend, "identity", None)
        if identity:
            return str(identity)
        return (
            f"{type(self.backend).__module__}.{type(self.backend).__qualname__}:"
            f"{getattr(self.backend, 'model', 'unspecified')}:"
            f"{getattr(self.backend, 'device', 'unspecified')}"
        )

    @staticmethod
    def _extract_chunk(source: Path, start_ms: int, end_ms: int, output: Path) -> None:
        ffmpeg = ffmpeg_executable()
        if ffmpeg is None:
            raise DiarizationError("ffmpeg is required for chunked diarization")
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_ms / 1000:.3f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{(end_ms - start_ms) / 1000:.3f}",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-y",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=max(600, (end_ms - start_ms) // 1000),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "unknown ffmpeg error"
            raise DiarizationError(f"could not extract diarization chunk: {detail}") from exc

    def _manifest(
        self,
        source: Path,
        duration_ms: int,
        digest: str,
        constraints: SpeakerConstraints,
    ) -> dict:
        return {
            "schema_version": _CHECKPOINT_SCHEMA,
            "source_name": source.name,
            "source_sha256": digest,
            "duration_ms": duration_ms,
            "backend_identity": self.backend_identity,
            "chunk_seconds": self.chunk_seconds,
            "speaker_constraints": {
                "num_speakers": constraints.num_speakers,
                "min_speakers": constraints.min_speakers,
                "max_speakers": constraints.max_speakers,
            },
            "speaker_identity_scope": "chunk-local",
        }

    def _admit_manifest(self, expected: dict) -> None:
        path = self.checkpoint_dir / "manifest.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DiarizationError("diarization checkpoint manifest is malformed") from exc
            if existing != expected:
                raise DiarizationError(
                    "diarization checkpoint does not match source, backend, or configuration"
                )
        else:
            _atomic_json(path, expected)

    def run(
        self,
        audio: str | Path,
        constraints: SpeakerConstraints | None = None,
        *,
        source_digest: str | None = None,
        duration_ms: int | None = None,
    ) -> DiarizationResult:
        source = Path(audio)
        if not source.is_file():
            raise DiarizationError(f"audio input not found: {source}")
        constraints = constraints or SpeakerConstraints()
        duration_ms = media_duration_ms(source) if duration_ms is None else duration_ms
        digest = source_sha256(source) if source_digest is None else source_digest
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._admit_manifest(
            self._manifest(source, duration_ms, digest, constraints)
        )

        chunk_ms = self.chunk_seconds * 1000
        turns: list[SpeakerTurn] = []
        backend = model = device = None
        confidence_available = False
        total_chunks = (duration_ms + chunk_ms - 1) // chunk_ms
        for index, start_ms in enumerate(range(0, duration_ms, chunk_ms)):
            end_ms = min(duration_ms, start_ms + chunk_ms)
            checkpoint = self.checkpoint_dir / "chunks" / f"{index:06d}.json"
            if checkpoint.exists():
                try:
                    result = _result_from_dict(
                        json.loads(checkpoint.read_text(encoding="utf-8"))
                    )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise DiarizationError(
                        f"diarization chunk checkpoint is malformed: {checkpoint.name}"
                    ) from exc
            else:
                with tempfile.TemporaryDirectory(
                    prefix="dubbing-diarization-"
                ) as directory:
                    chunk = Path(directory) / f"{index:06d}.wav"
                    self._extract_chunk(source, start_ms, end_ms, chunk)
                    local = self.backend.diarize(chunk, constraints=constraints)
                result = DiarizationResult(
                    turns=tuple(
                        SpeakerTurn(
                            start_ms=turn.start_ms + start_ms,
                            end_ms=turn.end_ms + start_ms,
                            speaker=f"CHUNK_{index:04d}_{turn.speaker}",
                            confidence=turn.confidence,
                        )
                        for turn in local.turns
                    ),
                    backend=local.backend,
                    model=local.model,
                    device=local.device,
                    confidence_available=local.confidence_available,
                )
                _atomic_json(checkpoint, result.to_dict())
            turns.extend(result.turns)
            backend, model, device = result.backend, result.model, result.device
            confidence_available = (
                confidence_available or result.confidence_available
            )
            _atomic_json(
                self.checkpoint_dir / "progress.json",
                {
                    "schema_version": "dubbing.diarization-progress.v1",
                    "completed_chunks": index + 1,
                    "total_chunks": total_chunks,
                    "completed_through_ms": end_ms,
                },
            )

        final = DiarizationResult(
            turns=tuple(
                sorted(turns, key=lambda item: (item.start_ms, item.end_ms, item.speaker))
            ),
            backend=backend or type(self.backend).__name__,
            model=f"{model or 'unspecified'}+chunked",
            device=f"chunked/{device or 'unspecified'}",
            confidence_available=confidence_available,
        )
        _atomic_json(self.checkpoint_dir / "result.json", final.to_dict())
        return final
