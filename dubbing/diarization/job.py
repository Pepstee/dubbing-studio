from __future__ import annotations

import json
import math
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
        provenance=document.get("provenance"),
    )


class ResumableDiarizationJob:
    """Checkpoint long diarisation and conservatively reconcile speakers globally."""

    def __init__(
        self,
        backend: DiarizationBackend,
        checkpoint_dir: str | Path,
        *,
        chunk_seconds: int = 2 * 60 * 60,
        global_speaker_threshold: float = 0.65,
        global_speaker_margin: float = 0.05,
    ) -> None:
        if not 60 <= chunk_seconds <= 3 * 60 * 60:
            raise ValueError("diarization chunk_seconds must be between 60 and 10800")
        self.backend = backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.chunk_seconds = chunk_seconds
        if not 0 < global_speaker_threshold <= 1:
            raise ValueError("global_speaker_threshold must be in (0, 1]")
        if not 0 <= global_speaker_margin < 1:
            raise ValueError("global_speaker_margin must be in [0, 1)")
        self.global_speaker_threshold = global_speaker_threshold
        self.global_speaker_margin = global_speaker_margin

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
            "speaker_identity_scope": "recording-global-embedding-cluster",
            "global_speaker_threshold": self.global_speaker_threshold,
            "global_speaker_margin": self.global_speaker_margin,
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

    @staticmethod
    def _normalise_embedding(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        embedding = tuple(float(value) for value in values)
        magnitude = math.sqrt(sum(value * value for value in embedding))
        if not embedding or magnitude <= 0:
            raise DiarizationError("speaker embedding must be a non-zero vector")
        return tuple(value / magnitude for value in embedding)

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right) or not left:
            raise DiarizationError("speaker embeddings have incompatible dimensions")
        return sum(a * b for a, b in zip(left, right))

    @classmethod
    def _centroid(cls, embeddings: list[tuple[float, ...]]) -> tuple[float, ...]:
        return cls._normalise_embedding(
            [
                sum(embedding[index] for embedding in embeddings) / len(embeddings)
                for index in range(len(embeddings[0]))
            ]
        )

    def _globalise_speakers(
        self,
        turns: list[SpeakerTurn],
        embeddings: dict[str, tuple[float, ...]],
    ) -> tuple[tuple[SpeakerTurn, ...], dict]:
        first_start = {
            speaker: min(turn.start_ms for turn in turns if turn.speaker == speaker)
            for speaker in {turn.speaker for turn in turns}
        }
        ordered_speakers = sorted(first_start, key=lambda speaker: (first_start[speaker], speaker))
        clusters: dict[str, list[tuple[float, ...]]] = {}
        mapping: dict[str, str] = {}
        decisions = []
        def chunk_scope(speaker: str) -> str:
            parts = speaker.split("_", 2)
            return "_".join(parts[:2]) if len(parts) == 3 and parts[0] == "CHUNK" else speaker

        for chunk_speaker in ordered_speakers:
            chunk_id = chunk_scope(chunk_speaker)
            forbidden = {
                global_speaker
                for local_speaker, global_speaker in mapping.items()
                if chunk_scope(local_speaker) == chunk_id
            }
            embedding = embeddings.get(chunk_speaker)
            scores = []
            if embedding is not None:
                scores = sorted(
                    (
                        (self._cosine(embedding, self._centroid(samples)), speaker)
                        for speaker, samples in clusters.items()
                        if speaker not in forbidden and samples
                    ),
                    reverse=True,
                )
            best_score = scores[0][0] if scores else None
            second_score = scores[1][0] if len(scores) > 1 else None
            margin = (
                best_score - second_score
                if best_score is not None and second_score is not None
                else None
            )
            matched = bool(
                embedding is not None
                and scores
                and best_score is not None
                and best_score >= self.global_speaker_threshold
                and (margin is None or margin >= self.global_speaker_margin)
            )
            if matched:
                global_speaker = scores[0][1]
                reason = "embedding-match"
            else:
                global_speaker = f"SPEAKER_{len(clusters):02d}"
                reason = (
                    "missing-embedding"
                    if embedding is None
                    else (
                        "below-threshold"
                        if best_score is None or best_score < self.global_speaker_threshold
                        else "ambiguous-margin"
                    )
                )
                clusters[global_speaker] = []
            mapping[chunk_speaker] = global_speaker
            if embedding is not None:
                clusters[global_speaker].append(embedding)
            decisions.append(
                {
                    "chunk_speaker": chunk_speaker,
                    "global_speaker": global_speaker,
                    "reason": reason,
                    "best_similarity": round(best_score, 6) if best_score is not None else None,
                    "second_similarity": (
                        round(second_score, 6) if second_score is not None else None
                    ),
                    "margin": round(margin, 6) if margin is not None else None,
                    "embedding_available": embedding is not None,
                }
            )
        global_turns = tuple(
            sorted(
                (
                    SpeakerTurn(
                        turn.start_ms,
                        turn.end_ms,
                        mapping[turn.speaker],
                        turn.confidence,
                    )
                    for turn in turns
                ),
                key=lambda item: (item.start_ms, item.end_ms, item.speaker),
            )
        )
        return global_turns, {
            "scope": "recording-global-embedding-cluster",
            "threshold": self.global_speaker_threshold,
            "minimum_margin": self.global_speaker_margin,
            "chunk_speaker_count": len(ordered_speakers),
            "global_speaker_count": len(clusters),
            "embedding_count": len(embeddings),
            "unresolved_count": sum(
                item["reason"] in {"missing-embedding", "ambiguous-margin"}
                for item in decisions
            ),
            "mapping": mapping,
            "decisions": decisions,
        }

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
        speaker_embeddings: dict[str, tuple[float, ...]] = {}
        backend = model = device = None
        confidence_available = False
        total_chunks = (duration_ms + chunk_ms - 1) // chunk_ms
        for index, start_ms in enumerate(range(0, duration_ms, chunk_ms)):
            end_ms = min(duration_ms, start_ms + chunk_ms)
            checkpoint = self.checkpoint_dir / "chunks" / f"{index:06d}.json"
            embeddings_checkpoint = (
                self.checkpoint_dir / "embeddings" / f"{index:06d}.json"
            )
            if checkpoint.exists():
                try:
                    result = _result_from_dict(
                        json.loads(checkpoint.read_text(encoding="utf-8"))
                    )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise DiarizationError(
                        f"diarization chunk checkpoint is malformed: {checkpoint.name}"
                    ) from exc
                try:
                    embeddings_document = json.loads(
                        embeddings_checkpoint.read_text(encoding="utf-8")
                    )
                    chunk_embeddings = {
                        speaker: self._normalise_embedding(values)
                        for speaker, values in embeddings_document["embeddings"].items()
                    }
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise DiarizationError(
                        f"diarization embedding checkpoint is malformed: {embeddings_checkpoint.name}"
                    ) from exc
            else:
                with tempfile.TemporaryDirectory(
                    prefix="dubbing-diarization-"
                ) as directory:
                    chunk = Path(directory) / f"{index:06d}.wav"
                    self._extract_chunk(source, start_ms, end_ms, chunk)
                    local = self.backend.diarize(chunk, constraints=constraints)
                    embed_speakers = getattr(self.backend, "speaker_embeddings", None)
                    local_embeddings = (
                        embed_speakers(chunk, local) if callable(embed_speakers) else {}
                    )
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
                chunk_embeddings = {
                    f"CHUNK_{index:04d}_{speaker}": self._normalise_embedding(values)
                    for speaker, values in local_embeddings.items()
                }
                _atomic_json(checkpoint, result.to_dict())
                _atomic_json(
                    embeddings_checkpoint,
                    {
                        "schema_version": "dubbing.diarization-speaker-embeddings.v1",
                        "chunk_index": index,
                        "embeddings": {
                            speaker: list(values)
                            for speaker, values in sorted(chunk_embeddings.items())
                        },
                    },
                )
            turns.extend(result.turns)
            speaker_embeddings.update(chunk_embeddings)
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

        global_turns, global_receipt = self._globalise_speakers(
            turns, speaker_embeddings
        )
        final = DiarizationResult(
            turns=global_turns,
            backend=backend or type(self.backend).__name__,
            model=f"{model or 'unspecified'}+chunked",
            device=f"chunked/{device or 'unspecified'}",
            confidence_available=confidence_available,
            provenance={"global_speaker_reconciliation": global_receipt},
        )
        _atomic_json(self.checkpoint_dir / "result.json", final.to_dict())
        return final
