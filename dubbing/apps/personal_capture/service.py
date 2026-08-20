from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from dubbing.apps.personal_capture.store import CaptureRecord, CaptureStore
from dubbing.diarization.base import DiarizationBackend
from dubbing.diarization.job import ResumableDiarizationJob
from dubbing.diarization.models import SpeakerConstraints
from dubbing.diarization.quality import (
    DiarizationQualityReport,
    DiarizationQualityStatus,
    evaluate_diarization_quality,
)
from dubbing.transcription import (
    AudioUnderstandingPipeline,
    ResumableTranscriptionJob,
    SpeechRegionDetector,
    TranscriptionBackend,
    TranscriptionOptions,
    attribute_transcript,
    transcription_result_from_dict,
    transcript_to_srt,
    transcript_to_text,
)
from dubbing.transcription.job import (
    create_source_binding,
    media_duration_ms,
    validate_source_binding,
)
from dubbing.transcription.adaptive import (
    AdaptiveChunkPlanner,
    AdaptiveLongFormCoordinator,
)
from dubbing.transcription.quality import (
    TranscriptQualityReport,
    TranscriptQualityStatus,
    evaluate_transcript_quality,
)
from dubbing.translation import (
    LanguageDetector,
    ResumableTranslationJob,
    TranslationBackend,
    translate_transcript,
)
from dubbing.translation.models import source_segment_identity

_PACKAGE_SCHEMA = "dubbing.personal-capture.v1"
_GIGA_EVENT_SCHEMA = "giga.personal-capture-event.v1"
_MEDIA_SUFFIXES = {
    ".aac", ".aiff", ".avi", ".flac", ".m4a", ".m4v", ".mkv", ".mov",
    ".mp3", ".mp4", ".mpeg", ".mpg", ".oga", ".ogg", ".opus", ".wav",
    ".weba", ".webm", ".wmv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(document: Mapping) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _translation_to_text(document: Mapping) -> str:
    lines = [
        segment.get("target_text")
        or f"[{segment.get('status', 'unknown')}] {segment.get('source_text', '')}"
        for segment in document.get("segments", [])
    ]
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def _speaker_aliases(aliases: Mapping[str, str] | None) -> dict[str, str]:
    normalized = {}
    for raw_key, raw_value in (aliases or {}).items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key or not value:
            continue
        if len(key) > 100 or len(value) > 200:
            raise ValueError("speaker aliases exceed the allowed length")
        normalized[key] = value
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class CaptureOutcome:
    capture_id: str
    source_name: str
    state: str
    package_path: Path | None
    replayed: bool = False
    error: str | None = None


class _UnsetSpeechRegionDetector:
    pass


_UNSET_SPEECH_REGION_DETECTOR = _UnsetSpeechRegionDetector()


class CaptureService:
    """Turn stable inbox media into reviewable, provenance-bearing packages."""

    def __init__(
        self,
        workspace: str | Path,
        transcription_backend: TranscriptionBackend | None = None,
        *,
        transcription_retry_backend: TranscriptionBackend | None = None,
        speech_region_detector: SpeechRegionDetector | None = None,
        silence_verification_detector: (
            SpeechRegionDetector | None | _UnsetSpeechRegionDetector
        ) = _UNSET_SPEECH_REGION_DETECTOR,
        targeted_retry_region_detector: (
            SpeechRegionDetector | None | _UnsetSpeechRegionDetector
        ) = _UNSET_SPEECH_REGION_DETECTOR,
        diarizer: DiarizationBackend | None = None,
        diarization_embedding_backend: DiarizationBackend | None = None,
        speaker_constraints: SpeakerConstraints | None = None,
        transcription_options: TranscriptionOptions | None = None,
        language_detector: LanguageDetector | None = None,
        translation_backend: TranslationBackend | None = None,
        translation_target: str = "en",
        packages_dir: str | Path = "outputs/packages",
        state_dir: str | Path = "state",
        processing_dir: str | Path = "processing",
        resumable: bool = True,
        transcription_strategy: str = "fixed",
        transcription_chunk_seconds: int = 30 * 60,
        transcription_minimum_chunk_seconds: int = 60,
        transcription_maximum_chunk_seconds: int = 8 * 60,
        transcription_overlap_seconds: int = 5,
        transcription_minimum_silence_seconds: float = 0.7,
        audio_candidate_policies: tuple[str, ...] = (
            "raw",
            "downmix",
            "channels",
        ),
        maximum_audio_candidate_channels: int = 4,
        diarization_chunk_seconds: int = 2 * 60 * 60,
        diarization_global_speaker_threshold: float = 0.80,
        diarization_global_speaker_margin: float = 0.05,
        minimum_free_bytes: int = 0,
        maximum_audio_seconds: int = 24 * 60 * 60,
        outbox_dir: str | Path | None = None,
        inbox_dir: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.packages = self._workspace_path(packages_dir)
        self.processing = self._workspace_path(processing_dir)
        state = self._workspace_path(state_dir)
        state.mkdir(parents=True, exist_ok=True)
        self.store = CaptureStore(state / "capture.sqlite3")
        self.transcription_backend = transcription_backend
        self.transcription_retry_backend = transcription_retry_backend
        self.speech_region_detector = speech_region_detector
        self.silence_verification_detector = (
            speech_region_detector
            if isinstance(silence_verification_detector, _UnsetSpeechRegionDetector)
            else silence_verification_detector
        )
        self.targeted_retry_region_detector = (
            speech_region_detector
            if isinstance(targeted_retry_region_detector, _UnsetSpeechRegionDetector)
            else targeted_retry_region_detector
        )
        self.resumable = resumable
        if transcription_strategy not in {"adaptive", "fixed"}:
            raise ValueError("transcription_strategy must be adaptive or fixed")
        self.transcription_strategy = transcription_strategy
        self.transcription_chunk_seconds = transcription_chunk_seconds
        self.transcription_minimum_chunk_seconds = transcription_minimum_chunk_seconds
        self.transcription_maximum_chunk_seconds = transcription_maximum_chunk_seconds
        self.transcription_overlap_seconds = transcription_overlap_seconds
        self.transcription_minimum_silence_seconds = (
            transcription_minimum_silence_seconds
        )
        self.audio_candidate_policies = audio_candidate_policies
        self.maximum_audio_candidate_channels = maximum_audio_candidate_channels
        self.diarization_chunk_seconds = diarization_chunk_seconds
        self.diarization_global_speaker_threshold = (
            diarization_global_speaker_threshold
        )
        self.diarization_global_speaker_margin = diarization_global_speaker_margin
        if minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes cannot be negative")
        self.minimum_free_bytes = minimum_free_bytes
        if not 1 <= maximum_audio_seconds <= 24 * 60 * 60:
            raise ValueError("maximum_audio_seconds must be between 1 and 86400")
        self.maximum_audio_seconds = maximum_audio_seconds
        self.outbox_dir = Path(outbox_dir).resolve() if outbox_dir else None
        self.inbox_dir = Path(inbox_dir).resolve() if inbox_dir else None
        self.packages_dir = Path(packages_dir)
        self.diarizer = diarizer
        self.diarization_embedding_backend = diarization_embedding_backend
        self.speaker_constraints = speaker_constraints
        self.transcription_options = transcription_options or TranscriptionOptions()
        if (language_detector is None) != (translation_backend is None):
            raise ValueError(
                "language_detector and translation_backend must be configured together"
            )
        self.language_detector = language_detector
        self.translation_backend = translation_backend
        self.translation_target = translation_target

    def _workspace_path(self, relative: str | Path) -> Path:
        path = (self.workspace / relative).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError(f"capture path escapes workspace: {relative}")
        return path

    @staticmethod
    def discover(
        inbox: str | Path,
        *,
        min_age_seconds: float = 30,
        now: float | None = None,
    ) -> tuple[Path, ...]:
        root = Path(inbox).resolve()
        if not root.is_dir():
            raise ValueError(f"capture inbox is not a directory: {root}")
        cutoff = (time.time() if now is None else now) - min_age_seconds
        discovered = []
        for candidate in sorted(root.iterdir()):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if candidate.suffix.lower() not in _MEDIA_SUFFIXES:
                continue
            if candidate.stat().st_mtime > cutoff:
                continue
            discovered.append(candidate)
        return tuple(discovered)

    def _write_package(
        self,
        source: Path,
        record: CaptureRecord,
        transcript,
        translation=None,
        quality_report: TranscriptQualityReport | None = None,
    ) -> Path:
        destination = self.packages / record.capture_id
        staging = self.packages / f".{record.capture_id}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)
        try:
            manifest = {
                "schema_version": _PACKAGE_SCHEMA,
                "capture_id": record.capture_id,
                "state": "review",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "name": source.name,
                    "sha256": record.source_sha256,
                    "size": record.source_size,
                    "mtime_ns": record.source_mtime_ns,
                    "copied": False,
                    "verified_unchanged_after_processing": True,
                },
                "execution": {
                    "resumable": self.resumable,
                    "transcription_strategy": self.transcription_strategy,
                    "transcription_chunk_seconds": self.transcription_chunk_seconds,
                    "transcription_minimum_chunk_seconds": self.transcription_minimum_chunk_seconds,
                    "transcription_maximum_chunk_seconds": self.transcription_maximum_chunk_seconds,
                    "transcription_overlap_seconds": self.transcription_overlap_seconds,
                    "transcription_minimum_silence_seconds": self.transcription_minimum_silence_seconds,
                    "audio_candidate_policies": list(self.audio_candidate_policies),
                    "maximum_audio_candidate_channels": self.maximum_audio_candidate_channels,
                    "transcription_backend": (
                        self.transcription_backend.identity
                        if self.transcription_backend is not None
                        else None
                    ),
                    "transcription_retry_backend": (
                        self.transcription_retry_backend.identity
                        if self.transcription_retry_backend is not None
                        else None
                    ),
                    "speech_region_detector": (
                        self.speech_region_detector.identity
                        if self.speech_region_detector is not None
                        else None
                    ),
                    "silence_verification_detector": (
                        self.silence_verification_detector.identity
                        if self.silence_verification_detector is not None
                        else None
                    ),
                    "targeted_retry_region_detector": (
                        self.targeted_retry_region_detector.identity
                        if self.targeted_retry_region_detector is not None
                        else None
                    ),
                    "diarization_chunk_seconds": self.diarization_chunk_seconds,
                    "diarization_backend": (
                        self.diarizer.identity if self.diarizer is not None else None
                    ),
                    "diarization_global_speaker_threshold": (
                        self.diarization_global_speaker_threshold
                    ),
                    "diarization_global_speaker_margin": (
                        self.diarization_global_speaker_margin
                    ),
                    "speaker_identity_scope": (
                        "recording-global-embedding-cluster"
                        if self.resumable and self.diarizer
                        else "recording"
                    ),
                },
                "transcript": {
                    "json": "transcript.json",
                    "text": "transcript.txt",
                    "srt": "transcript.srt",
                },
                "translation": None,
                "quality": (
                    {
                        "report": "quality-report.json",
                        "status": quality_report.status.value,
                        "policy_version": quality_report.policy_version,
                    }
                    if quality_report is not None
                    else None
                ),
                "diarization_quality": None,
                "review": {
                    "required": True,
                    "speaker_aliases": {},
                    "notes": "",
                },
            }
            (staging / "transcript.json").write_text(
                _json(transcript.to_dict()),
                encoding="utf-8",
            )
            (staging / "transcript.txt").write_text(
                transcript_to_text(transcript),
                encoding="utf-8",
            )
            (staging / "transcript.srt").write_text(
                transcript_to_srt(transcript),
                encoding="utf-8",
            )
            if self.diarizer is not None:
                diarization_path = staging / "diarization.json"
                diarization_path.write_text(
                    _json(transcript.diarization or {}), encoding="utf-8"
                )
                diarization_report = evaluate_diarization_quality(
                    transcript,
                    transcript_sha256=_sha256(staging / "transcript.json"),
                    diarization_sha256=_sha256(diarization_path),
                )
                diarization_report_path = staging / "diarization-quality-report.json"
                diarization_report_path.write_text(
                    _json(diarization_report.to_dict()), encoding="utf-8"
                )
                manifest["diarization_quality"] = {
                    "report": diarization_report_path.name,
                    "report_sha256": _sha256(diarization_report_path),
                    "diarization": diarization_path.name,
                    "diarization_sha256": _sha256(diarization_path),
                    "status": diarization_report.status.value,
                    "policy_version": diarization_report.policy_version,
                }
            (staging / "manifest.json").write_text(_json(manifest), encoding="utf-8")
            if quality_report is not None:
                (staging / "quality-report.json").write_text(
                    _json(quality_report.to_dict()),
                    encoding="utf-8",
                )
            if translation is not None:
                translation_document = translation.to_dict()
                (staging / "translation.json").write_text(
                    _json(translation_document),
                    encoding="utf-8",
                )
                (staging / "translation.txt").write_text(
                    _translation_to_text(translation_document),
                    encoding="utf-8",
                )
                manifest["translation"] = {
                    "json": "translation.json",
                    "text": "translation.txt",
                    "target_language": self.translation_target,
                    "backend": self.translation_backend.identity,
                }
                (staging / "manifest.json").write_text(
                    _json(manifest),
                    encoding="utf-8",
                )
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination

    def process(self, source: str | Path) -> CaptureOutcome:
        if self.transcription_backend is None:
            raise RuntimeError("a transcription backend is required to process captures")
        source_binding = create_source_binding(Path(source))
        path = source_binding.path
        digest = source_binding.sha256
        existing = self.store.find_by_source_snapshot(
            path.name,
            source_binding.stat.size_bytes,
            source_binding.stat.mtime_ns,
        )
        if (
            existing is not None
            and existing.capture_id == digest
            and existing.source_sha256 == digest
            and existing.state
            in {
                "review",
                "approved",
                "rejected",
                "failed",
                "processing",
            }
        ):
            return CaptureOutcome(
                existing.capture_id,
                existing.source_name,
                existing.state,
                Path(existing.package_path) if existing.package_path else None,
                replayed=True,
            )
        record, claimed = self.store.claim(
            capture_id=digest,
            source_name=path.name,
            source_sha256=digest,
            source_size=source_binding.stat.size_bytes,
            source_mtime_ns=source_binding.stat.mtime_ns,
        )
        package = Path(record.package_path) if record.package_path else None
        if not claimed:
            return CaptureOutcome(
                record.capture_id,
                record.source_name,
                record.state,
                package,
                replayed=True,
            )
        try:
            if shutil.disk_usage(self.workspace).free < self.minimum_free_bytes:
                raise OSError(
                    "capture workspace does not satisfy the configured free-space reserve"
                )
            if self.resumable:
                checkpoint_root = self.processing / record.capture_id
                duration_ms = media_duration_ms(path)
                if duration_ms > self.maximum_audio_seconds * 1000:
                    raise ValueError(
                        "capture exceeds the configured maximum audio duration"
                    )
                if self.transcription_strategy == "adaptive":
                    planner = AdaptiveChunkPlanner(
                        target_seconds=self.transcription_chunk_seconds,
                        minimum_seconds=self.transcription_minimum_chunk_seconds,
                        maximum_seconds=self.transcription_maximum_chunk_seconds,
                        overlap_seconds=self.transcription_overlap_seconds,
                        hard_boundary_silence_seconds=(
                            self.transcription_minimum_silence_seconds
                        ),
                    )
                    transcript, quality_document = AdaptiveLongFormCoordinator(
                        self.transcription_backend,
                        checkpoint_root / "transcription-adaptive",
                        planner=planner,
                        minimum_silence_seconds=(
                            self.transcription_minimum_silence_seconds
                        ),
                        language_retry_policy={"ko": "always"},
                        retry_backend=self.transcription_retry_backend,
                        silence_verification_detector=(
                            self.silence_verification_detector
                        ),
                        targeted_retry_region_detector=(
                            self.targeted_retry_region_detector
                        ),
                        audio_candidate_policies=self.audio_candidate_policies,
                        maximum_audio_candidate_channels=(
                            self.maximum_audio_candidate_channels
                        ),
                    ).run(
                        path,
                        self.transcription_options,
                        source_binding=source_binding,
                    )
                    quality_report = TranscriptQualityReport.from_dict(
                        quality_document
                    )
                else:
                    transcript = ResumableTranscriptionJob(
                        self.transcription_backend,
                        checkpoint_root / "transcription",
                        chunk_seconds=self.transcription_chunk_seconds,
                        overlap_seconds=self.transcription_overlap_seconds,
                    ).run(
                        path,
                        self.transcription_options,
                        source_digest=digest,
                        duration_ms=duration_ms,
                    )
                    quality_report = evaluate_transcript_quality(
                        transcript,
                        expected_duration_ms=duration_ms,
                    )
                if self.diarizer is not None:
                    diarization = ResumableDiarizationJob(
                        self.diarizer,
                        checkpoint_root / "diarization",
                        speaker_embedding_backend=self.diarization_embedding_backend,
                        chunk_seconds=self.diarization_chunk_seconds,
                        global_speaker_threshold=(
                            self.diarization_global_speaker_threshold
                        ),
                        global_speaker_margin=self.diarization_global_speaker_margin,
                    ).run(
                        path,
                        constraints=self.speaker_constraints,
                        source_digest=digest,
                        duration_ms=duration_ms,
                    )
                    transcript = attribute_transcript(transcript, diarization)
            else:
                transcript = AudioUnderstandingPipeline(
                    self.transcription_backend
                ).run(
                    path,
                    options=self.transcription_options,
                    diarizer=self.diarizer,
                    speaker_constraints=self.speaker_constraints,
                )
            if not self.resumable:
                quality_report = evaluate_transcript_quality(
                    transcript,
                    expected_duration_ms=transcript.duration_ms,
                )
            translation = None
            if (
                quality_report.status is TranscriptQualityStatus.PASS
                and self.translation_backend is not None
                and self.language_detector is not None
            ):
                if self.resumable:
                    translation = ResumableTranslationJob(
                        self.language_detector,
                        self.translation_backend,
                        checkpoint_root / "translation",
                        target_language=self.translation_target,
                    ).run(transcript)
                else:
                    translation = translate_transcript(
                        transcript,
                        detector=self.language_detector,
                        backend=self.translation_backend,
                        target_language=self.translation_target,
                    )
            validate_source_binding(path, source_binding)
            package = self._write_package(
                path,
                record,
                transcript,
                translation,
                quality_report,
            )
            self.store.transition(record.capture_id, "review", package_path=str(package))
            return CaptureOutcome(record.capture_id, path.name, "review", package)
        except Exception as exc:
            self.store.transition(
                record.capture_id,
                "failed",
                error=f"{type(exc).__name__}: {exc}"[:2000],
            )
            return CaptureOutcome(
                record.capture_id,
                path.name,
                "failed",
                None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def scan(
        self,
        inbox: str | Path,
        *,
        min_age_seconds: float = 30,
    ) -> tuple[CaptureOutcome, ...]:
        return tuple(
            self.process(path)
            for path in self.discover(inbox, min_age_seconds=min_age_seconds)
        )

    def _publish_approved(self) -> None:
        if self.outbox_dir is None:
            return
        from dubbing.apps.personal_capture.outbox import export_approved

        export_approved(
            self.workspace,
            self.outbox_dir,
            inbox=self.inbox_dir,
            packages_dir=self.packages_dir,
        )

    def approve(
        self,
        capture_id: str,
        *,
        speaker_aliases: Mapping[str, str] | None = None,
        notes: str = "",
        diarization_review_acknowledged: bool = False,
    ) -> Path:
        record = self.store.get(capture_id)
        if record is None or record.package_path is None:
            raise KeyError(f"capture is not reviewable: {capture_id}")
        if record.state not in {"review", "approved"}:
            raise ValueError(f"capture cannot be approved from state {record.state}")
        package = Path(record.package_path)
        existing_event = package / "giga-event.json"
        if record.state == "approved" and existing_event.is_file():
            self._publish_approved()
            return existing_event
        transcript_path = package / "transcript.json"
        transcript_document = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript = transcription_result_from_dict(transcript_document)
        manifest_path = package / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        quality_path = package / "quality-report.json"
        quality_sha256 = None
        quality_status = None
        if manifest.get("quality") is not None:
            if not quality_path.is_file():
                raise ValueError("quality-controlled package is missing quality-report.json")
            quality_report = TranscriptQualityReport.from_dict(
                json.loads(quality_path.read_text(encoding="utf-8"))
            )
            refreshed = evaluate_transcript_quality(
                transcript,
                expected_duration_ms=transcript.duration_ms,
            )
            if refreshed.to_dict() != quality_report.to_dict():
                _atomic_text(quality_path, _json(refreshed.to_dict()))
                quality_report = refreshed
            quality_manifest = manifest["quality"]
            if (
                quality_manifest.get("status") != quality_report.status.value
                or quality_manifest.get("policy_version")
                != quality_report.policy_version
            ):
                quality_manifest["status"] = quality_report.status.value
                quality_manifest["policy_version"] = quality_report.policy_version
                _atomic_text(manifest_path, _json(manifest))
            quality_status = quality_report.status.value
            if quality_report.status in {
                TranscriptQualityStatus.FAILED,
                TranscriptQualityStatus.REPROCESS_REQUIRED,
            }:
                raise ValueError(
                    f"transcript quality is {quality_status}; approval and outbox emission are blocked"
                )
            if quality_report.status is not TranscriptQualityStatus.PASS and not notes.strip():
                raise ValueError(
                    f"transcript quality is {quality_status}; explicit review notes are required"
                )
            quality_sha256 = _sha256(quality_path)
        diarization_quality_sha256 = None
        diarization_evidence_sha256 = None
        diarization_quality_status = None
        diarization_required = bool(
            manifest.get("execution", {}).get("diarization_backend")
            or manifest.get("diarization_quality") is not None
            or transcript.diarization is not None
        )
        if diarization_required:
            diarization_path = package / "diarization.json"
            diarization_quality_path = package / "diarization-quality-report.json"
            if not diarization_path.is_file() or not diarization_quality_path.is_file():
                raise ValueError(
                    "diarization-controlled package is missing its evidence or quality report"
                )
            diarization_document = json.loads(
                diarization_path.read_text(encoding="utf-8")
            )
            if transcript.diarization != diarization_document:
                raise ValueError("transcript diarization does not match diarization.json")
            diarization_evidence_sha256 = _sha256(diarization_path)
            refreshed_diarization_quality = evaluate_diarization_quality(
                transcript,
                transcript_sha256=_sha256(transcript_path),
                diarization_sha256=diarization_evidence_sha256,
            )
            stored_diarization_quality = DiarizationQualityReport.from_dict(
                json.loads(diarization_quality_path.read_text(encoding="utf-8"))
            )
            if (
                refreshed_diarization_quality.to_dict()
                != stored_diarization_quality.to_dict()
            ):
                _atomic_text(
                    diarization_quality_path,
                    _json(refreshed_diarization_quality.to_dict()),
                )
            diarization_quality = refreshed_diarization_quality
            diarization_quality_sha256 = _sha256(diarization_quality_path)
            diarization_quality_status = diarization_quality.status.value
            manifest["diarization_quality"] = {
                "report": diarization_quality_path.name,
                "report_sha256": diarization_quality_sha256,
                "diarization": diarization_path.name,
                "diarization_sha256": diarization_evidence_sha256,
                "status": diarization_quality_status,
                "policy_version": diarization_quality.policy_version,
            }
            _atomic_text(manifest_path, _json(manifest))
            if (
                diarization_quality.status
                is DiarizationQualityStatus.REPROCESS_REQUIRED
            ):
                raise ValueError(
                    "diarization quality is REPROCESS_REQUIRED; approval and outbox "
                    "emission are blocked"
                )
            if (
                diarization_quality.status
                is DiarizationQualityStatus.HUMAN_REVIEW_REQUIRED
                and not diarization_review_acknowledged
            ):
                raise ValueError(
                    "diarization quality requires explicit review or correction "
                    "acknowledgement"
                )
        _atomic_text(package / "transcript.txt", transcript_to_text(transcript))
        _atomic_text(package / "transcript.srt", transcript_to_srt(transcript))
        transcript_sha256 = _sha256(transcript_path)
        translation_path = package / "translation.json"
        translation_sha256 = None
        if translation_path.is_file():
            translation = json.loads(translation_path.read_text(encoding="utf-8"))
            translated_segments = translation.get("segments", [])
            if len(translated_segments) != len(transcript.segments):
                raise ValueError("translation does not match transcript segment count")
            for source, translated in zip(
                transcript.segments, translated_segments, strict=True
            ):
                if translated.get("source_text") != source.text:
                    raise ValueError("translation source text is stale")
                source_id, source_hash = source_segment_identity(
                    source.start_ms, source.end_ms, source.text
                )
                if translated.get("source_segment_id") not in {None, source_id}:
                    raise ValueError("translation source segment identity is stale")
                if translated.get("source_segment_sha256") not in {None, source_hash}:
                    raise ValueError("translation source segment hash is stale")
                if translated.get("status") == "source_changed_review_required":
                    raise ValueError(
                        "translation requires review after a source transcript edit"
                    )
            _atomic_text(package / "translation.txt", _translation_to_text(translation))
            translation_sha256 = _sha256(translation_path)
        stored_review = manifest.get("review", {})
        aliases = _speaker_aliases(
            speaker_aliases or stored_review.get("speaker_aliases", {})
        )
        notes = str(notes or stored_review.get("notes", ""))[:20_000]
        approval = {
            "schema_version": "dubbing.personal-capture-approval.v1",
            "capture_id": capture_id,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "speaker_aliases": aliases,
            "notes": notes,
            "diarization_review_acknowledged": (
                diarization_review_acknowledged if diarization_required else None
            ),
        }
        _atomic_text(package / "approval.json", _json(approval))
        approval_sha256 = _sha256(package / "approval.json")
        evidence_identity = hashlib.sha256(
            ":".join(
                value
                for value in (
                    transcript_sha256,
                    translation_sha256,
                    approval_sha256,
                    quality_sha256,
                    diarization_quality_sha256,
                    diarization_evidence_sha256,
                )
                if value is not None
            ).encode("ascii")
        ).hexdigest()
        event = {
            "schema_version": _GIGA_EVENT_SCHEMA,
            "event_id": f"personal-capture:{capture_id}",
            "idempotency_key": f"personal-capture:v1:{capture_id}:{evidence_identity}",
            "kind": "personal.audio.transcript",
            "trust_class": "operator-reviewed-derived-evidence",
            "source": {
                "capture_id": capture_id,
                "audio_name": record.source_name,
                "audio_sha256": record.source_sha256,
                "transcript_sha256": transcript_sha256,
                "translation_sha256": translation_sha256,
                "approval_sha256": approval_sha256,
                "quality_report_sha256": quality_sha256,
                "quality_status": quality_status,
                "diarization_quality_report_sha256": diarization_quality_sha256,
                "diarization_quality_status": diarization_quality_status,
                "diarization_sha256": diarization_evidence_sha256,
            },
            "payload": {
                "transcript_file": "transcript.json",
                "translation_file": (
                    "translation.json" if translation_path.is_file() else None
                ),
                "approval_file": "approval.json",
                "quality_report_file": (
                    "quality-report.json" if quality_path.is_file() else None
                ),
                "diarization_file": (
                    "diarization.json" if diarization_evidence_sha256 else None
                ),
                "diarization_quality_report_file": (
                    "diarization-quality-report.json"
                    if diarization_quality_sha256
                    else None
                ),
                "speaker_aliases": aliases,
                "review_notes": notes,
                "original_language_authoritative": True,
                "memory_admission": "operator-reviewed-evidence-only",
            },
        }
        _atomic_text(package / "giga-event.json", _json(event))
        manifest["state"] = "approved"
        manifest["review"] = {
            "required": False,
            "speaker_aliases": aliases,
            "notes": notes,
        }
        manifest["evidence_sha256"] = {
            "transcript.json": transcript_sha256,
            "translation.json": translation_sha256,
            "approval.json": approval_sha256,
            "quality-report.json": quality_sha256,
            "diarization.json": diarization_evidence_sha256,
            "diarization-quality-report.json": diarization_quality_sha256,
        }
        _atomic_text(manifest_path, _json(manifest))
        self.store.transition(capture_id, "approved")
        self._publish_approved()
        return package / "giga-event.json"

    def reject(self, capture_id: str, *, notes: str = "") -> CaptureRecord:
        record = self.store.get(capture_id)
        if record is None or record.package_path is None:
            raise KeyError(f"capture is not reviewable: {capture_id}")
        if record.state not in {"review", "rejected"}:
            raise ValueError(f"capture cannot be rejected from state {record.state}")
        package = Path(record.package_path)
        _atomic_text(
            package / "rejection.json",
            _json(
                {
                    "schema_version": "dubbing.personal-capture-rejection.v1",
                    "capture_id": capture_id,
                    "rejected_at": datetime.now(timezone.utc).isoformat(),
                    "notes": notes[:20_000],
                }
            ),
        )
        return self.store.transition(capture_id, "rejected")

    def update_review(
        self,
        capture_id: str,
        *,
        transcript_segments: list[dict],
        translation_segments: list[dict] | None,
        speaker_aliases: Mapping[str, str],
        notes: str,
    ) -> Path:
        record = self.store.get(capture_id)
        if record is None or record.package_path is None or record.state != "review":
            current = record.state if record else "missing"
            raise ValueError(f"capture is not editable from state {current}")
        package = Path(record.package_path)
        transcript_path = package / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        if len(transcript_segments) != len(transcript.get("segments", [])):
            raise ValueError("transcript segment count cannot change in review")
        source_changed: list[bool] = []
        for existing, edit in zip(transcript["segments"], transcript_segments, strict=True):
            text = str(edit.get("text", "")).strip()
            if not text:
                raise ValueError("transcript segments cannot be blank")
            if len(text) > 20_000:
                raise ValueError("transcript segment is too long")
            changed = text != existing["text"]
            source_changed.append(changed)
            existing["text"] = text
            if changed:
                existing["words"] = []
                existing["confidence"] = None
        transcript["text"] = " ".join(item["text"] for item in transcript["segments"]).strip()
        transcript["review"] = {
            "manual_text_edits": any(source_changed),
            "word_timestamps_removed_from_edited_segments": any(source_changed),
        }
        validated_transcript = transcription_result_from_dict(transcript)
        _atomic_text(transcript_path, _json(transcript))
        _atomic_text(package / "transcript.txt", transcript_to_text(validated_transcript))
        _atomic_text(package / "transcript.srt", transcript_to_srt(validated_transcript))
        translation_path = package / "translation.json"
        if translation_path.is_file():
            translation = json.loads(translation_path.read_text(encoding="utf-8"))
            edits = translation_segments
            if edits is None:
                edits = [
                    {"target_text": item.get("target_text")}
                    for item in translation.get("segments", [])
                ]
            if len(edits) != len(translation.get("segments", [])):
                raise ValueError("translation segment count cannot change in review")
            for index, (existing, edit) in enumerate(
                zip(translation["segments"], edits, strict=True)
            ):
                target = str(edit.get("target_text", "")).strip()
                if len(target) > 20_000:
                    raise ValueError("translation segment is too long")
                old_target = existing.get("target_text") or ""
                source_segment = transcript["segments"][index]
                existing["source_text"] = source_segment["text"]
                source_id, source_hash = source_segment_identity(
                    source_segment["start_ms"],
                    source_segment["end_ms"],
                    source_segment["text"],
                )
                existing["source_segment_id"] = source_id
                existing["source_segment_sha256"] = source_hash
                existing["target_text"] = target
                if source_changed[index] and target == old_target:
                    existing["status"] = "source_changed_review_required"
                else:
                    existing["status"] = "reviewed"
            _atomic_text(translation_path, _json(translation))
            _atomic_text(package / "translation.txt", _translation_to_text(translation))
        manifest_path = package / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["review"]["speaker_aliases"] = _speaker_aliases(speaker_aliases)
        manifest["review"]["notes"] = str(notes)[:20_000]
        manifest["evidence_sha256"] = {
            "transcript.json": _sha256(transcript_path),
            "translation.json": (
                _sha256(translation_path) if translation_path.is_file() else None
            ),
        }
        _atomic_text(manifest_path, _json(manifest))
        return manifest_path

    def records(self) -> Iterable[CaptureRecord]:
        with self.store._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM captures ORDER BY created_at, capture_id"
            ).fetchall()
        return tuple(record for row in rows if (record := self.store._record(row)) is not None)
