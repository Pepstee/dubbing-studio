from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dubbing.evaluation.metrics import character_tokens, levenshtein_distance, word_tokens
from dubbing.media import ffmpeg_executable
from dubbing.transcription.cloud import (
    CloudTransport,
    ElevenLabsHTTPTransport,
    PROVIDERS,
)
from dubbing.transcription.job import source_sha256
from dubbing.transcription.models import (
    DecodeDiagnostics,
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)
from dubbing.transcription.quality import evaluate_transcript_quality


_POLICY_SCHEMA = "dubbing.cloud-teacher-policy.v1"
_REPORT_SCHEMA = "dubbing.cloud-teacher-report.v1"
_CORPUS_SCHEMA = "dubbing.asr-silver-corpus.v1"
_ALLOWED_LANGUAGES = {"en", "ru", "ro", "ko"}


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("programme timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class CloudTeacherPolicy:
    programme_id: str
    provider: str
    begins_at: datetime
    ends_at: datetime
    operator_authorization_id: str
    cloud_allowed: bool
    training_corpus_allowed: bool
    model_improvement_opt_out_attested: bool
    retention_mode: str
    max_total_audio_seconds: int
    max_estimated_cost_usd: float
    estimated_price_per_hour_usd: float
    chunk_seconds: int
    maximum_agreement_wer: float
    maximum_agreement_cer: float

    @classmethod
    def from_dict(cls, document: dict) -> "CloudTeacherPolicy":
        if document.get("schema_version") != _POLICY_SCHEMA:
            raise ValueError(f"cloud teacher policy must use schema {_POLICY_SCHEMA}")
        privacy = document.get("privacy")
        limits = document.get("limits")
        training = document.get("training")
        if not all(isinstance(item, dict) for item in (privacy, limits, training)):
            raise ValueError("policy privacy, limits and training must be objects")
        policy = cls(
            programme_id=str(document.get("programme_id", "")).strip(),
            provider=str(document.get("provider", "")).strip(),
            begins_at=_utc(str(document.get("begins_at", ""))),
            ends_at=_utc(str(document.get("ends_at", ""))),
            operator_authorization_id=str(
                document.get("operator_authorization_id", "")
            ).strip(),
            cloud_allowed=document.get("cloud_allowed") is True,
            training_corpus_allowed=document.get("training_corpus_allowed") is True,
            model_improvement_opt_out_attested=(
                privacy.get("model_improvement_opt_out_attested") is True
            ),
            retention_mode=str(privacy.get("retention_mode", "")).strip(),
            max_total_audio_seconds=int(limits.get("max_total_audio_seconds", 0)),
            max_estimated_cost_usd=float(limits.get("max_estimated_cost_usd", 0)),
            estimated_price_per_hour_usd=float(
                limits.get("estimated_price_per_hour_usd", 0)
            ),
            chunk_seconds=int(limits.get("chunk_seconds", 0)),
            maximum_agreement_wer=float(training.get("maximum_agreement_wer", -1)),
            maximum_agreement_cer=float(training.get("maximum_agreement_cer", -1)),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.programme_id or not self.operator_authorization_id:
            raise ValueError("programme and operator authorization IDs are required")
        if self.provider != "elevenlabs":
            raise ValueError("month-one primary teacher must be elevenlabs")
        if self.ends_at <= self.begins_at:
            raise ValueError("programme end must follow programme start")
        if self.ends_at - self.begins_at > timedelta(days=31):
            raise ValueError("cloud teacher programme cannot exceed 31 days")
        if not self.cloud_allowed:
            raise ValueError("cloud_allowed must be true before any programme upload")
        if not self.training_corpus_allowed:
            raise ValueError("training_corpus_allowed must be explicitly true")
        if not self.model_improvement_opt_out_attested:
            raise ValueError("provider model-improvement opt-out must be attested")
        if not self.retention_mode:
            raise ValueError("provider retention mode must be recorded")
        if self.max_total_audio_seconds <= 0 or self.max_estimated_cost_usd <= 0:
            raise ValueError("positive programme audio and cost caps are required")
        if self.estimated_price_per_hour_usd <= 0:
            raise ValueError("positive estimated provider price is required")
        if not 60 <= self.chunk_seconds <= 7_200:
            raise ValueError("cloud chunks must be between 60 seconds and 2 hours")
        if not 0 <= self.maximum_agreement_wer <= 0.10:
            raise ValueError("silver-label WER threshold must be in [0, 0.10]")
        if not 0 <= self.maximum_agreement_cer <= 0.15:
            raise ValueError("silver-label CER threshold must be in [0, 0.15]")

    def assert_active(self, now: datetime | None = None) -> None:
        instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if not self.begins_at <= instant < self.ends_at:
            raise TranscriptionError("cloud teacher programme is outside its authorized window")

    @property
    def configuration_sha256(self) -> str:
        payload = {
            "programme_id": self.programme_id,
            "provider": self.provider,
            "begins_at": self.begins_at.isoformat(),
            "ends_at": self.ends_at.isoformat(),
            "retention_mode": self.retention_mode,
            "max_total_audio_seconds": self.max_total_audio_seconds,
            "max_estimated_cost_usd": self.max_estimated_cost_usd,
            "estimated_price_per_hour_usd": self.estimated_price_per_hour_usd,
            "chunk_seconds": self.chunk_seconds,
            "maximum_agreement_wer": self.maximum_agreement_wer,
            "maximum_agreement_cer": self.maximum_agreement_cer,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_cloud_teacher_policy(path: str | Path) -> CloudTeacherPolicy:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load cloud teacher policy: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("cloud teacher policy must be a JSON object")
    return CloudTeacherPolicy.from_dict(document)


def _extract_flac(source: Path, start_ms: int, end_ms: int, destination: Path) -> None:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise TranscriptionError("ffmpeg is required for cloud teacher extraction")
    process = subprocess.run(
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
            "flac",
            "-y",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=max(180, (end_ms - start_ms) // 1000 + 120),
    )
    if process.returncode:
        raise TranscriptionError(f"cloud FLAC extraction failed: {process.stderr.strip()}")


def _error_rate(reference: list[str], candidate: list[str]) -> float:
    if not reference:
        return 0.0 if not candidate else 1.0
    return levenshtein_distance(reference, candidate) / len(reference)


def _overlapping_text(result: TranscriptionResult, start_ms: int, end_ms: int) -> str:
    return " ".join(
        segment.text
        for segment in result.segments
        if start_ms <= (segment.start_ms + segment.end_ms) // 2 <= end_ms
    ).strip()


def _split_for_recording(recording_sha256: str) -> str:
    bucket = int(hashlib.sha256(recording_sha256.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "locked_test"


class CloudTeacherRunner:
    """Run a time-limited full-recording cloud shadow and build strict silver labels."""

    def __init__(
        self,
        policy: CloudTeacherPolicy,
        output_dir: str | Path,
        programme_state: str | Path,
        *,
        transport: CloudTransport | None = None,
        now: datetime | None = None,
    ) -> None:
        self.policy = policy
        self.output_dir = Path(output_dir).resolve()
        self.programme_state = Path(programme_state).resolve()
        self.transport = transport or ElevenLabsHTTPTransport()
        self.now = now

    def _load_usage(self) -> dict:
        if not self.programme_state.exists():
            return {
                "schema_version": "dubbing.cloud-teacher-usage.v1",
                "programme_id": self.policy.programme_id,
                "policy_sha256": self.policy.configuration_sha256,
                "chunks": {},
            }
        document = json.loads(self.programme_state.read_text(encoding="utf-8"))
        if document.get("programme_id") != self.policy.programme_id:
            raise TranscriptionError("programme state belongs to another programme")
        if document.get("policy_sha256") != self.policy.configuration_sha256:
            raise TranscriptionError("programme policy changed; checkpoint reuse denied")
        if not isinstance(document.get("chunks"), dict):
            raise TranscriptionError("programme state is malformed")
        return document

    def _assert_budget(self, usage: dict, duration_ms: int) -> None:
        consumed_seconds = sum(item["duration_ms"] for item in usage["chunks"].values()) / 1000
        consumed_cost = sum(item["estimated_cost_usd"] for item in usage["chunks"].values())
        requested_seconds = duration_ms / 1000
        requested_cost = requested_seconds / 3600 * self.policy.estimated_price_per_hour_usd
        if consumed_seconds + requested_seconds > self.policy.max_total_audio_seconds:
            raise TranscriptionError("cloud teacher audio cap would be exceeded")
        if consumed_cost + requested_cost > self.policy.max_estimated_cost_usd:
            raise TranscriptionError("cloud teacher cost cap would be exceeded")

    def _chunk_bounds(self, duration_ms: int) -> tuple[tuple[int, int], ...]:
        chunk_ms = self.policy.chunk_seconds * 1000
        return tuple(
            (start, min(duration_ms, start + chunk_ms))
            for start in range(0, duration_ms, chunk_ms)
        )

    def _load_checkpoint(
        self, path: Path, source_digest: str, start_ms: int, end_ms: int
    ) -> dict | None:
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "recording_sha256": source_digest,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "provider": self.policy.provider,
            "policy_sha256": self.policy.configuration_sha256,
        }
        if any(document.get(key) != value for key, value in expected.items()):
            raise TranscriptionError("cloud teacher checkpoint binding mismatch")
        if not isinstance(document.get("candidate"), dict):
            raise TranscriptionError("cloud teacher checkpoint has no candidate")
        return document

    def _candidate_segments(
        self, candidate: dict, offset_ms: int, chunk_duration_ms: int
    ) -> tuple[TranscriptSegment, ...]:
        segments = []
        for item in candidate.get("segments", []):
            try:
                start_ms = offset_ms + int(item["start_ms"])
                end_ms = offset_ms + int(item["end_ms"])
                text = str(item["text"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise TranscriptionError("cloud teacher candidate is malformed") from exc
            if (
                start_ms < offset_ms
                or end_ms > offset_ms + chunk_duration_ms
                or end_ms <= start_ms
                or not text
            ):
                raise TranscriptionError("cloud teacher candidate contains invalid speech")
            segments.append(
                TranscriptSegment(
                    start_ms,
                    end_ms,
                    text,
                    confidence=item.get("confidence"),
                    speaker=item.get("speaker"),
                    speaker_status=("attributed" if item.get("speaker") else "not_requested"),
                    language=item.get("language"),
                    uncertain=False,
                    diagnostics=DecodeDiagnostics(
                        backend_metadata={
                            "cloud_teacher": True,
                            "provider": self.policy.provider,
                            "model": PROVIDERS[self.policy.provider].model,
                        }
                    ),
                )
            )
        return tuple(segments)

    def _build_silver_corpus(
        self,
        source: Path,
        local: TranscriptionResult,
        cloud: TranscriptionResult,
        local_quality: dict,
        cloud_quality: dict,
    ) -> dict:
        corpus = self.output_dir / "silver-corpus"
        audio_dir = corpus / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        split = _split_for_recording(local.source_sha256 or "")
        accepted = []
        excluded = []
        for index, segment in enumerate(local.segments):
            cloud_text = _overlapping_text(cloud, segment.start_ms, segment.end_ms)
            local_words = word_tokens(segment.text)
            cloud_words = word_tokens(cloud_text)
            local_chars = character_tokens(segment.text)
            cloud_chars = character_tokens(cloud_text)
            wer = _error_rate(local_words, cloud_words)
            cer = _error_rate(local_chars, cloud_chars)
            duration_ms = segment.end_ms - segment.start_ms
            reasons = []
            if local_quality.get("status") in {"FAILED", "REPROCESS_REQUIRED"}:
                reasons.append("LOCAL_QUALITY_FAIL_CLOSED")
            if cloud_quality.get("status") in {"FAILED", "REPROCESS_REQUIRED"}:
                reasons.append("CLOUD_QUALITY_FAIL_CLOSED")
            if segment.uncertain or segment.text.startswith("[UNCERTAIN:"):
                reasons.append("LOCAL_UNCERTAIN")
            if not 400 <= duration_ms <= 30_000:
                reasons.append("DURATION_OUTSIDE_TRAINING_BOUND")
            if len(local_words) < 3 or len(cloud_words) < 3:
                reasons.append("TOO_FEW_TOKENS")
            if segment.speaker_status == "overlap" or len(segment.speakers) > 1:
                reasons.append("OVERLAPPING_SPEAKERS")
            if wer > self.policy.maximum_agreement_wer:
                reasons.append("LOCAL_CLOUD_WER_DISAGREEMENT")
            if cer > self.policy.maximum_agreement_cer:
                reasons.append("LOCAL_CLOUD_CER_DISAGREEMENT")
            cloud_overlaps = [
                item
                for item in cloud.segments
                if item.start_ms < segment.end_ms and item.end_ms > segment.start_ms
            ]
            known_languages = {item.language for item in cloud_overlaps if item.language}
            if known_languages - _ALLOWED_LANGUAGES:
                reasons.append("LANGUAGE_OUTSIDE_PROGRAMME_SCOPE")
            confidences = [
                item.confidence for item in cloud_overlaps if item.confidence is not None
            ]
            if confidences and sum(confidences) / len(confidences) < 0.55:
                reasons.append("LOW_CLOUD_ACOUSTIC_CONFIDENCE")
            row_id = hashlib.sha256(
                f"{local.source_sha256}:{segment.start_ms}:{segment.end_ms}".encode()
            ).hexdigest()[:24]
            evidence = {
                "row_id": row_id,
                "recording_sha256": local.source_sha256,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "speaker": segment.speaker or "UNKNOWN",
                "language": segment.language or (
                    next(iter(known_languages)) if len(known_languages) == 1 else None
                ),
                "local_text": segment.text,
                "cloud_text": cloud_text,
                "target_text": None,
                "agreement_wer": wer,
                "agreement_cer": cer,
                "split": split,
                "label_class": "CONSENSUS_SILVER" if not reasons else "EXCLUDED",
                "exclusion_reasons": reasons,
                "cloud_provider": self.policy.provider,
                "cloud_model": cloud.model,
                "policy_sha256": self.policy.configuration_sha256,
            }
            if reasons:
                excluded.append(evidence)
                continue
            evidence["target_text"] = cloud_text
            clip = audio_dir / f"{row_id}.flac"
            if not clip.is_file():
                _extract_flac(source, segment.start_ms, segment.end_ms, clip)
            evidence["audio"] = str(clip.relative_to(corpus))
            evidence["audio_sha256"] = source_sha256(clip)
            accepted.append(evidence)

        cloud_only = []
        for cloud_segment in cloud.segments:
            midpoint = (cloud_segment.start_ms + cloud_segment.end_ms) // 2
            if any(
                local_segment.start_ms <= midpoint <= local_segment.end_ms
                for local_segment in local.segments
            ):
                continue
            if (
                cloud_only
                and cloud_segment.start_ms - cloud_only[-1][-1].end_ms <= 1_000
                and cloud_segment.speaker == cloud_only[-1][-1].speaker
            ):
                cloud_only[-1].append(cloud_segment)
            else:
                cloud_only.append([cloud_segment])
        for group in cloud_only:
            start_ms = group[0].start_ms
            end_ms = group[-1].end_ms
            text = " ".join(item.text for item in group)
            row_id = hashlib.sha256(
                f"{local.source_sha256}:cloud-only:{start_ms}:{end_ms}".encode()
            ).hexdigest()[:24]
            excluded.append(
                {
                    "row_id": row_id,
                    "recording_sha256": local.source_sha256,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker": group[0].speaker or "UNKNOWN",
                    "language": group[0].language,
                    "local_text": "",
                    "cloud_text": text,
                    "target_text": None,
                    "agreement_wer": 1.0,
                    "agreement_cer": 1.0,
                    "split": split,
                    "label_class": "CLOUD_ONLY_UNVERIFIED",
                    "exclusion_reasons": ["NO_LOCAL_CORROBORATION"],
                    "cloud_provider": self.policy.provider,
                    "cloud_model": cloud.model,
                    "policy_sha256": self.policy.configuration_sha256,
                }
            )

        for name, rows in (("accepted.jsonl", accepted), ("excluded.jsonl", excluded)):
            payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            )
            path = corpus / name
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
        manifest = {
            "schema_version": _CORPUS_SCHEMA,
            "recording_sha256": local.source_sha256,
            "split": split,
            "split_unit": "recording",
            "accepted_count": len(accepted),
            "excluded_count": len(excluded),
            "label_policy": "local-cloud agreement; cloud-only labels forbidden",
            "local_quality": local_quality,
            "cloud_quality": cloud_quality,
            "human_ground_truth": False,
            "giga_admission_allowed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(corpus / "manifest.json", manifest)
        return manifest

    def run(self, source: str | Path, local: TranscriptionResult) -> dict:
        self.policy.assert_active(self.now)
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise TranscriptionError(f"media input not found: {source_path}")
        digest = source_sha256(source_path)
        if local.source_sha256 != digest:
            raise TranscriptionError("local transcript does not match cloud teacher source")
        if not local.duration_ms or local.duration_ms <= 0:
            raise TranscriptionError("local transcript must preserve source duration")
        provider = PROVIDERS[self.policy.provider]
        credential = os.environ.get(provider.credential_environment_variable)
        if not credential:
            raise TranscriptionError(
                f"{provider.credential_environment_variable} is not configured locally; no upload occurred"
            )
        usage = self._load_usage()
        unseen_ms = sum(
            end - start
            for start, end in self._chunk_bounds(local.duration_ms)
            if f"{digest}:{start}:{end}:{provider.name}" not in usage["chunks"]
        )
        self._assert_budget(usage, unseen_ms)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = self.output_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        all_segments = []
        chunk_reports = []
        with tempfile.TemporaryDirectory(prefix="dubbing-cloud-teacher-") as directory:
            temporary_dir = Path(directory)
            for index, (start_ms, end_ms) in enumerate(
                self._chunk_bounds(local.duration_ms)
            ):
                checkpoint_path = checkpoints / f"chunk-{index:06d}.json"
                checkpoint = self._load_checkpoint(
                    checkpoint_path, digest, start_ms, end_ms
                )
                reused = checkpoint is not None
                if checkpoint is None:
                    audio = temporary_dir / f"chunk-{index:06d}.flac"
                    _extract_flac(source_path, start_ms, end_ms, audio)
                    audio_bytes = audio.read_bytes()
                    request = provider.build_request()
                    response = self.transport(request, audio_bytes, credential)
                    candidate = response.get("candidate")
                    if not isinstance(candidate, dict):
                        raise TranscriptionError("cloud teacher returned no structured candidate")
                    request_receipt = response.get("request_receipt")
                    if not isinstance(request_receipt, str) or not request_receipt.strip():
                        raise TranscriptionError("cloud teacher returned no request receipt")
                    estimated_cost = (
                        (end_ms - start_ms)
                        / 3_600_000
                        * self.policy.estimated_price_per_hour_usd
                    )
                    checkpoint = {
                        "schema_version": "dubbing.cloud-teacher-checkpoint.v1",
                        "recording_sha256": digest,
                        "source_span_sha256": source_sha256(audio),
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "provider": provider.name,
                        "model": provider.model,
                        "policy_sha256": self.policy.configuration_sha256,
                        "operator_authorization_id": self.policy.operator_authorization_id,
                        "retention_mode": self.policy.retention_mode,
                        "request_configuration": request["options"],
                        "request_receipt": request_receipt,
                        "estimated_cost_usd": estimated_cost,
                        "candidate": candidate,
                        "candidate_provenance": response.get("provenance"),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    _atomic_json(checkpoint_path, checkpoint)
                    usage_key = f"{digest}:{start_ms}:{end_ms}:{provider.name}"
                    usage["chunks"][usage_key] = {
                        "duration_ms": end_ms - start_ms,
                        "estimated_cost_usd": estimated_cost,
                        "checkpoint": str(checkpoint_path),
                    }
                    _atomic_json(self.programme_state, usage)
                candidate_segments = self._candidate_segments(
                    checkpoint["candidate"], start_ms, end_ms - start_ms
                )
                all_segments.extend(candidate_segments)
                chunk_reports.append(
                    {
                        "index": index,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "checkpoint": str(checkpoint_path),
                        "reused": reused,
                        "segment_count": len(candidate_segments),
                        "estimated_cost_usd": checkpoint["estimated_cost_usd"],
                    }
                )

        ordered = tuple(sorted(all_segments, key=lambda item: (item.start_ms, item.end_ms)))
        cloud = TranscriptionResult(
            segments=ordered,
            text=" ".join(item.text for item in ordered),
            backend="cloud-teacher",
            model=provider.model,
            device="cloud",
            language=None,
            duration_ms=local.duration_ms,
            confidence_available=any(item.confidence is not None for item in ordered),
            source_sha256=digest,
            diarization={"provider": provider.name, "identity_scope": "anonymous-per-chunk"},
            provenance={
                "programme_id": self.policy.programme_id,
                "policy_sha256": self.policy.configuration_sha256,
                "provider": provider.name,
                "model": provider.model,
                "operator_authorization_id": self.policy.operator_authorization_id,
            },
        )
        quality = evaluate_transcript_quality(
            cloud, expected_duration_ms=local.duration_ms
        ).to_dict()
        local_quality = evaluate_transcript_quality(
            local, expected_duration_ms=local.duration_ms
        ).to_dict()
        _atomic_json(self.output_dir / "cloud-transcript.json", cloud.to_dict())
        corpus = self._build_silver_corpus(
            source_path, local, cloud, local_quality, quality
        )
        local_text = word_tokens(local.text)
        cloud_text = word_tokens(cloud.text)
        report = {
            "schema_version": _REPORT_SCHEMA,
            "programme_id": self.policy.programme_id,
            "recording_sha256": digest,
            "provider": provider.name,
            "model": provider.model,
            "policy_sha256": self.policy.configuration_sha256,
            "chunks": chunk_reports,
            "local_cloud_wer": _error_rate(local_text, cloud_text),
            "local_quality": local_quality,
            "cloud_quality": quality,
            "silver_corpus": corpus,
            "cloud_is_human_ground_truth": False,
            "local_transcript_overwritten": False,
            "giga_admission_allowed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(self.output_dir / "report.json", report)
        return report
