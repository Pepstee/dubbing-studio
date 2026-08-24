from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from dubbing.transcription.models import TranscriptionResult


DIARIZATION_QUALITY_POLICY_VERSION = "dubbing.diarization-quality-policy.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DiarizationQualityStatus(str, Enum):
    PASS = "PASS"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REPROCESS_REQUIRED = "REPROCESS_REQUIRED"


@dataclass(frozen=True)
class DiarizationQualityReport:
    status: DiarizationQualityStatus
    issues: tuple[dict, ...]
    metrics: dict
    transcript_sha256: str
    diarization_sha256: str
    policy_version: str = DIARIZATION_QUALITY_POLICY_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": "dubbing.diarization-quality-report.v1",
            "policy_version": self.policy_version,
            "status": self.status.value,
            "approval_allowed": self.status is DiarizationQualityStatus.PASS,
            "human_review_acknowledgement_required": (
                self.status is DiarizationQualityStatus.HUMAN_REVIEW_REQUIRED
            ),
            "transcript_sha256": self.transcript_sha256,
            "diarization_sha256": self.diarization_sha256,
            "issues": list(self.issues),
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, document: dict) -> DiarizationQualityReport:
        if document.get("schema_version") != "dubbing.diarization-quality-report.v1":
            raise ValueError("unsupported diarization quality report schema")
        transcript_sha256 = document.get("transcript_sha256")
        diarization_sha256 = document.get("diarization_sha256")
        if not isinstance(transcript_sha256, str) or not _SHA256.fullmatch(
            transcript_sha256
        ):
            raise ValueError("diarization quality transcript hash is malformed")
        if not isinstance(diarization_sha256, str) or not _SHA256.fullmatch(
            diarization_sha256
        ):
            raise ValueError("diarization quality evidence hash is malformed")
        return cls(
            status=DiarizationQualityStatus(document["status"]),
            policy_version=document.get(
                "policy_version", DIARIZATION_QUALITY_POLICY_VERSION
            ),
            transcript_sha256=transcript_sha256,
            diarization_sha256=diarization_sha256,
            issues=tuple(document.get("issues", [])),
            metrics=dict(document.get("metrics", {})),
        )


def evaluate_diarization_quality(
    transcript: TranscriptionResult,
    *,
    transcript_sha256: str,
    diarization_sha256: str,
    pass_coverage_ratio: float = 0.95,
    minimum_coverage_ratio: float = 0.90,
) -> DiarizationQualityReport:
    if not 0 < minimum_coverage_ratio <= pass_coverage_ratio <= 1:
        raise ValueError("diarization coverage thresholds are invalid")
    for label, value in (
        ("transcript_sha256", transcript_sha256),
        ("diarization_sha256", diarization_sha256),
    ):
        if not _SHA256.fullmatch(value):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    issues: list[dict] = []
    diarization = transcript.diarization
    turns = diarization.get("turns", []) if isinstance(diarization, dict) else []
    duration_ms = transcript.duration_ms or 0
    malformed_turn_count = 0
    previous_key = None
    for turn in turns if isinstance(turns, list) else ():
        if not isinstance(turn, dict):
            malformed_turn_count += 1
            continue
        start_ms = turn.get("start_ms")
        end_ms = turn.get("end_ms")
        speaker = turn.get("speaker")
        key = (start_ms, end_ms, speaker)
        valid = not (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
            or end_ms > duration_ms
            or not isinstance(speaker, str)
            or not speaker
        )
        if valid and previous_key is not None and key < previous_key:
            valid = False
        if not valid:
            malformed_turn_count += 1
        else:
            previous_key = key

    eligible_segments = []
    attributed_segments = []
    overlap_count = 0
    boundary_count = 0
    no_speech_count = 0
    unresolved_count = 0
    eligible_word_count = 0
    attributed_word_count = 0
    for segment in transcript.segments:
        if segment.speaker_status == "no_speech":
            no_speech_count += 1
            continue
        if segment.speaker_status == "overlap":
            overlap_count += 1
            continue
        if segment.speaker_status == "speaker_boundary":
            boundary_count += 1
            continue
        eligible_segments.append(segment)
        if (
            segment.speaker_status in {"attributed", "word_attributed"}
            and segment.speaker is not None
        ):
            attributed_segments.append(segment)
        else:
            unresolved_count += 1
        eligible_word_count += len(segment.words)
        attributed_word_count += sum(word.speaker is not None for word in segment.words)

    eligible_duration_ms = sum(
        segment.end_ms - segment.start_ms for segment in eligible_segments
    )
    attributed_duration_ms = sum(
        segment.end_ms - segment.start_ms for segment in attributed_segments
    )
    duration_ratio = (
        attributed_duration_ms / eligible_duration_ms if eligible_duration_ms else 1.0
    )
    word_ratio = (
        attributed_word_count / eligible_word_count if eligible_word_count else None
    )

    reconciliation = {}
    if isinstance(diarization, dict):
        provenance = diarization.get("provenance", {})
        if isinstance(provenance, dict):
            candidate = provenance.get("global_speaker_reconciliation", {})
            if isinstance(candidate, dict):
                reconciliation = candidate
    missing_embedding_count = int(reconciliation.get("unresolved_count", 0) or 0)
    ambiguous_pair_count = len(reconciliation.get("ambiguous_pairs", []) or [])

    if not isinstance(diarization, dict) or not isinstance(turns, list) or not turns:
        issues.append(
            {
                "code": "missing_diarization_evidence",
                "severity": "critical",
                "message": "Configured diarization produced no usable speaker turns.",
            }
        )
    if malformed_turn_count:
        issues.append(
            {
                "code": "malformed_diarization_turns",
                "severity": "critical",
                "message": "Speaker turns are malformed, unordered, or out of bounds.",
                "count": malformed_turn_count,
            }
        )
    if unresolved_count:
        issues.append(
            {
                "code": "unresolved_speech_segments",
                "severity": "critical",
                "message": "Speech segments remain without a single-speaker outcome.",
                "count": unresolved_count,
            }
        )
    if duration_ratio < minimum_coverage_ratio or (
        word_ratio is not None and word_ratio < minimum_coverage_ratio
    ):
        issues.append(
            {
                "code": "diarization_coverage_below_minimum",
                "severity": "critical",
                "message": "Speaker attribution coverage is below the fail-closed minimum.",
            }
        )
    elif duration_ratio < pass_coverage_ratio or (
        word_ratio is not None and word_ratio < pass_coverage_ratio
    ):
        issues.append(
            {
                "code": "diarization_coverage_requires_review",
                "severity": "review",
                "message": "Speaker attribution coverage requires operator review.",
            }
        )
    for code, count, message in (
        ("overlapping_speech", overlap_count, "Overlapping speech requires review."),
        ("speaker_boundaries", boundary_count, "Speaker boundaries require review."),
        (
            "missing_speaker_embeddings",
            missing_embedding_count,
            "Some diarized speakers lack reconciliation embeddings.",
        ),
        (
            "ambiguous_speaker_pairs",
            ambiguous_pair_count,
            "Global speaker reconciliation contains ambiguous pairs.",
        ),
    ):
        if count:
            issues.append(
                {"code": code, "severity": "review", "message": message, "count": count}
            )

    severities = {item["severity"] for item in issues}
    if "critical" in severities:
        status = DiarizationQualityStatus.REPROCESS_REQUIRED
    elif "review" in severities:
        status = DiarizationQualityStatus.HUMAN_REVIEW_REQUIRED
    else:
        status = DiarizationQualityStatus.PASS
    return DiarizationQualityReport(
        status=status,
        issues=tuple(issues),
        transcript_sha256=transcript_sha256,
        diarization_sha256=diarization_sha256,
        metrics={
            "segment_count": len(transcript.segments),
            "eligible_speech_segment_count": len(eligible_segments),
            "single_speaker_attributed_segment_count": len(attributed_segments),
            "unresolved_speech_segment_count": unresolved_count,
            "valid_overlap_segment_count": overlap_count,
            "speaker_boundary_segment_count": boundary_count,
            "excluded_no_speech_segment_count": no_speech_count,
            "eligible_speech_duration_ms": eligible_duration_ms,
            "attributed_speech_duration_ms": attributed_duration_ms,
            "attributed_speech_duration_ratio": round(duration_ratio, 6),
            "eligible_word_count": eligible_word_count,
            "attributed_word_count": attributed_word_count,
            "attributed_word_ratio": (
                round(word_ratio, 6) if word_ratio is not None else None
            ),
            "word_coverage_available": word_ratio is not None,
            "malformed_turn_count": malformed_turn_count,
            "missing_embedding_count": missing_embedding_count,
            "ambiguous_pair_count": ambiguous_pair_count,
            "pass_coverage_ratio": pass_coverage_ratio,
            "minimum_coverage_ratio": minimum_coverage_ratio,
        },
    )
