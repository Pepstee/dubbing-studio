from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dubbing.evaluation.metrics import token_agreement, word_tokens
from dubbing.transcription.models import TranscriptionResult, transcription_result_from_dict
from dubbing.transcription.adaptive import detect_silence_intervals
from dubbing.transcription.quality import (
    TranscriptQualityReport,
    TranscriptQualityStatus,
    evaluate_transcript_quality,
)


_HEALTHY_QUALITY_STATES = {
    TranscriptQualityStatus.PASS,
    TranscriptQualityStatus.PASS_WITH_UNCERTAIN_SPANS,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_result(path: Path) -> TranscriptionResult:
    return transcription_result_from_dict(json.loads(path.read_text(encoding="utf-8")))


def _quality_evidence(
    result_path: Path,
    result: TranscriptionResult,
    duration_ms: int,
    known_silence_intervals: tuple[tuple[int, int], ...] | None = None,
) -> tuple[dict, dict]:
    if known_silence_intervals is not None:
        report = evaluate_transcript_quality(
            result,
            expected_duration_ms=duration_ms,
            known_silence_intervals=known_silence_intervals,
        )
        return report.to_dict(), {
            "path": None,
            "sha256": None,
            "recomputed_from_source_vad": True,
        }
    for name in ("reviewed-quality-report.json", "quality-report.json"):
        path = result_path.parent / name
        if not path.is_file():
            continue
        report = TranscriptQualityReport.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        metrics = report.metrics
        if metrics.get("segment_count") != len(result.segments):
            continue
        if metrics.get("duration_ms") not in {None, duration_ms}:
            continue
        return report.to_dict(), {"path": str(path), "sha256": _sha256(path)}
    report = evaluate_transcript_quality(result, expected_duration_ms=duration_ms)
    return report.to_dict(), {"path": None, "sha256": None, "recomputed": True}


def _window_text(result: TranscriptionResult, start_ms: int, end_ms: int) -> str:
    return " ".join(
        segment.text
        for segment in result.segments
        if start_ms
        <= segment.start_ms + (segment.end_ms - segment.start_ms) // 2
        < end_ms
    )


@dataclass(frozen=True)
class Candidate:
    label: str
    path: Path
    result: TranscriptionResult
    quality: dict
    quality_evidence: dict


def adjudicate_transcripts(
    primary_path: str | Path,
    comparator_paths: Sequence[str | Path],
    *,
    output_path: str | Path | None = None,
    source_path: str | Path | None = None,
    window_ms: int = 60_000,
    low_agreement_threshold: float = 0.55,
    consensus_threshold: float = 0.75,
) -> dict:
    """Create fail-closed local consensus evidence without pretending consensus is truth.

    This function never emits an outbox or GIGA event. It can make a transcript eligible
    for the repository's separate explicit approval gate, but cannot approve it.
    """
    if window_ms < 10_000:
        raise ValueError("window_ms must be at least 10000")
    if not 0 <= low_agreement_threshold <= consensus_threshold <= 1:
        raise ValueError("agreement thresholds must satisfy 0 <= low <= consensus <= 1")
    if not comparator_paths:
        raise ValueError("at least one comparator is required")

    primary_file = Path(primary_path).resolve()
    primary = _load_result(primary_file)
    if not primary.source_sha256:
        raise ValueError("primary transcript is not source-bound")
    duration_ms = primary.duration_ms or max(
        (segment.end_ms for segment in primary.segments), default=0
    )
    silence_intervals = None
    source_evidence = None
    if source_path is not None:
        source = Path(source_path).resolve()
        source_digest = _sha256(source)
        if source_digest != primary.source_sha256:
            raise ValueError("source SHA-256 does not match the primary transcript")
        silence_intervals = detect_silence_intervals(source)
        source_evidence = {"path": str(source), "sha256": source_digest}
    primary_quality, primary_quality_evidence = _quality_evidence(
        primary_file, primary, duration_ms, silence_intervals
    )

    comparators: list[Candidate] = []
    for raw_path in comparator_paths:
        path = Path(raw_path).resolve()
        result = _load_result(path)
        if result.source_sha256 != primary.source_sha256:
            raise ValueError(f"source SHA-256 mismatch for comparator: {path}")
        quality, quality_evidence = _quality_evidence(
            path, result, duration_ms, silence_intervals
        )
        comparators.append(
            Candidate(path.parent.name, path, result, quality, quality_evidence)
        )

    healthy = [
        candidate
        for candidate in comparators
        if TranscriptQualityStatus(candidate.quality["status"]) in _HEALTHY_QUALITY_STATES
    ]
    rows = []
    for start_ms in range(0, duration_ms, window_ms):
        end_ms = min(duration_ms, start_ms + window_ms)
        primary_text = _window_text(primary, start_ms, end_ms)
        comparisons = []
        for candidate in comparators:
            candidate_text = _window_text(candidate.result, start_ms, end_ms)
            comparisons.append(
                {
                    "candidate": candidate.label,
                    "agreement": round(token_agreement(primary_text, candidate_text), 6),
                    "primary_token_count": len(word_tokens(primary_text)),
                    "candidate_token_count": len(word_tokens(candidate_text)),
                    "quality_eligible": candidate in healthy,
                }
            )
        healthy_scores = [
            item["agreement"] for item in comparisons if item["quality_eligible"]
        ]
        agreement = min(healthy_scores) if healthy_scores else None
        rows.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "comparisons": comparisons,
                "consensus_agreement": agreement,
                "uncertain": agreement is None or agreement < low_agreement_threshold,
            }
        )

    whole_document = [
        {
            "candidate": candidate.label,
            "agreement": round(token_agreement(primary.text, candidate.result.text), 6),
            "quality_status": candidate.quality["status"],
        }
        for candidate in comparators
    ]
    healthy_document_scores = [
        item["agreement"]
        for item in whole_document
        if item["quality_status"]
        in {status.value for status in _HEALTHY_QUALITY_STATES}
    ]
    minimum_document_agreement = (
        min(healthy_document_scores) if healthy_document_scores else None
    )
    uncertain_windows = [row for row in rows if row["uncertain"]]

    if primary_quality["status"] != TranscriptQualityStatus.PASS.value:
        status = "REJECTED_PRIMARY_QUALITY"
        reason = "The primary transcript does not pass the semantic quality contract."
    elif not healthy:
        status = "INSUFFICIENT_HEALTHY_COMPARATORS"
        reason = "No comparator passed the semantic quality contract."
    elif minimum_document_agreement is None or minimum_document_agreement < consensus_threshold:
        status = "AUTOMATED_CONSENSUS_WITH_UNCERTAINTY"
        reason = "Healthy local models disagree beyond the promotion threshold."
    elif uncertain_windows:
        status = "AUTOMATED_CONSENSUS_WITH_UNCERTAINTY"
        reason = "Document-level agreement passes, but low-agreement time windows remain."
    else:
        status = "AUTOMATED_CONSENSUS_PASS"
        reason = "All healthy local comparators pass the configured agreement gates."

    eligible_for_explicit_approval = status == "AUTOMATED_CONSENSUS_PASS"
    report = {
        "schema_version": "dubbing.local-transcript-adjudication.v1",
        "source_sha256": primary.source_sha256,
        "source": source_evidence,
        "policy": {
            "policy_version": "dubbing.local-consensus-policy.v1",
            "window_ms": window_ms,
            "low_agreement_threshold": low_agreement_threshold,
            "consensus_threshold": consensus_threshold,
            "cloud_allowed": False,
            "human_calibration_required": False,
            "consensus_is_ground_truth": False,
        },
        "primary": {
            "path": str(primary_file),
            "sha256": _sha256(primary_file),
            "backend": primary.backend,
            "model": primary.model,
            "quality": primary_quality,
            "quality_evidence": primary_quality_evidence,
        },
        "comparators": [
            {
                "label": candidate.label,
                "path": str(candidate.path),
                "sha256": _sha256(candidate.path),
                "backend": candidate.result.backend,
                "model": candidate.result.model,
                "quality": candidate.quality,
                "quality_evidence": candidate.quality_evidence,
            }
            for candidate in comparators
        ],
        "whole_document_comparisons": whole_document,
        "minimum_healthy_document_agreement": minimum_document_agreement,
        "windows": rows,
        "uncertain_window_count": len(uncertain_windows),
        "verdict": {
            "status": status,
            "reason": reason,
            "accuracy_certified_against_ground_truth": False,
            "eligible_for_explicit_approval": eligible_for_explicit_approval,
            "giga_admission_emitted": False,
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
