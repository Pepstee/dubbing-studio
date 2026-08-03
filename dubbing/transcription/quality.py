from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


class TranscriptQualityStatus(str, Enum):
    PASS = "PASS"
    PASS_WITH_UNCERTAIN_SPANS = "PASS_WITH_UNCERTAIN_SPANS"
    REPROCESS_REQUIRED = "REPROCESS_REQUIRED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    start_ms: int | None = None
    end_ms: int | None = None
    evidence: dict | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TranscriptQualityReport:
    status: TranscriptQualityStatus
    issues: tuple[QualityIssue, ...]
    metrics: dict
    policy_version: str = "dubbing.transcript-quality-policy.v1"

    @property
    def approval_allowed(self) -> bool:
        return self.status is TranscriptQualityStatus.PASS

    def to_dict(self) -> dict:
        return {
            "schema_version": "dubbing.transcript-quality-report.v1",
            "policy_version": self.policy_version,
            "status": self.status.value,
            "approval_allowed": self.approval_allowed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, document: dict) -> TranscriptQualityReport:
        if document.get("schema_version") != "dubbing.transcript-quality-report.v1":
            raise ValueError("unsupported transcript quality report schema")
        return cls(
            status=TranscriptQualityStatus(document["status"]),
            policy_version=document.get(
                "policy_version", "dubbing.transcript-quality-policy.v1"
            ),
            issues=tuple(
                QualityIssue(
                    code=item["code"],
                    severity=item["severity"],
                    message=item["message"],
                    start_ms=item.get("start_ms"),
                    end_ms=item.get("end_ms"),
                    evidence=item.get("evidence"),
                )
                for item in document.get("issues", [])
            ),
            metrics=document.get("metrics", {}),
        )


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _tokens(value: str) -> list[str]:
    normalized = _normalize(value)
    return normalized.split() if normalized else []


def _script_counts(value: str) -> dict[str, int]:
    counts = {"latin": 0, "cyrillic": 0, "hangul": 0, "other": 0}
    for character in value:
        if not character.isalpha():
            continue
        point = ord(character)
        if 0x0400 <= point <= 0x052F:
            counts["cyrillic"] += 1
        elif 0xAC00 <= point <= 0xD7AF or 0x1100 <= point <= 0x11FF:
            counts["hangul"] += 1
        elif "LATIN" in unicodedata.name(character, ""):
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return counts


def _max_token_run(tokens: list[str]) -> tuple[str | None, int]:
    best_token = None
    best = current = 0
    previous = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        if current > best:
            best_token, best = token, current
        previous = token
    return best_token, best


def _max_phrase_run(tokens: list[str], max_width: int = 6) -> tuple[str | None, int]:
    best_phrase = None
    best_repeats = 1
    for width in range(1, min(max_width, len(tokens)) + 1):
        index = 0
        while index + 2 * width <= len(tokens):
            phrase = tokens[index : index + width]
            repeats = 1
            cursor = index + width
            while tokens[cursor : cursor + width] == phrase:
                repeats += 1
                cursor += width
            if repeats > best_repeats:
                best_phrase = " ".join(phrase)
                best_repeats = repeats
            index = cursor if repeats > 1 else index + 1
    return best_phrase, best_repeats


def _covered_ms(segments: Iterable[TranscriptSegment]) -> int:
    intervals = sorted((segment.start_ms, segment.end_ms) for segment in segments)
    if not intervals:
        return 0
    total = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def evaluate_transcript_quality(
    transcript: TranscriptionResult,
    *,
    expected_duration_ms: int | None = None,
    max_gap_ms: int = 60_000,
) -> TranscriptQualityReport:
    """Evaluate semantic and structural fitness; never equate valid JSON with quality."""
    issues: list[QualityIssue] = []
    segments = transcript.segments
    duration_ms = expected_duration_ms or transcript.duration_ms or 0
    tokens = _tokens(transcript.text)

    if not segments or not tokens:
        issues.append(
            QualityIssue(
                "malformed_or_empty_output",
                "fatal",
                "Transcript has no usable timestamped speech.",
            )
        )

    adjacent_duplicates = 0
    adjacent_duplicate_chain = 0
    max_adjacent_duplicate_chain = 0
    timestamp_overlaps = 0
    large_gaps: list[tuple[int, int]] = []
    previous: TranscriptSegment | None = None
    for segment in segments:
        if previous is not None:
            if _normalize(previous.text) == _normalize(segment.text):
                adjacent_duplicates += 1
                adjacent_duplicate_chain += 1
                max_adjacent_duplicate_chain = max(
                    max_adjacent_duplicate_chain, adjacent_duplicate_chain
                )
            else:
                adjacent_duplicate_chain = 0
            if segment.start_ms < previous.start_ms:
                issues.append(
                    QualityIssue(
                        "timestamp_disorder",
                        "fatal",
                        "Transcript segments are not monotonic.",
                        segment.start_ms,
                        segment.end_ms,
                    )
                )
            if segment.start_ms < previous.end_ms:
                timestamp_overlaps += 1
            gap = segment.start_ms - previous.end_ms
            if gap > max_gap_ms:
                large_gaps.append((previous.end_ms, segment.start_ms))
        diagnostics = segment.diagnostics
        if segment.text == "[UNCERTAIN: LOCAL TRANSCRIPTION FAILED]":
            issues.append(
                QualityIssue(
                    "span_transcription_failed",
                    "critical",
                    "A local span exhausted all configured transcription attempts.",
                    segment.start_ms,
                    segment.end_ms,
                )
            )
        if diagnostics and diagnostics.fallback_exhausted:
            issues.append(
                QualityIssue(
                    "decoder_fallback_exhausted",
                    "critical",
                    "Decoder exhausted its configured fallbacks.",
                    segment.start_ms,
                    segment.end_ms,
                    {"fallback_history": list(diagnostics.fallback_history)},
                )
            )
        previous = segment

    token, token_run = _max_token_run(tokens)
    phrase, phrase_run = _max_phrase_run(tokens)
    if (
        token_run >= 8
        or phrase_run >= 6
        or adjacent_duplicates >= 20
        or max_adjacent_duplicate_chain >= 4
    ):
        issues.append(
            QualityIssue(
                "pathological_repetition",
                "critical",
                "Repeated decoding output indicates a hallucination loop.",
                evidence={
                    "max_token": token,
                    "max_token_run": token_run,
                    "max_phrase": phrase,
                    "max_phrase_run": phrase_run,
                    "adjacent_duplicate_segment_transitions": adjacent_duplicates,
                    "max_adjacent_duplicate_chain": max_adjacent_duplicate_chain,
                },
            )
        )

    spoken_ms = sum(segment.end_ms - segment.start_ms for segment in segments)
    speaking_rate = len(tokens) / (spoken_ms / 1000) if spoken_ms else 0.0
    if speaking_rate > 7.5:
        issues.append(
            QualityIssue(
                "implausible_speaking_rate",
                "critical",
                "Token rate is implausibly high for conversational speech.",
                evidence={"tokens_per_second": round(speaking_rate, 4)},
            )
        )

    coverage_ms = _covered_ms(segments)
    coverage_ratio = coverage_ms / duration_ms if duration_ms else 0.0
    if duration_ms >= 60_000 and coverage_ratio < 0.02:
        issues.append(
            QualityIssue(
                "weak_speech_coverage",
                "review",
                "Timestamped speech covers less than 2% of the recording.",
                evidence={"speech_coverage_ratio": round(coverage_ratio, 6)},
            )
        )
    if large_gaps:
        issues.append(
            QualityIssue(
                "large_unexplained_gaps",
                "review",
                "Long gaps require comparison with VAD evidence.",
                evidence={"gap_count": len(large_gaps), "largest_gap_ms": max(b - a for a, b in large_gaps)},
            )
        )
    if timestamp_overlaps > max(2, len(segments) // 20):
        issues.append(
            QualityIssue(
                "excessive_timestamp_overlap",
                "critical",
                "Too many segments overlap in time.",
                evidence={"overlap_count": timestamp_overlaps},
            )
        )

    uncertain_count = sum(segment.uncertain for segment in segments)
    script_counts = _script_counts(transcript.text)
    expected_language = transcript.language
    if expected_language in {"ru"} and script_counts["latin"] > 4 * max(1, script_counts["cyrillic"]):
        issues.append(
            QualityIssue(
                "script_language_mismatch",
                "review",
                "Declared Russian output is overwhelmingly Latin script.",
                evidence=script_counts,
            )
        )
    if expected_language in {"ko"} and script_counts["latin"] > 4 * max(1, script_counts["hangul"]):
        issues.append(
            QualityIssue(
                "script_language_mismatch",
                "review",
                "Declared Korean output is overwhelmingly Latin script.",
                evidence=script_counts,
            )
        )

    severities = {issue.severity for issue in issues}
    if "fatal" in severities:
        status = TranscriptQualityStatus.FAILED
    elif "critical" in severities:
        status = TranscriptQualityStatus.REPROCESS_REQUIRED
    elif "review" in severities:
        status = TranscriptQualityStatus.HUMAN_REVIEW_REQUIRED
    elif uncertain_count:
        status = TranscriptQualityStatus.PASS_WITH_UNCERTAIN_SPANS
    else:
        status = TranscriptQualityStatus.PASS

    return TranscriptQualityReport(
        status=status,
        issues=tuple(issues),
        metrics={
            "duration_ms": duration_ms,
            "segment_count": len(segments),
            "token_count": len(tokens),
            "covered_ms": coverage_ms,
            "speech_coverage_ratio": round(coverage_ratio, 6),
            "tokens_per_second_of_segment_time": round(speaking_rate, 4),
            "adjacent_duplicate_segment_transitions": adjacent_duplicates,
            "max_adjacent_duplicate_chain": max_adjacent_duplicate_chain,
            "max_token_run": token_run,
            "max_phrase_run": phrase_run,
            "timestamp_overlap_count": timestamp_overlaps,
            "large_gap_count": len(large_gaps),
            "uncertain_segment_count": uncertain_count,
            "script_counts": script_counts,
        },
    )
