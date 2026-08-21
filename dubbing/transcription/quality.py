from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from dubbing.transcription.language import has_dominant_unsupported_script
from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


QUALITY_POLICY_VERSION = "dubbing.transcript-quality-policy.v6"


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
    policy_version: str = QUALITY_POLICY_VERSION

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


_STOCK_HALLUCINATION_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "nu uitați să vă abonați",
    "mulțumim pentru vizionare",
    "субтитры сделал",
    "продолжение следует",
    "다음 영상에서 만나요",
    "시청해주셔서 감사합니다",
    "ご視聴ありがとうございました",
)


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


def _max_phrase_run(tokens: list[str], max_width: int = 20) -> tuple[str | None, int]:
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


@dataclass(frozen=True)
class RepetitionFinding:
    kind: str
    phrase: str
    repeat_count: int
    repeated_token_count: int
    phrase_width: int
    start_ms: int
    end_ms: int

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "phrase": self.phrase,
            "repeat_count": self.repeat_count,
            "repeated_token_count": self.repeated_token_count,
            "phrase_width": self.phrase_width,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


@dataclass(frozen=True)
class _RepetitionCandidate:
    token_start: int
    token_end: int
    phrase: tuple[str, ...]
    repeat_count: int

    @property
    def phrase_width(self) -> int:
        return len(self.phrase)

    @property
    def repeated_token_count(self) -> int:
        return self.token_end - self.token_start


def _repetition_threshold_met(phrase_width: int, repeat_count: int) -> bool:
    repeated_tokens = phrase_width * repeat_count
    if phrase_width == 1:
        return repeat_count >= 8
    if phrase_width <= 3:
        return repeat_count >= 6
    return repeat_count >= 4 and repeated_tokens >= 24


def detect_repetition_pathologies(
    segments: Iterable[TranscriptSegment],
    *,
    maximum_phrase_width: int = 20,
) -> tuple[RepetitionFinding, ...]:
    """Return timestamp-localized decoder-loop evidence.

    Thresholds depend on phrase width so ordinary conversational acknowledgements
    do not receive the same treatment as long clauses repeated mechanically.
    """
    ordered = tuple(segments)
    if maximum_phrase_width < 1:
        raise ValueError("maximum_phrase_width must be positive")
    tokens: list[str] = []
    token_segments: list[int] = []
    for segment_index, segment in enumerate(ordered):
        segment_tokens = _tokens(segment.text)
        tokens.extend(segment_tokens)
        token_segments.extend([segment_index] * len(segment_tokens))

    candidates: list[_RepetitionCandidate] = []
    for start in range(len(tokens)):
        available_width = min(maximum_phrase_width, (len(tokens) - start) // 2)
        for width in range(1, available_width + 1):
            phrase = tuple(tokens[start : start + width])
            repeats = 1
            cursor = start + width
            while tokens[cursor : cursor + width] == list(phrase):
                repeats += 1
                cursor += width
            if _repetition_threshold_met(width, repeats):
                candidates.append(
                    _RepetitionCandidate(start, cursor, phrase, repeats)
                )

    # Prefer the primitive phrase covering the largest run and suppress the same
    # run rediscovered at later token offsets or as a multiple-width phrase.
    selected: list[_RepetitionCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.token_start,
            -item.repeated_token_count,
            item.phrase_width,
        ),
    ):
        if any(
            (
                candidate.token_start >= existing.token_start
                and candidate.token_end <= existing.token_end
            )
            or (
                max(
                    0,
                    min(candidate.token_end, existing.token_end)
                    - max(candidate.token_start, existing.token_start),
                )
                / min(candidate.repeated_token_count, existing.repeated_token_count)
                >= 0.8
            )
            for existing in selected
        ):
            continue
        selected.append(candidate)

    findings = [
        RepetitionFinding(
            kind="repeated_token" if item.phrase_width == 1 else "repeated_phrase",
            phrase=" ".join(item.phrase),
            repeat_count=item.repeat_count,
            repeated_token_count=item.repeated_token_count,
            phrase_width=item.phrase_width,
            start_ms=ordered[token_segments[item.token_start]].start_ms,
            end_ms=ordered[token_segments[item.token_end - 1]].end_ms,
        )
        for item in selected
    ]

    chain_start = 0
    for index in range(1, len(ordered) + 1):
        still_equal = (
            index < len(ordered)
            and _normalize(ordered[index].text)
            and _normalize(ordered[index].text) == _normalize(ordered[index - 1].text)
        )
        if still_equal:
            continue
        repeat_count = index - chain_start
        phrase_tokens = _tokens(ordered[chain_start].text)
        if phrase_tokens and _repetition_threshold_met(
            len(phrase_tokens), repeat_count
        ):
            finding = RepetitionFinding(
                kind="repeated_segment",
                phrase=_normalize(ordered[chain_start].text),
                repeat_count=repeat_count,
                repeated_token_count=len(phrase_tokens) * repeat_count,
                phrase_width=len(phrase_tokens),
                start_ms=ordered[chain_start].start_ms,
                end_ms=ordered[index - 1].end_ms,
            )
            if not any(
                finding.start_ms >= existing.start_ms
                and finding.end_ms <= existing.end_ms
                for existing in findings
            ):
                findings.append(finding)
        chain_start = index
    return tuple(sorted(findings, key=lambda item: (item.start_ms, item.end_ms)))


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


def _interval_coverage_ms(
    start_ms: int,
    end_ms: int,
    intervals: Iterable[tuple[int, int]],
) -> int:
    clipped = sorted(
        (max(start_ms, start), min(end_ms, end))
        for start, end in intervals
        if start < end_ms and end > start_ms
    )
    if not clipped:
        return 0
    covered = 0
    cursor_start, cursor_end = clipped[0]
    for next_start, next_end in clipped[1:]:
        if next_start <= cursor_end:
            cursor_end = max(cursor_end, next_end)
        else:
            covered += cursor_end - cursor_start
            cursor_start, cursor_end = next_start, next_end
    return covered + cursor_end - cursor_start


def evaluate_transcript_quality(
    transcript: TranscriptionResult,
    *,
    expected_duration_ms: int | None = None,
    max_gap_ms: int = 60_000,
    known_silence_intervals: tuple[tuple[int, int], ...] = (),
    explained_gap_silence_ratio: float = 0.8,
) -> TranscriptQualityReport:
    """Evaluate semantic and structural fitness; never equate valid JSON with quality."""
    issues: list[QualityIssue] = []
    segments = transcript.segments
    duration_ms = (
        expected_duration_ms
        if expected_duration_ms is not None
        else transcript.duration_ms or 0
    )
    duration_bounds = {
        name: value
        for name, value in (
            ("expected_duration_ms", expected_duration_ms),
            ("transcript_duration_ms", transcript.duration_ms),
        )
        if value is not None
    }
    timestamp_upper_bound_ms = (
        min(duration_bounds.values()) if duration_bounds else None
    )
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
    explained_silence_gaps: list[tuple[int, int, float]] = []
    previous: TranscriptSegment | None = None
    stock_hallucination_count = 0
    segment_timestamp_out_of_bounds_count = 0
    word_timestamp_out_of_bounds_count = 0
    word_recording_bounds_violation_count = 0
    word_segment_containment_violation_count = 0
    word_parent_boundary_straddle_count = 0
    for segment_index, segment in enumerate(segments):
        segment_interval_invalid = (
            segment.start_ms < 0 or segment.end_ms <= segment.start_ms
        )
        violated_duration_bounds = {
            name: bound
            for name, bound in duration_bounds.items()
            if segment.end_ms > bound
        }
        if segment_interval_invalid or violated_duration_bounds:
            segment_timestamp_out_of_bounds_count += 1
            issues.append(
                QualityIssue(
                    "timestamp_out_of_bounds",
                    "fatal" if segment_interval_invalid else "critical",
                    "Segment timestamps fall outside their valid recording interval.",
                    segment.start_ms,
                    segment.end_ms,
                    {
                        "kind": "segment",
                        "segment_index": segment_index,
                        "observed_start_ms": segment.start_ms,
                        "observed_end_ms": segment.end_ms,
                        "violated_duration_bounds": violated_duration_bounds,
                    },
                )
            )
        for word_index, word in enumerate(segment.words):
            word_interval_invalid = word.start_ms < 0 or word.end_ms <= word.start_ms
            word_duration_bounds = {
                name: bound
                for name, bound in duration_bounds.items()
                if word.end_ms > bound
            }
            straddles_segment_boundary = (
                word.start_ms < segment.start_ms or word.end_ms > segment.end_ms
            )
            word_midpoint_ms = word.start_ms + (word.end_ms - word.start_ms) // 2
            outside_segment = not (
                segment.start_ms <= word_midpoint_ms < segment.end_ms
            )
            if straddles_segment_boundary:
                word_parent_boundary_straddle_count += 1
            if not (word_interval_invalid or word_duration_bounds or outside_segment):
                continue
            word_timestamp_out_of_bounds_count += 1
            if word_duration_bounds:
                word_recording_bounds_violation_count += 1
            if outside_segment:
                word_segment_containment_violation_count += 1
            issues.append(
                QualityIssue(
                    "timestamp_out_of_bounds",
                    "fatal" if word_interval_invalid else "critical",
                    "Word timestamps fall outside their recording or parent segment interval.",
                    word.start_ms,
                    word.end_ms,
                    {
                        "kind": "word",
                        "segment_index": segment_index,
                        "word_index": word_index,
                        "observed_start_ms": word.start_ms,
                        "observed_end_ms": word.end_ms,
                        "parent_segment_start_ms": segment.start_ms,
                        "parent_segment_end_ms": segment.end_ms,
                        "word_midpoint_ms": word_midpoint_ms,
                        "violated_duration_bounds": word_duration_bounds,
                        "outside_parent_segment": outside_segment,
                        "straddles_parent_segment_boundary": (
                            straddles_segment_boundary
                        ),
                    },
                )
            )
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
                silence_ms = _interval_coverage_ms(
                    previous.end_ms,
                    segment.start_ms,
                    known_silence_intervals,
                )
                silence_ratio = silence_ms / gap
                if silence_ratio >= explained_gap_silence_ratio:
                    explained_silence_gaps.append(
                        (previous.end_ms, segment.start_ms, silence_ratio)
                    )
                else:
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
        normalized_segment = _normalize(segment.text)
        matched_stock_phrase = next(
            (
                phrase
                for phrase in _STOCK_HALLUCINATION_PHRASES
                if _normalize(phrase) in normalized_segment
            ),
            None,
        )
        if (
            matched_stock_phrase is not None
            and diagnostics is not None
            and diagnostics.no_speech_probability is not None
            and diagnostics.no_speech_probability >= 0.6
        ):
            stock_hallucination_count += 1
            issues.append(
                QualityIssue(
                    "stock_hallucination_under_no_speech",
                    "critical",
                    "Decoder emitted known boilerplate despite strong no-speech evidence.",
                    segment.start_ms,
                    segment.end_ms,
                    {
                        "phrase": matched_stock_phrase,
                        "no_speech_probability": round(
                            diagnostics.no_speech_probability, 6
                        ),
                    },
                )
            )
        previous = segment

    token, token_run = _max_token_run(tokens)
    phrase, phrase_run = _max_phrase_run(tokens)
    repetition_findings = detect_repetition_pathologies(segments)
    for finding in repetition_findings:
        issues.append(
            QualityIssue(
                "pathological_repetition",
                "critical",
                "Repeated decoding output indicates a hallucination loop.",
                start_ms=finding.start_ms,
                end_ms=finding.end_ms,
                evidence={
                    "kind": finding.kind,
                    "phrase": finding.phrase,
                    "repeat_count": finding.repeat_count,
                    "repeated_token_count": finding.repeated_token_count,
                    "phrase_width": finding.phrase_width,
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
    unsupported_script_segment_count = 0
    for index, segment in enumerate(segments):
        if not has_dominant_unsupported_script(segment.text):
            continue
        unsupported_script_segment_count += 1
        issues.append(
            QualityIssue(
                "unsupported_script_language_mismatch",
                "critical",
                "A turn is dominated by a script outside the supported language set.",
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                evidence={
                    "segment_index": index,
                    "declared_language": segment.language,
                    "script_counts": _script_counts(segment.text),
                },
            )
        )
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
            "max_token": token,
            "max_phrase_run": phrase_run,
            "max_phrase": phrase,
            "repetition_finding_count": len(repetition_findings),
            "repetition_findings": [
                finding.to_dict() for finding in repetition_findings
            ],
            "stock_hallucination_under_no_speech_count": stock_hallucination_count,
            "timestamp_upper_bound_ms": timestamp_upper_bound_ms,
            "timestamp_out_of_bounds_count": (
                segment_timestamp_out_of_bounds_count
                + word_timestamp_out_of_bounds_count
            ),
            "segment_timestamp_out_of_bounds_count": (
                segment_timestamp_out_of_bounds_count
            ),
            "word_timestamp_out_of_bounds_count": (
                word_timestamp_out_of_bounds_count
            ),
            "word_recording_bounds_violation_count": (
                word_recording_bounds_violation_count
            ),
            "word_segment_containment_violation_count": (
                word_segment_containment_violation_count
            ),
            "word_parent_boundary_straddle_count": (
                word_parent_boundary_straddle_count
            ),
            "timestamp_overlap_count": timestamp_overlaps,
            "large_gap_count": len(large_gaps),
            "explained_silence_gap_count": len(explained_silence_gaps),
            "explained_silence_gaps": [
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "silence_ratio": round(silence_ratio, 6),
                }
                for start_ms, end_ms, silence_ratio in explained_silence_gaps
            ],
            "uncertain_segment_count": uncertain_count,
            "script_counts": script_counts,
            "unsupported_script_segment_count": unsupported_script_segment_count,
        },
    )
