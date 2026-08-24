from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from dubbing.diarization.models import SpeakerTurn


DiarizationStabilityStatus = Literal["AUTO_STABLE", "AUTO_UNCERTAIN", "FAILED"]


@dataclass(frozen=True)
class ThresholdPartition:
    """One diarization partition produced at a specific clustering threshold."""

    threshold: float
    turns: tuple[SpeakerTurn, ...]

    @property
    def speaker_count(self) -> int:
        return len({turn.speaker for turn in self.turns})


@dataclass(frozen=True)
class CannotLinkEvidence:
    """Similarity evidence for speakers that overlap and therefore cannot be identical."""

    left_speaker: str
    right_speaker: str
    similarity: float
    decision_threshold: float

    @property
    def separation(self) -> float:
        return self.decision_threshold - self.similarity


@dataclass(frozen=True)
class DiarizationStabilityPolicy:
    frame_ms: int = 100
    minimum_candidates: int = 3
    minimum_comparable_frame_ratio: float = 0.90
    minimum_label_invariant_frame_agreement: float = 0.90
    minimum_pairwise_partition_agreement: float = 0.95
    minimum_plateau_candidates: int = 2
    minimum_plateau_width: float = 0.05
    minimum_embedding_coverage: float = 0.90
    maximum_transient_burden: float = 0.10
    minimum_cannot_link_separation: float = 0.05


@dataclass(frozen=True)
class CandidateAgreement:
    left_threshold: float
    right_threshold: float
    comparable_frames: int
    speech_frames: int
    comparable_frame_ratio: float
    label_invariant_frame_agreement: float
    pairwise_partition_agreement: float

    def to_dict(self) -> dict:
        return {
            "left_threshold": _json_rounded(self.left_threshold),
            "right_threshold": _json_rounded(self.right_threshold),
            "comparable_frames": self.comparable_frames,
            "speech_frames": self.speech_frames,
            "comparable_frame_ratio": _json_rounded(self.comparable_frame_ratio),
            "label_invariant_frame_agreement": _json_rounded(
                self.label_invariant_frame_agreement
            ),
            "pairwise_partition_agreement": _json_rounded(
                self.pairwise_partition_agreement
            ),
        }


@dataclass(frozen=True)
class DiarizationStabilityReport:
    status: DiarizationStabilityStatus
    reasons: tuple[str, ...]
    candidate_count: int
    thresholds: tuple[float, ...]
    speaker_counts: tuple[int, ...]
    count_monotonic: bool
    plateau_speaker_count: int | None
    plateau_candidate_count: int
    plateau_width: float
    minimum_comparable_frame_ratio: float | None
    minimum_label_invariant_frame_agreement: float | None
    minimum_pairwise_partition_agreement: float | None
    speech_duration_ms: int
    embedding_covered_ms: int
    embedding_coverage: float | None
    transient_speech_ms: int
    transient_burden: float | None
    cannot_link_pair_count: int
    minimum_cannot_link_separation: float | None
    cannot_link_violation_count: int
    agreements: tuple[CandidateAgreement, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": "dubbing.diarization-stability.v1",
            "status": self.status,
            "reasons": list(self.reasons),
            "candidate_count": self.candidate_count,
            "thresholds": [_json_rounded(value) for value in self.thresholds],
            "speaker_counts": list(self.speaker_counts),
            "count_monotonic": self.count_monotonic,
            "stable_plateau": {
                "speaker_count": self.plateau_speaker_count,
                "candidate_count": self.plateau_candidate_count,
                "width": _json_rounded(self.plateau_width),
            },
            "agreement": {
                "minimum_comparable_frame_ratio": _optional_rounded(
                    self.minimum_comparable_frame_ratio
                ),
                "minimum_label_invariant_frame_agreement": _optional_rounded(
                    self.minimum_label_invariant_frame_agreement
                ),
                "minimum_pairwise_partition_agreement": _optional_rounded(
                    self.minimum_pairwise_partition_agreement
                ),
                "candidate_pairs": [item.to_dict() for item in self.agreements],
            },
            "speech_weighted_evidence": {
                "speech_duration_ms": self.speech_duration_ms,
                "embedding_covered_ms": self.embedding_covered_ms,
                "embedding_coverage": _optional_rounded(self.embedding_coverage),
                "transient_speech_ms": self.transient_speech_ms,
                "transient_burden": _optional_rounded(self.transient_burden),
            },
            "cannot_link": {
                "pair_count": self.cannot_link_pair_count,
                "minimum_separation": _optional_rounded(
                    self.minimum_cannot_link_separation
                ),
                "violation_count": self.cannot_link_violation_count,
            },
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def evaluate_diarization_stability(
    candidates: tuple[ThresholdPartition, ...],
    *,
    speech_duration_ms: int,
    embedding_covered_ms: int,
    transient_speech_ms: int,
    cannot_link: tuple[CannotLinkEvidence, ...] = (),
    policy: DiarizationStabilityPolicy | None = None,
) -> DiarizationStabilityReport:
    """Evaluate threshold robustness without trusting backend-specific speaker labels.

    The function is deliberately pure: it performs no model invocation or file I/O. Invalid
    or incomplete evidence produces a typed ``FAILED`` report instead of an exception or an
    optimistic default.
    """

    policy = policy or DiarizationStabilityPolicy()
    ordered = tuple(sorted(candidates, key=_candidate_sort_key))
    thresholds = tuple(item.threshold for item in ordered)
    speaker_counts = tuple(item.speaker_count for item in ordered)
    fatal_reasons = _validate_inputs(
        ordered,
        speech_duration_ms=speech_duration_ms,
        embedding_covered_ms=embedding_covered_ms,
        transient_speech_ms=transient_speech_ms,
        cannot_link=cannot_link,
        policy=policy,
    )
    count_monotonic = all(
        left <= right for left, right in zip(speaker_counts, speaker_counts[1:])
    )
    plateau_count, plateau_candidates, plateau_width = _stable_plateau(ordered)

    agreements: tuple[CandidateAgreement, ...] = ()
    if not fatal_reasons:
        agreements = tuple(
            _compare_partitions(left, right, frame_ms=policy.frame_ms)
            for left, right in combinations(ordered, 2)
        )
        if not agreements or any(item.comparable_frames < 2 for item in agreements):
            fatal_reasons.append("INSUFFICIENT_COMPARABLE_SPEECH_FRAMES")

    coverage = (
        embedding_covered_ms / speech_duration_ms if speech_duration_ms > 0 else None
    )
    transient_burden = (
        transient_speech_ms / speech_duration_ms if speech_duration_ms > 0 else None
    )
    separations = tuple(
        item.separation
        for item in cannot_link
        if math.isfinite(item.similarity) and math.isfinite(item.decision_threshold)
    )
    raw_minimum_separation = min(separations) if separations else None
    minimum_separation = _optional_rounded(raw_minimum_separation)
    violation_count = sum(item.similarity >= item.decision_threshold for item in cannot_link)

    minimum_comparable = min(
        (item.comparable_frame_ratio for item in agreements), default=None
    )
    minimum_frame_agreement = min(
        (item.label_invariant_frame_agreement for item in agreements), default=None
    )
    minimum_partition_agreement = min(
        (item.pairwise_partition_agreement for item in agreements), default=None
    )

    reasons = list(fatal_reasons)
    status: DiarizationStabilityStatus
    if fatal_reasons:
        status = "FAILED"
    else:
        if not count_monotonic:
            reasons.append("SPEAKER_COUNT_NOT_MONOTONIC")
        if plateau_candidates < policy.minimum_plateau_candidates:
            reasons.append("NO_STABLE_SPEAKER_COUNT_PLATEAU")
        elif plateau_width + 1e-12 < policy.minimum_plateau_width:
            reasons.append("SPEAKER_COUNT_PLATEAU_TOO_NARROW")
        if minimum_comparable is not None and (
            minimum_comparable < policy.minimum_comparable_frame_ratio
        ):
            reasons.append("LOW_COMPARABLE_FRAME_COVERAGE")
        if minimum_frame_agreement is not None and (
            minimum_frame_agreement < policy.minimum_label_invariant_frame_agreement
        ):
            reasons.append("UNSTABLE_FRAME_ASSIGNMENT")
        if minimum_partition_agreement is not None and (
            minimum_partition_agreement < policy.minimum_pairwise_partition_agreement
        ):
            reasons.append("UNSTABLE_PAIRWISE_PARTITION")
        if coverage is not None and coverage < policy.minimum_embedding_coverage:
            reasons.append("LOW_EMBEDDING_COVERAGE")
        if transient_burden is not None and transient_burden > policy.maximum_transient_burden:
            reasons.append("HIGH_TRANSIENT_SPEAKER_BURDEN")
        if not cannot_link:
            reasons.append("CANNOT_LINK_EVIDENCE_MISSING")
        elif violation_count:
            reasons.append("CANNOT_LINK_VIOLATION")
        elif (
            raw_minimum_separation is not None
            and raw_minimum_separation < policy.minimum_cannot_link_separation
        ):
            reasons.append("WEAK_CANNOT_LINK_SEPARATION")
        status = "AUTO_UNCERTAIN" if reasons else "AUTO_STABLE"

    return DiarizationStabilityReport(
        status=status,
        reasons=tuple(reasons),
        candidate_count=len(ordered),
        thresholds=thresholds,
        speaker_counts=speaker_counts,
        count_monotonic=count_monotonic,
        plateau_speaker_count=plateau_count,
        plateau_candidate_count=plateau_candidates,
        plateau_width=_rounded(plateau_width),
        minimum_comparable_frame_ratio=_optional_rounded(minimum_comparable),
        minimum_label_invariant_frame_agreement=_optional_rounded(minimum_frame_agreement),
        minimum_pairwise_partition_agreement=_optional_rounded(
            minimum_partition_agreement
        ),
        speech_duration_ms=speech_duration_ms,
        embedding_covered_ms=embedding_covered_ms,
        embedding_coverage=_optional_rounded(coverage),
        transient_speech_ms=transient_speech_ms,
        transient_burden=_optional_rounded(transient_burden),
        cannot_link_pair_count=len(cannot_link),
        minimum_cannot_link_separation=minimum_separation,
        cannot_link_violation_count=violation_count,
        agreements=agreements,
    )


def _validate_inputs(
    candidates: tuple[ThresholdPartition, ...],
    *,
    speech_duration_ms: int,
    embedding_covered_ms: int,
    transient_speech_ms: int,
    cannot_link: tuple[CannotLinkEvidence, ...],
    policy: DiarizationStabilityPolicy,
) -> list[str]:
    reasons: list[str] = []
    if (
        policy.frame_ms <= 0
        or policy.minimum_candidates < 2
        or policy.minimum_plateau_candidates < 2
        or not math.isfinite(policy.minimum_plateau_width)
        or policy.minimum_plateau_width < 0
        or not math.isfinite(policy.minimum_cannot_link_separation)
        or policy.minimum_cannot_link_separation < 0
    ):
        reasons.append("INVALID_POLICY")
    ratio_values = (
        policy.minimum_comparable_frame_ratio,
        policy.minimum_label_invariant_frame_agreement,
        policy.minimum_pairwise_partition_agreement,
        policy.minimum_embedding_coverage,
        policy.maximum_transient_burden,
    )
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in ratio_values):
        reasons.append("INVALID_POLICY")
    if len(candidates) < policy.minimum_candidates:
        reasons.append("INSUFFICIENT_THRESHOLD_CANDIDATES")
    if len(set(item.threshold for item in candidates)) != len(candidates):
        reasons.append("DUPLICATE_THRESHOLD_CANDIDATES")
    if any(
        not math.isfinite(item.threshold) or not 0 <= item.threshold <= 1
        for item in candidates
    ):
        reasons.append("INVALID_THRESHOLD")
    if any(not item.turns or item.speaker_count < 1 for item in candidates):
        reasons.append("EMPTY_DIARIZATION_PARTITION")
    if speech_duration_ms <= 0:
        reasons.append("NO_SPEECH_EVIDENCE")
    if (
        embedding_covered_ms < 0
        or transient_speech_ms < 0
        or embedding_covered_ms > speech_duration_ms
        or transient_speech_ms > speech_duration_ms
    ):
        reasons.append("INVALID_SPEECH_WEIGHTED_EVIDENCE")
    if any(
        not item.left_speaker
        or not item.right_speaker
        or item.left_speaker == item.right_speaker
        or not math.isfinite(item.similarity)
        or not -1 <= item.similarity <= 1
        or not math.isfinite(item.decision_threshold)
        or not 0 <= item.decision_threshold <= 1
        for item in cannot_link
    ):
        reasons.append("INVALID_CANNOT_LINK_EVIDENCE")
    return list(dict.fromkeys(reasons))


def _candidate_sort_key(candidate: ThresholdPartition) -> tuple:
    threshold_key = (
        (0, candidate.threshold)
        if math.isfinite(candidate.threshold)
        else (1, str(candidate.threshold))
    )
    turns_key = tuple(
        (turn.start_ms, turn.end_ms, turn.speaker, turn.confidence)
        for turn in candidate.turns
    )
    return (*threshold_key, turns_key)


def _stable_plateau(
    candidates: tuple[ThresholdPartition, ...],
) -> tuple[int | None, int, float]:
    if not candidates:
        return None, 0, 0.0
    if any(not math.isfinite(candidate.threshold) for candidate in candidates):
        return None, 0, 0.0
    plateaus: list[tuple[int, float, int, float]] = []
    start = 0
    counts = [item.speaker_count for item in candidates]
    for index in range(1, len(candidates) + 1):
        if index == len(candidates) or counts[index] != counts[start]:
            width = candidates[index - 1].threshold - candidates[start].threshold
            plateaus.append((index - start, width, counts[start], candidates[start].threshold))
            start = index
    length, width, count, _ = max(
        plateaus, key=lambda item: (item[0], item[1], -item[3])
    )
    return count, length, width


def _compare_partitions(
    left: ThresholdPartition,
    right: ThresholdPartition,
    *,
    frame_ms: int,
) -> CandidateAgreement:
    timeline_end = max(
        max(turn.end_ms for turn in left.turns),
        max(turn.end_ms for turn in right.turns),
    )
    left_frames = _frame_labels(left.turns, timeline_end=timeline_end, frame_ms=frame_ms)
    right_frames = _frame_labels(right.turns, timeline_end=timeline_end, frame_ms=frame_ms)
    left_labels: list[str] = []
    right_labels: list[str] = []
    speech_frames = 0
    for left_label, right_label in zip(left_frames, right_frames):
        if left_label is not None or right_label is not None:
            speech_frames += 1
        if left_label is not None and right_label is not None:
            left_labels.append(left_label)
            right_labels.append(right_label)

    comparable = len(left_labels)
    comparable_ratio = comparable / speech_frames if speech_frames else 0.0
    frame_agreement = _best_label_mapping_agreement(left_labels, right_labels)
    partition_agreement = _rand_partition_agreement(left_labels, right_labels)
    return CandidateAgreement(
        left_threshold=left.threshold,
        right_threshold=right.threshold,
        comparable_frames=comparable,
        speech_frames=speech_frames,
        comparable_frame_ratio=comparable_ratio,
        label_invariant_frame_agreement=frame_agreement,
        pairwise_partition_agreement=partition_agreement,
    )


def _frame_labels(
    turns: tuple[SpeakerTurn, ...], *, timeline_end: int, frame_ms: int
) -> tuple[str | None, ...]:
    ordered = sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker))
    active: list[SpeakerTurn] = []
    turn_index = 0
    labels: list[str | None] = []
    for start_ms in range(0, timeline_end, frame_ms):
        midpoint = start_ms + min(frame_ms, timeline_end - start_ms) / 2
        while turn_index < len(ordered) and ordered[turn_index].start_ms <= midpoint:
            active.append(ordered[turn_index])
            turn_index += 1
        active = [turn for turn in active if turn.end_ms > midpoint]
        speakers = {turn.speaker for turn in active}
        labels.append(next(iter(speakers)) if len(speakers) == 1 else None)
    return tuple(labels)


def _best_label_mapping_agreement(left: list[str], right: list[str]) -> float:
    if not left:
        return 0.0
    left_values = sorted(set(left))
    right_values = sorted(set(right))
    size = max(len(left_values), len(right_values))
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    left_index = {label: index for index, label in enumerate(left_values)}
    right_index = {label: index for index, label in enumerate(right_values)}
    for left_label, right_label in zip(left, right):
        matrix[left_index[left_label]][right_index[right_label]] += 1
    return _maximum_assignment_weight(matrix) / len(left)


def _maximum_assignment_weight(weights: list[list[int]]) -> int:
    """Maximum-weight square assignment using deterministic O(n^3) Hungarian matching."""

    size = len(weights)
    if not size:
        return 0
    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    potentials_left = [0] * (size + 1)
    potentials_right = [0] * (size + 1)
    matching = [0] * (size + 1)
    previous = [0] * (size + 1)
    for row in range(1, size + 1):
        matching[0] = row
        column = 0
        minimums = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = matching[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate_column - 1]
                    - potentials_left[current_row]
                    - potentials_right[candidate_column]
                )
                if reduced < minimums[candidate_column]:
                    minimums[candidate_column] = reduced
                    previous[candidate_column] = column
                if minimums[candidate_column] < delta:
                    delta = minimums[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    potentials_left[matching[candidate_column]] += delta
                    potentials_right[candidate_column] -= delta
                else:
                    minimums[candidate_column] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            next_column = previous[column]
            matching[column] = matching[next_column]
            column = next_column
            if column == 0:
                break
    return sum(
        weights[matching[column] - 1][column - 1]
        for column in range(1, size + 1)
    )


def _rand_partition_agreement(left: list[str], right: list[str]) -> float:
    count = len(left)
    if count < 2:
        return 0.0
    contingency: dict[tuple[str, str], int] = {}
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for left_label, right_label in zip(left, right):
        contingency[(left_label, right_label)] = (
            contingency.get((left_label, right_label), 0) + 1
        )
        left_counts[left_label] = left_counts.get(left_label, 0) + 1
        right_counts[right_label] = right_counts.get(right_label, 0) + 1
    def choose_two(value: int) -> int:
        return value * (value - 1) // 2

    total_pairs = choose_two(count)
    same_both = sum(choose_two(value) for value in contingency.values())
    same_left = sum(choose_two(value) for value in left_counts.values())
    same_right = sum(choose_two(value) for value in right_counts.values())
    different_both = total_pairs - same_left - same_right + same_both
    return (same_both + different_both) / total_pairs


def _rounded(value: float) -> float:
    return round(value, 6)


def _optional_rounded(value: float | None) -> float | None:
    return None if value is None else _rounded(value)


def _json_rounded(value: float) -> float | None:
    return _rounded(value) if math.isfinite(value) else None
