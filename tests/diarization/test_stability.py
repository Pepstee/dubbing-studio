from __future__ import annotations

import json

from dubbing.diarization import (
    CannotLinkEvidence,
    DiarizationStabilityPolicy,
    SpeakerTurn,
    ThresholdPartition,
    evaluate_diarization_stability,
)


def _partition(threshold: float, turns: tuple[SpeakerTurn, ...]) -> ThresholdPartition:
    return ThresholdPartition(threshold=threshold, turns=turns)


def _two_speakers(
    threshold: float, *, labels: tuple[str, str] = ("A", "B")
) -> ThresholdPartition:
    return _partition(
        threshold,
        (
            SpeakerTurn(0, 1000, labels[0]),
            SpeakerTurn(1000, 2000, labels[1]),
        ),
    )


def _cannot_link(similarity: float = 0.20, threshold: float = 0.50):
    return (
        CannotLinkEvidence(
            left_speaker="A",
            right_speaker="B",
            similarity=similarity,
            decision_threshold=threshold,
        ),
    )


def test_stable_label_invariant_plateau_passes() -> None:
    report = evaluate_diarization_stability(
        (
            _two_speakers(0.40, labels=("A", "B")),
            _two_speakers(0.45, labels=("speaker-9", "speaker-2")),
            _two_speakers(0.50, labels=("x", "y")),
        ),
        speech_duration_ms=2000,
        embedding_covered_ms=1950,
        transient_speech_ms=50,
        cannot_link=_cannot_link(),
    )

    assert report.status == "AUTO_STABLE"
    assert report.reasons == ()
    assert report.speaker_counts == (2, 2, 2)
    assert report.plateau_candidate_count == 3
    assert report.plateau_width == 0.1
    assert report.minimum_label_invariant_frame_agreement == 1.0
    assert report.minimum_pairwise_partition_agreement == 1.0


def test_label_mapping_cannot_hide_different_partition() -> None:
    report = evaluate_diarization_stability(
        (
            _two_speakers(0.40),
            _partition(
                0.45,
                (
                    SpeakerTurn(0, 500, "X"),
                    SpeakerTurn(500, 1500, "Y"),
                    SpeakerTurn(1500, 2000, "X"),
                ),
            ),
            _two_speakers(0.50, labels=("Q", "R")),
        ),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(),
    )

    assert report.status == "AUTO_UNCERTAIN"
    assert "UNSTABLE_FRAME_ASSIGNMENT" in report.reasons
    assert "UNSTABLE_PAIRWISE_PARTITION" in report.reasons


def test_non_monotonic_speaker_counts_are_uncertain() -> None:
    one = (SpeakerTurn(0, 2000, "A"),)
    three = (
        SpeakerTurn(0, 600, "A"),
        SpeakerTurn(600, 1300, "B"),
        SpeakerTurn(1300, 2000, "C"),
    )
    report = evaluate_diarization_stability(
        (_partition(0.4, three), _partition(0.45, one), _partition(0.5, three)),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(),
    )

    assert report.status == "AUTO_UNCERTAIN"
    assert report.count_monotonic is False
    assert "SPEAKER_COUNT_NOT_MONOTONIC" in report.reasons


def test_speech_weighted_embedding_and_transient_gates() -> None:
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(0.45), _two_speakers(0.5)),
        speech_duration_ms=10_000,
        embedding_covered_ms=8_000,
        transient_speech_ms=1_500,
        cannot_link=_cannot_link(),
    )

    assert report.status == "AUTO_UNCERTAIN"
    assert report.embedding_coverage == 0.8
    assert report.transient_burden == 0.15
    assert "LOW_EMBEDDING_COVERAGE" in report.reasons
    assert "HIGH_TRANSIENT_SPEAKER_BURDEN" in report.reasons


def test_current_canary_like_hard_negative_overlap_is_uncertain() -> None:
    """Overlap-implied different speakers with near-identical embeddings must not auto-pass."""

    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(0.45), _two_speakers(0.5)),
        speech_duration_ms=10_000,
        embedding_covered_ms=9_800,
        transient_speech_ms=200,
        cannot_link=_cannot_link(similarity=0.49, threshold=0.50),
    )

    assert report.status == "AUTO_UNCERTAIN"
    assert report.cannot_link_violation_count == 0
    assert report.minimum_cannot_link_separation == 0.01
    assert report.reasons == ("WEAK_CANNOT_LINK_SEPARATION",)


def test_cannot_link_threshold_violation_is_uncertain() -> None:
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(0.45), _two_speakers(0.5)),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(similarity=0.51, threshold=0.50),
    )

    assert report.status == "AUTO_UNCERTAIN"
    assert report.cannot_link_violation_count == 1
    assert "CANNOT_LINK_VIOLATION" in report.reasons


def test_missing_cannot_link_evidence_never_auto_passes() -> None:
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(0.45), _two_speakers(0.5)),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
    )

    assert report.status == "AUTO_UNCERTAIN"
    assert report.reasons == ("CANNOT_LINK_EVIDENCE_MISSING",)


def test_malformed_or_insufficient_evidence_fails_closed() -> None:
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(0.5)),
        speech_duration_ms=0,
        embedding_covered_ms=1,
        transient_speech_ms=0,
        cannot_link=_cannot_link(),
    )

    assert report.status == "FAILED"
    assert "INSUFFICIENT_THRESHOLD_CANDIDATES" in report.reasons
    assert "NO_SPEECH_EVIDENCE" in report.reasons
    assert "INVALID_SPEECH_WEIGHTED_EVIDENCE" in report.reasons


def test_overlap_reduces_comparability_without_becoming_malformed() -> None:
    overlapping = _partition(
        0.45,
        (SpeakerTurn(0, 1500, "A"), SpeakerTurn(1000, 2000, "B")),
    )
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), overlapping, _two_speakers(0.5)),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(),
    )

    assert report.status == "AUTO_UNCERTAIN"
    assert "LOW_COMPARABLE_FRAME_COVERAGE" in report.reasons


def test_report_json_is_deterministic_and_sorted() -> None:
    candidates = (_two_speakers(0.5), _two_speakers(0.4), _two_speakers(0.45))
    kwargs = dict(
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(),
    )

    first = evaluate_diarization_stability(candidates, **kwargs).to_json()
    second = evaluate_diarization_stability(tuple(reversed(candidates)), **kwargs).to_json()

    assert first == second
    assert json.loads(first)["schema_version"] == "dubbing.diarization-stability.v1"
    assert json.loads(first)["thresholds"] == [0.4, 0.45, 0.5]


def test_policy_can_raise_minimum_plateau_requirement() -> None:
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(0.45), _two_speakers(0.5)),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(),
        policy=DiarizationStabilityPolicy(minimum_plateau_candidates=4),
    )

    assert report.status == "AUTO_UNCERTAIN"
    assert report.reasons == ("NO_STABLE_SPEAKER_COUNT_PLATEAU",)


def test_invalid_cannot_link_similarity_fails_closed() -> None:
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(0.45), _two_speakers(0.5)),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(similarity=1.1),
    )

    assert report.status == "FAILED"
    assert report.reasons == ("INVALID_CANNOT_LINK_EVIDENCE",)


def test_non_finite_threshold_produces_strict_json_failed_report() -> None:
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(float("nan")), _two_speakers(0.5)),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(),
    )

    document = json.loads(report.to_json())
    assert report.status == "FAILED"
    assert report.reasons == ("INVALID_THRESHOLD",)
    assert None in document["thresholds"]


def test_cannot_link_gate_uses_unrounded_separation() -> None:
    report = evaluate_diarization_stability(
        (_two_speakers(0.4), _two_speakers(0.45), _two_speakers(0.5)),
        speech_duration_ms=2000,
        embedding_covered_ms=2000,
        transient_speech_ms=0,
        cannot_link=_cannot_link(similarity=0.4500004, threshold=0.50),
    )

    assert report.minimum_cannot_link_separation == 0.05
    assert report.status == "AUTO_UNCERTAIN"
    assert report.reasons == ("WEAK_CANNOT_LINK_SEPARATION",)
