from __future__ import annotations

from dubbing.evaluation.accuracy_gate import (
    build_accuracy_gate,
    build_accuracy_portfolio,
)


def _fixture(*, split: str = "test", eligible: bool = True) -> dict:
    return {
        "schema_version": "dubbing.fleurs-ground-truth.v1",
        "source": {"split": split},
        "coverage_gaps": [],
        "selection_policy": {
            "accuracy_certification_eligible": eligible,
            "reference_text_is_human_ground_truth": True,
            "reference_timestamps_are_human_ground_truth": True,
            "scope_limitation": "test fixture",
        },
    }


def _rows(*, errors: dict[str, int] | None = None) -> list[dict]:
    errors = errors or {}
    rows = []
    for language in ("en", "ru", "ro", "ko"):
        remaining = errors.get(language, 0)
        for index in range(25):
            row_errors = min(4, remaining)
            remaining -= row_errors
            rows.append(
                {
                    "id": f"{language}-{index}",
                    "language": language,
                    "wer": {
                        "reference_units": 4,
                        "candidate_units": 4,
                        "edits": row_errors,
                        "rate": row_errors / 4,
                    },
                    "cer": {
                        "reference_units": 20,
                        "candidate_units": 20,
                        "edits": row_errors,
                        "rate": row_errors / 20,
                    },
                }
            )
    return rows


def test_aggregate_above_ninety_cannot_hide_failing_language() -> None:
    gate = build_accuracy_gate(
        _fixture(),
        _rows(errors={"ro": 15}),
        deployable_selector=True,
        quality_gate_passed=True,
        word_timestamp_gate_passed=True,
    )

    assert gate["observations"]["aggregate_word"]["observed_accuracy"] > 0.9
    assert gate["measurement_gate"]["status"] == "FAIL"
    assert gate["benchmark_claim"]["status"] == "FAIL"
    assert any(
        reason["code"] == "RO_WORD_ACCURACY_BELOW_TARGET"
        for reason in gate["measurement_gate"]["performance_failures"]
    )


def test_tiny_perfect_fixture_is_insufficient_instead_of_pass() -> None:
    gate = build_accuracy_gate(
        _fixture(),
        [
            {
                "id": "en-1",
                "language": "en",
                "wer": {
                    "reference_units": 2,
                    "candidate_units": 2,
                    "edits": 0,
                    "rate": 0.0,
                },
                "cer": {
                    "reference_units": 10,
                    "candidate_units": 10,
                    "edits": 0,
                    "rate": 0.0,
                },
            }
        ],
        deployable_selector=True,
        quality_gate_passed=True,
        word_timestamp_gate_passed=True,
    )

    assert gate["observations"]["aggregate_word"]["observed_accuracy"] == 1.0
    assert gate["measurement_gate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert not gate["measurement_gate"]["passed"]


def test_reference_informed_selector_cannot_receive_benchmark_pass() -> None:
    gate = build_accuracy_gate(
        _fixture(),
        _rows(),
        deployable_selector=False,
        quality_gate_passed=True,
        word_timestamp_gate_passed=True,
    )

    assert gate["measurement_gate"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert gate["benchmark_claim"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert any(
        reason["code"] == "REFERENCE_INFORMED_SELECTOR"
        for reason in gate["benchmark_claim"]["evidence_gaps"]
    )


def test_adequate_heldout_fixture_can_pass_benchmark_but_not_production() -> None:
    gate = build_accuracy_gate(
        _fixture(),
        _rows(),
        deployable_selector=True,
        quality_gate_passed=True,
        word_timestamp_gate_passed=True,
    )

    assert gate["measurement_gate"]["status"] == "PASS"
    assert gate["benchmark_claim"]["status"] == "PASS"
    assert gate["production_claim"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert not gate["production_claim"]["passed"]


def test_validation_split_is_not_mislabelled_as_heldout_benchmark() -> None:
    gate = build_accuracy_gate(
        _fixture(split="validation"),
        _rows(),
        deployable_selector=True,
        quality_gate_passed=True,
        word_timestamp_gate_passed=True,
    )

    assert gate["measurement_gate"]["status"] == "PASS"
    assert gate["benchmark_claim"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert any(
        reason["code"] == "NOT_HELD_OUT"
        for reason in gate["benchmark_claim"]["evidence_gaps"]
    )


def _report(fixture: dict, rows: list[dict]) -> dict:
    return {
        "source_sha256": fixture.get("source", {}).get("sha256", "source"),
        "accuracy_gate": build_accuracy_gate(
            fixture,
            rows,
            deployable_selector=True,
            quality_gate_passed=True,
            word_timestamp_gate_passed=True,
        ),
    }


def test_portfolio_stays_insufficient_when_only_clean_read_speech_exists() -> None:
    portfolio = build_accuracy_portfolio([_report(_fixture(), _rows())])

    assert portfolio["status"] == "INSUFFICIENT_EVIDENCE"
    assert not portfolio["production_claim_passed"]
    assert "natural_long_form_holdout" in portfolio["missing_requirements"]


def test_complete_multi_scope_portfolio_can_make_explicit_production_claim() -> None:
    clean = _fixture()
    clean["source"]["sha256"] = "clean"
    noisy = {
        "schema_version": "dubbing.derived-noisy-codeswitch-ground-truth.v1",
        "source": {"sha256": "noisy"},
        "coverage_gaps": [],
        "selection_policy": {
            "accuracy_certification_eligible": False,
            "reference_text_is_human_ground_truth": True,
        },
    }
    natural = {
        "schema_version": "dubbing.natural-long-form-ground-truth.v1",
        "source": {"sha256": "natural"},
        "coverage_gaps": [],
        "selection_policy": {
            "accuracy_certification_eligible": True,
            "reference_text_is_human_ground_truth": True,
            "reference_timestamps_are_human_ground_truth": True,
        },
        "evidence_profile": {
            "scope": "natural_long_form",
            "held_out": True,
            "natural_audio": True,
            "long_form": True,
            "complete_recordings": True,
            "recording_count": 3,
            "duration_ms": 3 * 60 * 60 * 1000,
            "speaker_count": 3,
            "speech_coverage_measured": True,
            "uncertainty_rate": 0.02,
            "speaker_attributed": True,
            "overlapping_speech": True,
            "speaker_attributed_wer_measured": True,
            "der_jer_measured": True,
        },
    }

    portfolio = build_accuracy_portfolio(
        [_report(clean, _rows()), _report(noisy, _rows()), _report(natural, _rows())]
    )

    assert portfolio["status"] == "PASS"
    assert portfolio["production_claim_passed"]
    assert portfolio["missing_requirements"] == []


def test_failing_measurement_forces_portfolio_failure() -> None:
    portfolio = build_accuracy_portfolio(
        [_report(_fixture(), _rows(errors={"ro": 15}))]
    )

    assert portfolio["status"] == "FAIL"
    assert portfolio["failed_measurement_count"] == 1


def test_failed_comparator_does_not_override_passing_method_in_same_report() -> None:
    passing = _report(_fixture(), _rows())["accuracy_gate"]
    failing = _report(_fixture(), _rows(errors={"ro": 15}))["accuracy_gate"]
    portfolio = build_accuracy_portfolio(
        [
            {
                "source_sha256": "clean",
                "methods": {
                    "selected": {"deployable": True, "accuracy_gate": passing},
                    "comparator": {"deployable": True, "accuracy_gate": failing},
                },
            }
        ]
    )

    assert portfolio["status"] == "INSUFFICIENT_EVIDENCE"
    assert portfolio["failed_measurement_count"] == 0
