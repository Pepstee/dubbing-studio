from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "fleurs-validation-25x4"


def _load(name: str) -> dict:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def _sha256(name: str) -> str:
    return hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()


def test_fleurs_verdict_is_hash_bound_and_oracle_is_reproducible() -> None:
    verdict = _load("verdict.json")
    selected_name = "language-conditioned-global-confidence-report.json"
    selected = _load(selected_name)

    assert _sha256("manifest.json") == verdict["fixture"]["manifest_sha256"]
    assert (
        _sha256(selected_name) == verdict["selected_experiment"]["report_sha256"]
    )
    assert selected["sweep_sha256"] == verdict["selected_experiment"]["sweep_sha256"]
    assert selected["source_sha256"] == verdict["fixture"]["source_sha256"]
    assert selected["giga_admission_emitted"] is False
    assert selected["fixture_gate_passed"] is True
    assert selected["fixture_gate_passed_methods"] == ["language_conditioned_retry"]
    assert selected["promotion_passed"] is False

    control_files = {
        "automatic beam 5 temperature 0": "auto-beam5-report.json",
        "automatic beam 5 temperature 0 with multilingual disabled": (
            "auto-beam5-monolingual-report.json"
        ),
        "automatic beam 5 with capped fallback and word timestamps disabled": (
            "auto-beam5-fallback06-no-word-timestamps-report.json"
        ),
        "automatic beam 5 with capped fallback and previous-text conditioning": (
            "auto-beam5-fallback06-previous-text-report.json"
        ),
        "automatic beam 5 with capped fallback and VAD disabled": (
            "auto-beam5-fallback06-report.json"
        ),
        "VAD default 400 ms speech pad": "auto-beam5-fallback06-vad-report.json",
        "VAD 100 ms speech pad": "auto-beam5-fallback06-vad-pad100-report.json",
        "VAD 150 ms speech pad": "auto-beam5-fallback06-vad-pad150-report.json",
        "VAD 250 ms speech pad": "auto-beam5-fallback06-vad-pad250-report.json",
    }
    assert {item["name"] for item in verdict["negative_controls"]} == set(control_files)
    for item in verdict["negative_controls"]:
        report = _load(control_files[item["name"]])
        assert _sha256(control_files[item["name"]]) == item["report_sha256"]
        assert report["sweep_sha256"] == item["sweep_sha256"]
        assert report["giga_admission_emitted"] is False

    selected_metrics = selected["methods"]["language_conditioned_retry"]["metrics"]
    assert selected_metrics["word_accuracy"] == pytest.approx(
        verdict["selected_experiment"]["word_accuracy"]
    )
    assert selected_metrics["cer"]["rate"] == pytest.approx(
        verdict["selected_experiment"]["cer"]
    )

    variant_analysis = verdict["variant_selection_analysis"]
    merged = _load("vad-variant-merged-report.json")
    assert _sha256("vad-variant-merged-report.json") == variant_analysis["report_sha256"]
    assert merged["sweep_sha256"] == variant_analysis["merged_sweep_sha256"]
    assert merged["fixture_gate_passed"] is False
    assert merged["promotion_passed"] is False
    assert merged["giga_admission_emitted"] is False
    maximum_log_probability = merged["methods"]["maximum_average_log_probability"]
    oracle = merged["methods"]["oracle_minimum_edits"]
    assert maximum_log_probability["deployable"] is True
    assert maximum_log_probability["metrics"]["word_accuracy"] == pytest.approx(
        variant_analysis["maximum_average_log_probability"]["word_accuracy"]
    )
    assert maximum_log_probability["metrics"]["all_language_targets_passed"] is False
    assert oracle["deployable"] is False
    assert oracle["metrics"]["word_accuracy"] == pytest.approx(
        variant_analysis["oracle_minimum_edits"]["word_accuracy"]
    )
    assert oracle["metrics"]["all_language_targets_passed"] is True

    frozen_policy = verdict["frozen_policy"]
    assert _sha256(frozen_policy["path"]) == frozen_policy["sha256"]
    policy = _load(frozen_policy["path"])
    assert policy["frozen_before_holdout_access"] is True
    assert policy["holdout_contract"]["status_at_freeze"] == "NOT_ACCESSED_OR_EVALUATED"
    assert policy["giga_admission_emitted"] is False
    assert policy["validation_receipt"]["report_sha256"] == _sha256(selected_name)
    assert policy["validation_receipt"]["merged_sweep_sha256"] == selected["sweep_sha256"]

    assert verdict["gates"]["every_language_accuracy_at_least_90_percent"] is True
    assert verdict["gates"]["clean_validation_fixture_passed"] is True
    assert verdict["gates"]["untouched_holdout_passed"] is False
    assert verdict["gates"]["giga_admission_emitted"] is False


def test_post_holdout_validation_controls_are_rejected() -> None:
    controls = {
        "orthography-prompt-v1-report.json": (
            "5b88587c8ae93b9e4b86c75aecab112ddebecfc155de03befac3112773a099b9",
            0.8728606356968215,
        ),
        "forced-ko-beam10-v1-report.json": (
            "d243aec1c08e416083b74c9fca63d686342005f118115017cec69f505b493fd1",
            0.8973105134474327,
        ),
    }
    for name, (expected_sha256, korean_accuracy) in controls.items():
        report = _load(name)
        assert _sha256(name) == expected_sha256
        assert report["fixture_gate_passed"] is False
        assert report["promotion_passed"] is False
        assert report["giga_admission_emitted"] is False
        assert report["methods"]["language_conditioned_retry"]["per_language"]["ko"][
            "word_accuracy"
        ] == pytest.approx(korean_accuracy)
