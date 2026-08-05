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
    selected_name = "auto-beam5-fallback06-vad-pad200-report.json"
    selected = _load(selected_name)

    assert _sha256("manifest.json") == verdict["fixture"]["manifest_sha256"]
    assert (
        _sha256(selected_name) == verdict["selected_experiment"]["report_sha256"]
    )
    assert selected["sweep_sha256"] == verdict["selected_experiment"]["sweep_sha256"]
    assert selected["source_sha256"] == verdict["fixture"]["source_sha256"]
    assert selected["giga_admission_emitted"] is False
    assert selected["fixture_gate_passed"] is False
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

    selected_metrics = selected["methods"]["automatic_minimum_beam"]["metrics"]
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

    assert verdict["gates"]["every_language_accuracy_at_least_90_percent"] is False
    assert verdict["gates"]["giga_admission_emitted"] is False
