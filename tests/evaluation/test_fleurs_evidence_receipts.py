from __future__ import annotations

import hashlib
import json
from collections import defaultdict
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
    selected = _load("auto-beam5-fallback06-vad-report.json")
    baseline = _load("auto-beam5-fallback06-report.json")

    assert _sha256("manifest.json") == verdict["fixture"]["manifest_sha256"]
    assert (
        _sha256("auto-beam5-fallback06-vad-report.json")
        == verdict["selected_experiment"]["report_sha256"]
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

    baseline_spans = {
        row["id"]: row
        for row in baseline["methods"]["automatic_minimum_beam"]["spans"]
    }
    vad_spans = {
        row["id"]: row
        for row in selected["methods"]["automatic_minimum_beam"]["spans"]
    }
    assert baseline_spans.keys() == vad_spans.keys()

    aggregates: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "reference_words": 0,
            "oracle_edits": 0,
            "vad_wins": 0,
            "non_vad_wins": 0,
            "ties": 0,
        }
    )
    for identifier, baseline_row in baseline_spans.items():
        vad_row = vad_spans[identifier]
        language = baseline_row["language"]
        assert vad_row["language"] == language
        baseline_edits = baseline_row["wer"]["edits"]
        vad_edits = vad_row["wer"]["edits"]
        aggregate = aggregates[language]
        aggregate["reference_words"] += baseline_row["wer"]["reference_units"]
        aggregate["oracle_edits"] += min(baseline_edits, vad_edits)
        if vad_edits < baseline_edits:
            aggregate["vad_wins"] += 1
        elif baseline_edits < vad_edits:
            aggregate["non_vad_wins"] += 1
        else:
            aggregate["ties"] += 1

    for language, aggregate in aggregates.items():
        expected = verdict["paired_vad_oracle"]["per_language"][language]
        assert aggregate == {
            key: expected[key]
            for key in (
                "reference_words",
                "oracle_edits",
                "vad_wins",
                "non_vad_wins",
                "ties",
            )
        }
        accuracy = 1.0 - aggregate["oracle_edits"] / aggregate["reference_words"]
        assert accuracy == pytest.approx(expected["oracle_word_accuracy"])

    assert verdict["paired_vad_oracle"]["all_language_targets_passed"] is False
    assert verdict["gates"]["every_language_accuracy_at_least_90_percent"] is False
    assert verdict["gates"]["giga_admission_emitted"] is False
