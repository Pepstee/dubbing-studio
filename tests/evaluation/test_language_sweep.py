from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dubbing.evaluation.language_sweep import _quality_diagnostics, evaluate_language_sweep


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def test_quality_diagnostics_fail_closed_on_rejected_spacing() -> None:
    diagnostics = _quality_diagnostics(
        [
            (
                {"id": "ko-1", "no_speech": False},
                {
                    "text": "한국어",
                    "segments": [],
                    "spacing_normalization": {
                        "status": "REJECTED_CHARACTER_CHANGE"
                    },
                },
            )
        ],
        [0.0],
    )

    assert diagnostics["quality_gate_passed"] is False
    assert diagnostics["issues"]["spacing_normalization_rejected"] == ["ko-1"]


def test_evaluate_language_sweep_separates_deployable_and_oracle(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture = {
        "source": {"sha256": "source"},
        "coverage_gaps": ["ro"],
        "spans": [
            {
                "id": "one",
                "text": "hello world",
                "language": "en",
                "no_speech": False,
                "start_ms": 0,
                "end_ms": 1000,
                "segment_start_ms": 0,
                "segment_end_ms": 1000,
            }
        ],
    }
    _write(fixture_path, fixture)
    sweep_path = tmp_path / "sweep.json"
    _write(
        sweep_path,
        {
            "source_sha256": "source",
            "fixture_sha256": _hash(fixture_path),
            "model": "model",
            "model_bin_sha256": "model-hash",
            "compute_type": "int8_float16",
            "runtime_seconds": 1.0,
            "beam_sizes": [1],
            "language_retry_policy": {"en": "always"},
            "spans": [
                {
                    "id": "one",
                    "declared_reference_language": "en",
                    "candidates": [
                        {
                            "forced_language": None,
                            "detected_language": "en",
                            "beam_size": 1,
                            "avg_log_probability": -0.1,
                            "text": "hello there",
                            "segments": [
                                {
                                    "start_ms": 0,
                                    "end_ms": 1000,
                                    "words": [
                                        {"start_ms": 0, "end_ms": 500, "text": "hello"},
                                        {"start_ms": 500, "end_ms": 1000, "text": "there"},
                                    ],
                                }
                            ],
                        },
                        {
                            "forced_language": "en",
                            "detected_language": "en",
                            "beam_size": 1,
                            "avg_log_probability": -0.2,
                            "text": "hello world",
                            "segments": [
                                {
                                    "start_ms": 0,
                                    "end_ms": 1000,
                                    "words": [
                                        {"start_ms": 0, "end_ms": 500, "text": "hello"},
                                        {"start_ms": 500, "end_ms": 1000, "text": "world"},
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        },
    )
    report = evaluate_language_sweep(fixture_path, sweep_path, tmp_path / "report.json")
    assert report["methods"]["automatic_minimum_beam"]["metrics"]["word_accuracy"] == 0.5
    assert report["methods"]["automatic_minimum_beam"]["metrics"]["cer"]["edits"] == 5
    assert report["methods"]["maximum_average_log_probability"]["metrics"]["word_accuracy"] == 0.5
    assert report["methods"]["detected_language_retry"]["metrics"]["word_accuracy"] == 1.0
    assert report["methods"]["language_conditioned_retry"]["metrics"]["word_accuracy"] == 1.0
    assert report["methods"]["detected_language_retry"]["deployable"] is True
    assert report["methods"]["oracle_minimum_edits"]["metrics"]["word_accuracy"] == 1.0
    assert report["methods"]["oracle_minimum_edits"]["deployable"] is False
    assert (
        report["methods"]["automatic_minimum_beam"]["metrics"][
            "every_required_language_word_threshold_passed"
        ]
        is False
    )
    assert report["measurement_gate_passed"] is False
    assert report["benchmark_claim_passed"] is False
    assert report["decode_configuration"]["condition_on_previous_text"] is False
    assert report["decode_configuration"]["word_timestamps"] is True
    assert report["decode_configuration"]["vad_filter"] is False
    assert report["decode_configuration"]["vad_threshold"] is None
    assert report["methods"]["automatic_minimum_beam"]["quality_diagnostics"] == {
        "selected_candidate_count": 1,
        "maximum_compression_ratio": None,
        "configured_maximum_temperature": 0.0,
        "maximum_used_temperature": 0.0,
        "fallback_used_count": 0,
        "fallback_used_ids": [],
        "issues": {
            "blank_hypotheses": [],
            "malformed_timestamps": [],
            "adjacent_duplicate_segments": [],
                "repeated_token_runs": [],
                "fallback_exhausted": [],
        },
        "quality_gate_passed": True,
    }
    assert report["production_claim_passed"] is False
    assert (
        report["methods"]["detected_language_retry"]["accuracy_gate"][
            "measurement_gate"
        ]["status"]
        == "INSUFFICIENT_EVIDENCE"
    )
    assert report["giga_admission_emitted"] is False


def test_evaluate_language_sweep_quality_gate_fails_closed(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture = {
        "source": {"sha256": "source"},
        "coverage_gaps": [],
        "selection_policy": {"accuracy_certification_eligible": True},
        "spans": [
            {
                "id": "blank",
                "text": "spoken words",
                "language": "en",
                "no_speech": False,
                "start_ms": 0,
                "end_ms": 1000,
                "segment_start_ms": 0,
                "segment_end_ms": 1000,
            },
            {
                "id": "loop",
                "text": "loop loop loop loop",
                "language": "en",
                "no_speech": False,
                "start_ms": 0,
                "end_ms": 1000,
                "segment_start_ms": 0,
                "segment_end_ms": 1000,
            },
        ],
    }
    _write(fixture_path, fixture)
    sweep_path = tmp_path / "sweep.json"
    candidates = {
        "blank": {"text": "", "segments": [], "decode_temperatures": [0.0]},
        "loop": {
            "text": "loop loop loop loop",
            "decode_temperatures": [0.6],
            "segments": [
                {
                    "start_ms": 100,
                    "end_ms": 50,
                    "text": "loop loop loop loop",
                    "compression_ratio": 3.0,
                    "temperature": 0.6,
                    "words": [],
                }
            ],
        },
    }
    _write(
        sweep_path,
        {
            "source_sha256": "source",
            "fixture_sha256": _hash(fixture_path),
            "model": "model",
            "model_bin_sha256": "model-hash",
            "compute_type": "int8_float16",
            "runtime_seconds": 1.0,
            "beam_sizes": [1],
            "temperatures": [0.0, 0.6],
            "spans": [
                {
                    "id": identifier,
                    "declared_reference_language": "en",
                    "candidates": [
                        {
                            "forced_language": None,
                            "detected_language": "en",
                            "beam_size": 1,
                            "avg_log_probability": -0.1,
                            **candidate,
                        }
                    ],
                }
                for identifier, candidate in candidates.items()
            ],
        },
    )

    report = evaluate_language_sweep(fixture_path, sweep_path, tmp_path / "report.json")
    diagnostics = report["methods"]["automatic_minimum_beam"]["quality_diagnostics"]
    assert diagnostics["quality_gate_passed"] is False
    assert diagnostics["issues"] == {
        "blank_hypotheses": ["blank"],
        "malformed_timestamps": ["loop"],
        "adjacent_duplicate_segments": [],
            "repeated_token_runs": ["loop"],
            "fallback_exhausted": ["loop"],
    }
    assert report["measurement_gate_passed"] is False
    assert report["giga_admission_emitted"] is False


def test_evaluate_language_sweep_does_not_certify_text_only_decode(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture = {
        "source": {"sha256": "source"},
        "coverage_gaps": [],
        "selection_policy": {"accuracy_certification_eligible": True},
        "spans": [
            {
                "id": "one",
                "text": "perfect text",
                "language": "en",
                "no_speech": False,
                "start_ms": 0,
                "end_ms": 1000,
                "segment_start_ms": 0,
                "segment_end_ms": 1000,
            }
        ],
    }
    _write(fixture_path, fixture)
    sweep_path = tmp_path / "sweep.json"
    _write(
        sweep_path,
        {
            "source_sha256": "source",
            "fixture_sha256": _hash(fixture_path),
            "model": "model",
            "model_bin_sha256": "model-hash",
            "compute_type": "int8_float16",
            "runtime_seconds": 1.0,
            "beam_sizes": [1],
            "temperatures": [0.0],
            "word_timestamps": False,
            "spans": [
                {
                    "id": "one",
                    "declared_reference_language": "en",
                    "candidates": [
                        {
                            "forced_language": None,
                            "detected_language": "en",
                            "beam_size": 1,
                            "avg_log_probability": -0.1,
                            "text": "perfect text",
                            "decode_temperatures": [0.0],
                            "segments": [
                                {
                                    "start_ms": 0,
                                    "end_ms": 1000,
                                    "text": "perfect text",
                                    "words": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )

    report = evaluate_language_sweep(
        fixture_path, sweep_path, tmp_path / "report.json", timing_policy="full_clip"
    )
    assert report["methods"]["automatic_minimum_beam"]["metrics"][
        "aggregate_word_accuracy_threshold_passed"
    ]
    assert report["methods"]["automatic_minimum_beam"]["quality_diagnostics"][
        "quality_gate_passed"
    ]
    assert report["word_timestamp_gate_passed"] is False
    assert report["measurement_gate_passed"] is False
    assert (
        report["methods"]["automatic_minimum_beam"]["accuracy_gate"][
            "measurement_gate"
        ]["status"]
        == "INSUFFICIENT_EVIDENCE"
    )
    assert report["giga_admission_emitted"] is False


def test_evaluate_language_sweep_rejects_fixture_hash_mismatch(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    _write(fixture_path, {"source": {"sha256": "source"}, "coverage_gaps": [], "spans": []})
    sweep_path = tmp_path / "sweep.json"
    _write(
        sweep_path,
        {
            "source_sha256": "source",
            "fixture_sha256": "wrong",
            "beam_sizes": [1],
            "spans": [],
        },
    )
    with pytest.raises(ValueError, match="fixture SHA-256"):
        evaluate_language_sweep(fixture_path, sweep_path, tmp_path / "report.json")
