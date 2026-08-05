from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dubbing.evaluation.language_sweep import evaluate_language_sweep


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
                                    "words": [
                                        {"start_ms": 0, "end_ms": 500, "text": "hello"},
                                        {"start_ms": 500, "end_ms": 1000, "text": "there"},
                                    ]
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
                                    "words": [
                                        {"start_ms": 0, "end_ms": 500, "text": "hello"},
                                        {"start_ms": 500, "end_ms": 1000, "text": "world"},
                                    ]
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
    assert report["methods"]["maximum_average_log_probability"]["metrics"]["word_accuracy"] == 0.5
    assert report["methods"]["oracle_minimum_edits"]["metrics"]["word_accuracy"] == 1.0
    assert report["methods"]["oracle_minimum_edits"]["deployable"] is False
    assert report["promotion_passed"] is False
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
