from __future__ import annotations

import json

import pytest

from scripts.merge_faster_whisper_variant_sweeps import merge_variant_sweeps


def _sweep(text: str, *, source: str = "source") -> dict:
    return {
        "fixture_sha256": "fixture",
        "source_sha256": source,
        "model": "model",
        "model_bin_sha256": "model-hash",
        "compute_type": "int8_float16",
        "cloud_allowed": False,
        "giga_admission_emitted": False,
        "beam_sizes": [5],
        "forced_languages": ["auto"],
        "temperatures": [0.0],
        "runtime_seconds": 1.0,
        "word_timestamps": True,
        "vad_filter": False,
        "spans": [
            {
                "id": "one",
                "declared_reference_language": "en",
                "candidates": [
                    {
                        "forced_language": None,
                        "detected_language": "en",
                        "beam_size": 5,
                        "avg_log_probability": -0.1,
                        "text": text,
                        "segments": [],
                    }
                ],
            }
        ],
    }


def test_merge_variant_sweeps_preserves_candidates_and_hashes(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_sweep("first")), encoding="utf-8")
    second.write_text(json.dumps(_sweep("second")), encoding="utf-8")

    merged = merge_variant_sweeps(
        {"base": first, "vad": second},
        tmp_path / "merged.json",
        language_retry_policy={"ko": "always", "ro": "confidence"},
    )
    assert [item["variant_id"] for item in merged["variants"]] == ["base", "vad"]
    assert [item["variant_id"] for item in merged["spans"][0]["candidates"]] == [
        "base",
        "vad",
    ]
    assert all(len(item["sweep_sha256"]) == 64 for item in merged["variants"])
    assert merged["runtime_seconds"] == 2.0
    assert merged["giga_admission_emitted"] is False
    assert merged["language_retry_policy"] == {"ko": "always", "ro": "confidence"}


def test_merge_variant_sweeps_rejects_source_mismatch(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_sweep("first")), encoding="utf-8")
    second.write_text(json.dumps(_sweep("second", source="other")), encoding="utf-8")

    with pytest.raises(ValueError, match="source_sha256"):
        merge_variant_sweeps({"base": first, "vad": second}, tmp_path / "merged.json")


def test_merge_variant_sweeps_rejects_unknown_retry_policy(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_sweep("first")), encoding="utf-8")
    second.write_text(json.dumps(_sweep("second")), encoding="utf-8")

    with pytest.raises(ValueError, match="always, confidence, or global_confidence"):
        merge_variant_sweeps(
            {"base": first, "vad": second},
            tmp_path / "merged.json",
            language_retry_policy={"ko": "oracle"},
        )
