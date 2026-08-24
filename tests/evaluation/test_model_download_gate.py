import json
from pathlib import Path


def test_large_v3_download_receipt_matches_the_pinned_gate() -> None:
    repository = Path(__file__).resolve().parents[2]
    gate = json.loads(
        (
            repository
            / "benchmarks"
            / "fixtures"
            / "lesson-2026-08-01-193908"
            / "gigabyte-large-v3-model-gate.json"
        ).read_text(encoding="utf-8")
    )
    candidate = gate["candidate"]
    files = candidate["files"]

    assert sum(item["bytes"] for item in files) == candidate["total_download_bytes"]
    model = next(item for item in files if item["path"] == "model.bin")
    assert model["bytes"] == candidate["model_bin_bytes"]
    assert len(model["sha256"]) == 64
    latest = gate["hardware_observation"]["latest_preflight"]
    assert latest["estimated_free_headroom_mib"] == (
        latest["idle_free_vram_mib"]
        - gate["hardware_observation"]["published_faster_whisper_large_int8_vram_mib"]
    )
    assert gate["authorization"]["download_authorized"]
    assert gate["authorization"]["download_completed"]
    assert gate["download_receipt"]["verified_total_bytes"] == candidate[
        "total_download_bytes"
    ]
    assert gate["download_receipt"]["verified_model_bin_sha256"] == model["sha256"]
    assert gate["download_receipt"]["all_pinned_file_hashes_passed"]
    assert gate["download_receipt"]["int8_float16_runtime_preflight"]["model_loaded"]
    assert not gate["validation_evidence"]["automatic_all_language_targets_passed"]
    assert not gate["validation_evidence"]["oracle_all_language_targets_passed"]
    assert gate["long_form_evidence"]["quality_status"] == "REPROCESS_REQUIRED"
    assert gate["promotion_verdict"]["state"] == "REJECTED_AS_PRODUCTION_DEFAULT"
    assert not gate["giga_admission_emitted"]
