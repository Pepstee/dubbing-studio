import json
from pathlib import Path


def test_large_v3_download_gate_is_complete_and_unauthorized() -> None:
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
    assert not gate["authorization"]["download_authorized"]
    assert not gate["authorization"]["download_started"]
    assert not gate["giga_admission_emitted"]
