import json
from pathlib import Path

import pytest

from dubbing.apps.personal_capture.config import load_config, validate_config


def deployment_config(tmp_path: Path) -> dict:
    workspace = tmp_path / "life-logging" / "audio-processing"
    return {
        "schema_version": "dubbing.personal-capture-deployment.v1",
        "machine": "test",
        "landing_inbox": {
            "wsl_path": str(workspace / "recordings" / "inbox"),
            "minimum_file_age_seconds": 60,
        },
        "workspace": {"wsl_path": str(workspace)},
        "paths": {
            "packages": "outputs/packages",
            "processing": "processing",
            "state": "state",
        },
        "defaults": {
            "asr_backend": "faster-whisper",
            "resumable": True,
            "transcription_chunk_seconds": 1800,
            "transcription_overlap_seconds": 5,
            "diarization_chunk_seconds": 7200,
            "maximum_audio_seconds": 86400,
            "minimum_free_bytes": 0,
            "asr_model": str(tmp_path / "asr-model"),
            "asr_device": "cuda",
            "asr_compute_type": "float16",
            "translation_target": "en",
            "translation_model": str(tmp_path / "translation-model"),
            "translation_device": "cuda",
            "diarization_device": "cpu",
            "segmentation_model": str(tmp_path / "segmentation.onnx"),
            "embedding_model": str(tmp_path / "embedding.onnx"),
            "model_manifest": str(tmp_path / "model-manifest.json"),
            "offline_models_required": True,
            "review_required": True,
        },
        "service": {"poll_seconds": 15, "restart_policy": "on-failure"},
        "network": {
            "bind_host": "127.0.0.1",
            "port": 7433,
            "max_upload_bytes": 2_147_483_648,
            "token_file": str(tmp_path / "capture.token"),
            "secure_cookie": False,
            "exposure": "tailscale-serve-only",
        },
        "giga_outbox": {
            "path": str(workspace / "outbox" / "giga"),
            "automatic_interpreted_memory_promotion": False,
        },
        "boundaries": {
            "watcher_enabled": True,
            "automatic_giga_promotion": False,
            "network_upload_by_dubbing_studio": True,
            "source_deletion_by_dubbing_studio": False,
        },
    }


def test_canonical_config_is_valid(tmp_path):
    config = deployment_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    assert load_config(path) == config


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config["defaults"].update(resumable=False), "resumable"),
        (
            lambda config: config["paths"].update(processing="../outside"),
            "escapes",
        ),
        (
            lambda config: config["boundaries"].update(
                automatic_giga_promotion=True
            ),
            "promotion",
        ),
        (
            lambda config: config["network"].update(bind_host="0.0.0.0"),
            "loopback",
        ),
    ],
)
def test_unsafe_production_config_is_rejected(tmp_path, mutation, message):
    config = deployment_config(tmp_path)
    mutation(config)

    with pytest.raises(ValueError, match=message):
        validate_config(config)
