from __future__ import annotations

import json
from pathlib import Path

_SCHEMA = "dubbing.personal-capture-deployment.v1"


def _absolute_path(document: dict, *keys: str) -> Path:
    value = document
    for key in keys:
        value = value[key]
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{'.'.join(keys)} must be an absolute path")
    return path.resolve()


def _workspace_child(workspace: Path, value: str, label: str) -> Path:
    path = (workspace / value).resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError(f"{label} escapes the workspace")
    return path


def validate_config(document: dict) -> dict:
    if document.get("schema_version") != _SCHEMA:
        raise ValueError(f"deployment config must use schema {_SCHEMA}")
    for key in (
        "landing_inbox",
        "workspace",
        "paths",
        "defaults",
        "service",
        "network",
        "giga_outbox",
        "boundaries",
    ):
        if not isinstance(document.get(key), dict):
            raise ValueError(f"deployment config is missing object {key}")

    workspace = _absolute_path(document, "workspace", "wsl_path")
    inbox = _absolute_path(document, "landing_inbox", "wsl_path")
    outbox = _absolute_path(document, "giga_outbox", "path")
    if workspace not in inbox.parents:
        raise ValueError("landing inbox must be inside the project workspace")
    if workspace not in outbox.parents:
        raise ValueError("GIGA outbox must be inside the project workspace")

    paths = document["paths"]
    for key in ("packages", "processing", "state"):
        value = paths.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"paths.{key} must be a non-empty relative path")
        if Path(value).is_absolute():
            raise ValueError(f"paths.{key} must be relative to the workspace")
        _workspace_child(workspace, value, f"paths.{key}")

    defaults = document["defaults"]
    for key in (
        "asr_model",
        "asr_device",
        "asr_compute_type",
        "asr_retry_model",
        "asr_retry_device",
        "asr_retry_compute_type",
        "translation_target",
        "translation_model",
        "translation_device",
        "diarization_device",
        "segmentation_model",
        "embedding_model",
        "model_manifest",
    ):
        if not isinstance(defaults.get(key), str) or not defaults[key]:
            raise ValueError(f"defaults.{key} must be a non-empty string")
    if defaults.get("asr_backend") != "faster-whisper":
        raise ValueError("production Personal Capture requires asr_backend=faster-whisper")
    if defaults.get("offline_models_required") is not True:
        raise ValueError("production Personal Capture requires offline_models_required=true")
    for key in ("asr_model", "asr_retry_model", "translation_model"):
        if not Path(defaults[key]).is_absolute():
            raise ValueError(f"defaults.{key} must be an absolute local model directory")
    if Path(defaults["asr_model"]).resolve() == Path(defaults["asr_retry_model"]).resolve():
        raise ValueError("defaults.asr_retry_model must be independent from defaults.asr_model")
    if not Path(defaults["model_manifest"]).is_absolute():
        raise ValueError("defaults.model_manifest must be an absolute path")
    if defaults.get("resumable") is not True:
        raise ValueError("production Personal Capture requires resumable=true")
    if defaults.get("transcription_strategy") != "adaptive":
        raise ValueError(
            "production Personal Capture requires transcription_strategy=adaptive"
        )
    if defaults.get("vad_backend") != "faster-whisper-silero":
        raise ValueError(
            "production Personal Capture requires vad_backend=faster-whisper-silero"
        )
    strict_vad = float(defaults.get("vad_strict_threshold", 0))
    sensitive_vad = float(defaults.get("vad_sensitive_threshold", 0))
    if not 0 < sensitive_vad < strict_vad < 1:
        raise ValueError("VAD thresholds must satisfy 0 < sensitive < strict < 1")
    for key in (
        "vad_minimum_speech_ms",
        "vad_minimum_silence_ms",
        "vad_speech_pad_ms",
    ):
        if int(defaults.get(key, -1)) < 0:
            raise ValueError(f"defaults.{key} cannot be negative")
    if not 0 < float(defaults.get("vad_maximum_region_seconds", 0)) <= 30:
        raise ValueError("vad_maximum_region_seconds must be between 0 and 30")
    if defaults.get("review_required") is not True:
        raise ValueError("production Personal Capture requires review_required=true")
    transcription_chunk = int(defaults.get("transcription_chunk_seconds", 0))
    minimum_chunk = int(defaults.get("transcription_minimum_chunk_seconds", 0))
    maximum_chunk = int(defaults.get("transcription_maximum_chunk_seconds", 0))
    if not 1 <= minimum_chunk <= transcription_chunk <= maximum_chunk <= 1800:
        raise ValueError(
            "adaptive chunk bounds must satisfy 1 <= minimum <= target <= maximum <= 1800"
        )
    transcription_overlap = float(defaults.get("transcription_overlap_seconds", -1))
    if not 0 <= transcription_overlap < minimum_chunk / 2:
        raise ValueError(
            "transcription_overlap_seconds must be non-negative and below half the minimum chunk"
        )
    minimum_silence = float(
        defaults.get("transcription_minimum_silence_seconds", 0)
    )
    if minimum_silence <= 0:
        raise ValueError("transcription_minimum_silence_seconds must be positive")
    diarization_chunk = int(defaults.get("diarization_chunk_seconds", 0))
    if not 60 <= diarization_chunk <= 10_800:
        raise ValueError("diarization_chunk_seconds must be between 60 and 10800")
    maximum_duration = int(defaults.get("maximum_audio_seconds", 0))
    if not 1 <= maximum_duration <= 86_400:
        raise ValueError("maximum_audio_seconds must be between 1 and 86400")
    if int(defaults.get("minimum_free_bytes", -1)) < 0:
        raise ValueError("minimum_free_bytes cannot be negative")

    if float(document["landing_inbox"].get("minimum_file_age_seconds", 0)) < 30:
        raise ValueError("minimum_file_age_seconds must be at least 30")
    if float(document["service"].get("poll_seconds", 0)) < 2:
        raise ValueError("service.poll_seconds must be at least 2")
    if document["service"].get("restart_policy") != "on-failure":
        raise ValueError("service.restart_policy must be on-failure")
    if int(document["network"].get("max_upload_bytes", 0)) <= 0:
        raise ValueError("network.max_upload_bytes must be positive")
    upload_timeout = int(document["network"].get("upload_timeout_seconds", 0))
    if not 120 <= upload_timeout <= 86_400:
        raise ValueError("network.upload_timeout_seconds must be between 120 and 86400")
    if document["network"].get("bind_host") not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("production review service must bind to loopback")
    port = int(document["network"].get("port", 0))
    if not 1 <= port <= 65_535:
        raise ValueError("network.port must be between 1 and 65535")
    if document["network"].get("exposure") != "tailscale-serve-only":
        raise ValueError("production review exposure must be tailscale-serve-only")
    token_file = Path(document["network"].get("token_file", ""))
    if not token_file.is_absolute():
        raise ValueError("network.token_file must be an absolute path")
    if (
        document["giga_outbox"].get("automatic_interpreted_memory_promotion")
        is not False
    ):
        raise ValueError("GIGA outbox cannot promote interpreted memory automatically")

    boundaries = document["boundaries"]
    if boundaries.get("watcher_enabled") is not True:
        raise ValueError("the production watcher must remain enabled")
    if boundaries.get("network_upload_by_dubbing_studio") is not True:
        raise ValueError("the configured review service must own its upload endpoint")
    if boundaries.get("automatic_giga_promotion") is not False:
        raise ValueError("automatic interpreted-memory promotion must remain disabled")
    if boundaries.get("source_deletion_by_dubbing_studio") is not False:
        raise ValueError("Dubbing Studio must not receive source-deletion authority")
    return document


def load_config(path: str | Path) -> dict:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read deployment config: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("deployment config must be a JSON object")
    return validate_config(document)
