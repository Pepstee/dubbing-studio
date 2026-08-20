from unittest.mock import patch

from dubbing.apps.personal_capture.runtime import build_service
from tests.personal_capture.test_config import deployment_config


def test_production_runtime_wires_distinct_manifest_bound_retry_model(tmp_path):
    config = deployment_config(tmp_path)
    manifest = {
        "models": {
            "asr": {"revision": "a" * 40},
            "asr_retry": {"revision": "c" * 40},
            "translation": {"revision": "b" * 40},
        }
    }

    with patch(
        "dubbing.apps.personal_capture.runtime.verify_model_manifest",
        return_value=manifest,
    ), patch(
        "dubbing.apps.personal_capture.runtime.LinguaLanguageDetector",
        return_value=object(),
    ):
        service = build_service(config)

    assert service.transcription_backend.model == config["defaults"]["asr_model"]
    assert (
        service.transcription_retry_backend.model
        == config["defaults"]["asr_retry_model"]
    )
    assert service.transcription_backend.identity != service.transcription_retry_backend.identity
    assert service.transcription_retry_backend.model_revision == "c" * 40
    assert service.transcription_retry_backend.compute_type == "int8_float16"
    assert service.speech_region_detector is None
    assert service.silence_verification_detector.identity.startswith(
        "faster-whisper-silero:strict=0.2:sensitive=0.1"
    )
    assert (
        service.targeted_retry_region_detector
        is service.silence_verification_detector
    )
    assert service.diarizer.cluster_threshold == 0.85
    assert service.diarization_embedding_backend is service.diarizer


def test_runtime_separates_pyannote_diarization_from_titanet_embeddings(tmp_path):
    config = deployment_config(tmp_path)
    config["defaults"].update(
        {
            "diarization_backend": "pyannote-community-1",
            "diarization_device": "cpu",
            "pyannote_model": str(tmp_path / "pyannote-community-1"),
        }
    )
    manifest = {
        "models": {
            "asr": {"revision": "a" * 40},
            "asr_retry": {"revision": "c" * 40},
            "translation": {"revision": "b" * 40},
        }
    }

    with patch(
        "dubbing.apps.personal_capture.runtime.verify_model_manifest",
        return_value=manifest,
    ), patch(
        "dubbing.apps.personal_capture.runtime.LinguaLanguageDetector",
        return_value=object(),
    ):
        service = build_service(config)

    assert service.diarizer.model == config["defaults"]["pyannote_model"]
    assert service.diarization_embedding_backend.cluster_threshold == 0.85


def test_runtime_wires_whisperkit_primary_and_mlx_retry_without_translation(tmp_path):
    config = deployment_config(tmp_path)
    config["defaults"].update(
        {
            "asr_backend": "whisperkit",
            "asr_model_name": "large-v3",
            "asr_executable": "/opt/homebrew/bin/whisperkit-cli",
            "asr_server_port": 50060,
            "asr_retry_backend": "mlx",
            "asr_retry_temperatures": [0.0],
            "translation_backend": "none",
            "vad_silence_verification_enabled": True,
            "vad_targeted_retry_region_detection_enabled": False,
        }
    )
    manifest = {
        "models": {
            "asr": {"revision": "a" * 40},
            "asr_retry": {"revision": "c" * 40},
        }
    }

    with patch(
        "dubbing.apps.personal_capture.runtime.verify_model_manifest",
        return_value=manifest,
    ), patch("dubbing.apps.personal_capture.runtime.WhisperKitServerProcess") as server:
        server.return_value.endpoint = "http://127.0.0.1:50060"
        service = build_service(config)

    assert service.transcription_backend.model == "large-v3"
    assert service.transcription_retry_backend.model == config["defaults"]["asr_retry_model"]
    assert service.transcription_retry_backend.temperature == (0.0,)
    assert service.translation_backend is None
    assert service.language_detector is None
    assert service.silence_verification_detector.identity.startswith(
        "faster-whisper-silero:strict=0.2:sensitive=0.1"
    )
    assert service.targeted_retry_region_detector is None
