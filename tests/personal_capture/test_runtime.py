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
