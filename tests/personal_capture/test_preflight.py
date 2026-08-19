import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dubbing.apps.personal_capture.preflight import inspect_environment
from dubbing.apps.personal_capture.model_manifest import create_model_manifest
from tests.personal_capture.test_config import deployment_config


def test_preflight_prepares_paths_and_certifies_required_runtime(tmp_path):
    config = deployment_config(tmp_path)
    for key in ("segmentation_model", "embedding_model"):
        path = tmp_path / ("segmentation.onnx" if key == "segmentation_model" else "embedding.onnx")
        path.write_bytes(b"model")
        config["defaults"][key] = str(path)
    asr_model = tmp_path / "asr-model"
    asr_model.mkdir()
    for name in ("config.json", "model.bin", "tokenizer.json"):
        (asr_model / name).write_bytes(b"model")
    asr_retry_model = tmp_path / "asr-retry-model"
    asr_retry_model.mkdir()
    for name in ("config.json", "model.bin", "tokenizer.json"):
        (asr_retry_model / name).write_bytes(b"retry-model")
    translation_model = tmp_path / "translation-model"
    translation_model.mkdir()
    for name in ("config.json", "model.safetensors", "sentencepiece.bpe.model"):
        (translation_model / name).write_bytes(b"model")
    create_model_manifest(
        asr_directory=asr_model,
        asr_revision="a" * 40,
        asr_retry_directory=asr_retry_model,
        asr_retry_revision="c" * 40,
        translation_directory=translation_model,
        translation_revision="b" * 40,
        output=config["defaults"]["model_manifest"],
    )
    token = tmp_path / "capture.token"
    token.write_text("a" * 64, encoding="utf-8")
    os.chmod(token, 0o600)
    config["network"]["token_file"] = str(token)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    completed = SimpleNamespace(
        returncode=0,
        stdout="NVIDIA RTX 4060 Laptop GPU, 8188 MiB\n",
        stderr="",
    )
    asr_probe = MagicMock()
    translation_probe = MagicMock()
    diarization_probe = MagicMock()
    vad_probe = MagicMock()
    with patch(
        "dubbing.apps.personal_capture.preflight.ffmpeg_executable",
        return_value="/usr/bin/ffmpeg",
    ), patch(
        "dubbing.apps.personal_capture.preflight.shutil.which",
        side_effect=lambda name: f"/usr/bin/{name}",
    ), patch(
        "dubbing.apps.personal_capture.preflight.importlib.util.find_spec",
        return_value=object(),
    ), patch(
        "dubbing.apps.personal_capture.preflight.subprocess.run",
        return_value=completed,
    ), patch(
        "dubbing.apps.personal_capture.preflight.FasterWhisperTranscriptionBackend",
        return_value=asr_probe,
    ), patch(
        "dubbing.apps.personal_capture.preflight.FasterWhisperSileroSpeechRegionDetector",
        return_value=vad_probe,
    ), patch(
        "dubbing.apps.personal_capture.preflight.NLLBTranslationBackend",
        return_value=translation_probe,
    ), patch(
        "dubbing.apps.personal_capture.preflight.SherpaOnnxDiarizationBackend",
        return_value=diarization_probe,
    ):
        report = inspect_environment(config_path, prepare=True, load_models=True)

    assert report["ready"] is True
    assert report["model_load_certified"] is True
    assert all(item["status"] == "pass" for item in report["checks"])
    vad_probe.detect.assert_called_once()


def test_preflight_cannot_report_ready_without_loading_models(tmp_path):
    config = deployment_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with patch(
        "dubbing.apps.personal_capture.preflight.ffmpeg_executable",
        return_value="/usr/bin/ffmpeg",
    ), patch(
        "dubbing.apps.personal_capture.preflight.shutil.which",
        return_value="/usr/bin/tool",
    ), patch(
        "dubbing.apps.personal_capture.preflight.importlib.util.find_spec",
        return_value=object(),
    ), patch(
        "dubbing.apps.personal_capture.preflight.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="GPU", stderr=""),
    ):
        report = inspect_environment(config_path)

    assert report["ready"] is False
    assert report["model_load_certified"] is False
