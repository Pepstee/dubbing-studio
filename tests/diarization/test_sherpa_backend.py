from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dubbing.diarization import (
    DiarizationError,
    SherpaOnnxDiarizationBackend,
    SpeakerConstraints,
    UnsupportedSpeakerConstraintError,
)


def _backend(tmp_path: Path, **kwargs) -> SherpaOnnxDiarizationBackend:
    segmentation = tmp_path / "segmentation.onnx"
    embedding = tmp_path / "embedding.onnx"
    segmentation.write_bytes(b"model")
    embedding.write_bytes(b"model")
    return SherpaOnnxDiarizationBackend(segmentation, embedding, **kwargs)


def test_missing_audio_is_actionable(tmp_path):
    backend = _backend(tmp_path)

    with pytest.raises(DiarizationError, match="audio input not found"):
        backend.diarize(tmp_path / "missing.wav")


def test_missing_model_is_actionable(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    backend = SherpaOnnxDiarizationBackend(
        tmp_path / "missing-seg.onnx",
        tmp_path / "missing-emb.onnx",
    )

    with pytest.raises(DiarizationError, match="segmentation model not found"):
        backend.diarize(audio)


def test_min_max_only_constraints_are_rejected_not_fabricated(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    backend = _backend(tmp_path)

    with pytest.raises(UnsupportedSpeakerConstraintError, match="cannot guarantee"):
        backend.diarize(audio, SpeakerConstraints(min_speakers=2, max_speakers=4))


def test_missing_optional_dependency_has_install_command(tmp_path):
    backend = _backend(tmp_path)

    with patch(
        "dubbing.diarization.sherpa.importlib.import_module",
        side_effect=ImportError("missing"),
    ):
        with pytest.raises(DiarizationError, match=r"dubbing-studio\[diarization\]"):
            backend._dependencies()


def test_cuda_request_rejects_cpu_only_wheel(tmp_path):
    backend = _backend(tmp_path, device="cuda")

    class _Sherpa:
        __file__ = str(tmp_path / "sherpa_onnx" / "__init__.py")

    with patch(
        "dubbing.diarization.sherpa.importlib.import_module",
        side_effect=[_Sherpa(), object()],
    ):
        with pytest.raises(DiarizationError, match="CPU-only"):
            backend._dependencies()
