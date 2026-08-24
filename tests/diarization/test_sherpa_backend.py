from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
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


def test_embedding_audio_excludes_cross_speaker_overlap(tmp_path):
    backend = _backend(tmp_path)
    assert backend._subtract_overlaps(
        0,
        3_000,
        ((1_000, 2_000),),
    ) == ((0, 1_000), (2_000, 3_000))
    assert backend._subtract_overlaps(
        0,
        1_000,
        ((250, 750),),
    ) == ()


def test_identity_binds_exact_model_bytes_and_sizes(tmp_path):
    segmentation = tmp_path / "segmentation.onnx"
    embedding = tmp_path / "embedding.onnx"
    segmentation.write_bytes(b"segmentation-model")
    embedding.write_bytes(b"embedding-model")
    backend = SherpaOnnxDiarizationBackend(segmentation, embedding)

    integrity = backend.model_integrity

    assert integrity == {
        "segmentation": {
            "path": str(segmentation.resolve()),
            "size_bytes": 18,
            "sha256": hashlib.sha256(b"segmentation-model").hexdigest(),
        },
        "embedding": {
            "path": str(embedding.resolve()),
            "size_bytes": 15,
            "sha256": hashlib.sha256(b"embedding-model").hexdigest(),
        },
    }
    assert integrity["segmentation"]["sha256"] in backend.identity
    assert ":18:" in backend.identity
    first_identity = backend.identity
    embedding.write_bytes(b"different-model")
    assert backend.identity != first_identity


def test_result_provenance_contains_the_exact_model_integrity(tmp_path):
    backend = _backend(tmp_path)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    class _Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def validate(self):
            return True

    class _RawResult:
        def sort_by_start_time(self):
            return [SimpleNamespace(start=0.0, end=1.0, speaker=0)]

    sherpa = SimpleNamespace(
        OfflineSpeakerDiarizationConfig=_Config,
        OfflineSpeakerSegmentationModelConfig=_Config,
        OfflineSpeakerSegmentationPyannoteModelConfig=_Config,
        SpeakerEmbeddingExtractorConfig=_Config,
        FastClusteringConfig=_Config,
        OfflineSpeakerDiarization=lambda config: SimpleNamespace(
            process=lambda samples: _RawResult()
        ),
    )
    expected = backend.model_integrity
    with patch.object(backend, "_dependencies", return_value=(sherpa, object())), patch.object(
        backend, "_load_audio", return_value=object()
    ):
        result = backend.diarize(audio)

    assert result.provenance == {
        "backend_identity": backend._identity_from_integrity(expected),
        "model_integrity": expected,
    }
