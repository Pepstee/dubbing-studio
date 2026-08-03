from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dubbing.transcription import (
    FasterWhisperTranscriptionBackend,
    TranscriptionError,
    TranscriptionOptions,
)


class _Model:
    init = None
    call = None

    def __init__(self, model, **kwargs):
        type(self).init = (model, kwargs)

    def transcribe(self, path, **kwargs):
        type(self).call = (path, kwargs)
        words = [
            SimpleNamespace(start=0.1, end=0.6, word=" Hello", probability=0.91),
            SimpleNamespace(start=0.6, end=1.4, word=" world", probability=0.82),
        ]
        segments = [
            SimpleNamespace(
                start=0.1,
                end=1.4,
                text=" Hello world ",
                avg_logprob=-0.2,
                compression_ratio=1.1,
                no_speech_prob=0.02,
                temperature=0.0,
                words=words,
            )
        ]
        return iter(segments), SimpleNamespace(language="en", duration=1.5)


class _FasterWhisper:
    WhisperModel = _Model


def test_missing_audio_is_actionable(tmp_path):
    backend = FasterWhisperTranscriptionBackend("fixture")
    with pytest.raises(TranscriptionError, match="audio input not found"):
        backend.transcribe(tmp_path / "missing.wav")


def test_missing_dependency_has_install_command(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    backend = FasterWhisperTranscriptionBackend("fixture")
    with patch(
        "dubbing.transcription.faster_whisper.importlib.import_module",
        side_effect=ImportError("missing"),
    ):
        with pytest.raises(TranscriptionError, match="transcription-faster"):
            backend.transcribe(audio)


def test_parses_words_and_forwards_gpu_configuration(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    backend = FasterWhisperTranscriptionBackend(
        "fixture",
        device="cuda",
        compute_type="float16",
        cpu_threads=4,
        num_workers=2,
    )
    with patch(
        "dubbing.transcription.faster_whisper.importlib.import_module",
        return_value=_FasterWhisper(),
    ):
        result = backend.transcribe(
            audio,
            TranscriptionOptions(language="en", initial_prompt="Names: Artiom."),
        )
    assert _Model.init == (
        "fixture",
        {
            "device": "cuda",
            "compute_type": "float16",
            "cpu_threads": 4,
            "num_workers": 2,
        },
    )
    assert _Model.call[1]["language"] == "en"
    assert _Model.call[1]["initial_prompt"] == "Names: Artiom."
    assert result.text == "Hello world"
    assert result.segments[0].words[1].end_ms == 1400
    assert result.duration_ms == 1500
    assert result.device == "cuda"
    assert result.confidence_available is True
    assert result.segments[0].diagnostics.compression_ratio == 1.1
    assert result.provenance["persistent_model_instance"]


def test_model_is_loaded_only_once(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    backend = FasterWhisperTranscriptionBackend("fixture")
    with patch(
        "dubbing.transcription.faster_whisper.importlib.import_module",
        return_value=_FasterWhisper(),
    ) as dependency:
        backend.transcribe(audio)
        backend.transcribe(audio)
    dependency.assert_called_once()


def test_identity_includes_temperature_and_offline_policy():
    backend = FasterWhisperTranscriptionBackend(
        "/models/asr",
        temperature=0.25,
        local_files_only=True,
    )

    assert "0.25" in backend.identity
    assert "local=True" in backend.identity


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device": "tpu"}, "device"),
        ({"compute_type": ""}, "compute_type"),
        ({"cpu_threads": -1}, "cpu_threads"),
        ({"num_workers": 0}, "num_workers"),
    ],
)
def test_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FasterWhisperTranscriptionBackend("fixture", **kwargs)
