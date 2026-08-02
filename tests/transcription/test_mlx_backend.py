from __future__ import annotations

from unittest.mock import patch

import pytest

from dubbing.transcription import (
    MLXWhisperTranscriptionBackend,
    TranscriptionError,
    TranscriptionOptions,
)


class _MLX:
    @staticmethod
    def transcribe(path, **kwargs):
        return {
            "text": " Hello world ",
            "language": "en",
            "segments": [
                {
                    "start": 0.1,
                    "end": 1.4,
                    "text": " Hello world ",
                    "avg_logprob": -0.2,
                    "words": [
                        {
                            "start": 0.1,
                            "end": 0.6,
                            "word": " Hello",
                            "probability": 0.9,
                        },
                        {
                            "start": 0.6,
                            "end": 1.4,
                            "word": " world",
                            "probability": 0.8,
                        },
                    ],
                }
            ],
        }


def test_missing_audio_is_actionable(tmp_path):
    backend = MLXWhisperTranscriptionBackend("fixture")
    with pytest.raises(TranscriptionError, match="audio input not found"):
        backend.transcribe(tmp_path / "missing.wav")


def test_missing_dependency_has_install_command(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    backend = MLXWhisperTranscriptionBackend("fixture")
    with patch(
        "dubbing.transcription.mlx_whisper.importlib.import_module",
        side_effect=ImportError("missing"),
    ):
        with pytest.raises(TranscriptionError, match="transcription-mlx"):
            backend.transcribe(audio)


def test_parses_timestamped_words_and_forwards_options(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    backend = MLXWhisperTranscriptionBackend("fixture")
    with patch(
        "dubbing.transcription.mlx_whisper.importlib.import_module",
        return_value=_MLX(),
    ), patch.object(_MLX, "transcribe", wraps=_MLX.transcribe) as call:
        result = backend.transcribe(
            audio,
            TranscriptionOptions(
                language="en",
                initial_prompt="Names: Artiom.",
            ),
        )
    assert result.text == "Hello world"
    assert result.segments[0].start_ms == 100
    assert result.segments[0].words[1].end_ms == 1400
    assert result.confidence_available is True
    assert len(result.source_sha256 or "") == 64
    assert call.call_args.kwargs["language"] == "en"
    assert call.call_args.kwargs["initial_prompt"] == "Names: Artiom."
    assert call.call_args.kwargs["word_timestamps"] is True


def test_zero_length_model_fragments_are_skipped(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    class _Broken:
        @staticmethod
        def transcribe(path, **kwargs):
            return {
                "text": "",
                "segments": [
                    {"start": 1, "end": 1, "text": "bad"},
                    {"start": 0, "end": 1, "text": " "},
                ],
            }

    with patch(
        "dubbing.transcription.mlx_whisper.importlib.import_module",
        return_value=_Broken(),
    ):
        result = MLXWhisperTranscriptionBackend("fixture").transcribe(audio)
    assert result.segments == ()
    assert result.text == ""
