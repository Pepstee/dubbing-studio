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
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.03,
                    "temperature": 1.0,
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
    assert result.segments[0].diagnostics.compression_ratio == 1.2
    assert result.segments[0].diagnostics.fallback_exhausted
    assert result.provenance["promotion_state"] == "experimental-fallback"
    assert len(result.source_sha256 or "") == 64
    assert call.call_args.kwargs["language"] == "en"
    assert call.call_args.kwargs["initial_prompt"] == "Names: Artiom."
    assert call.call_args.kwargs["word_timestamps"] is True
    assert call.call_args.kwargs["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert call.call_args.kwargs["condition_on_previous_text"] is False
    assert call.call_args.kwargs["hallucination_silence_threshold"] == 2.0


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


def test_normalizes_out_of_order_model_timestamps(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    class _Unordered:
        @staticmethod
        def transcribe(path, **kwargs):
            return {
                "text": "first second",
                "segments": [
                    {
                        "start": 2,
                        "end": 3,
                        "text": "second",
                        "words": [
                            {"start": 2.5, "end": 3, "word": "part two"},
                            {"start": 2, "end": 2.5, "word": "part one"},
                        ],
                    },
                    {"start": 0, "end": 1, "text": "first"},
                ],
            }

    with patch(
        "dubbing.transcription.mlx_whisper.importlib.import_module",
        return_value=_Unordered(),
    ):
        result = MLXWhisperTranscriptionBackend("fixture").transcribe(audio)

    assert [segment.text for segment in result.segments] == ["first", "second"]
    assert [word.text for word in result.segments[1].words] == ["part one", "part two"]
