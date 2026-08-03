import io
import json
from unittest.mock import patch

import pytest

from dubbing.transcription.models import TranscriptionOptions
from dubbing.transcription.whisperkit import WhisperKitTranscriptionBackend


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_whisperkit_local_server_parses_verbose_diagnostics(tmp_path):
    audio = tmp_path / "span.wav"
    audio.write_bytes(b"audio")
    payload = {
        "text": "Привет 안녕",
        "language": "ru",
        "duration": 2.0,
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Привет 안녕",
                "avg_logprob": -0.2,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.01,
                "temperature": 0.0,
                "language_probabilities": {"ru": 0.7, "ko": 0.3},
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "Привет", "probability": 0.9},
                    {"start": 1.0, "end": 2.0, "word": "안녕", "probability": 0.8},
                ],
            }
        ],
    }
    backend = WhisperKitTranscriptionBackend(model="small")
    with patch(
        "dubbing.transcription.whisperkit.urllib.request.urlopen",
        return_value=_Response(json.dumps(payload).encode()),
    ) as request:
        result = backend.transcribe(audio, TranscriptionOptions(language="ru"))
    assert result.text == "Привет 안녕"
    assert result.segments[0].diagnostics.compression_ratio == 1.1
    assert result.segments[0].diagnostics.language_probabilities["ru"] == 0.7
    assert result.provenance["boundary"] == "official-openai-compatible-local-server"
    assert b'name="language"' in request.call_args.args[0].data


@pytest.mark.parametrize("url", ["https://127.0.0.1:50060", "http://example.com:50060"])
def test_whisperkit_endpoint_must_be_loopback_http(url):
    with pytest.raises(ValueError, match="loopback"):
        WhisperKitTranscriptionBackend(model="small", endpoint=url)
