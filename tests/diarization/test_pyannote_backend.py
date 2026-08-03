from types import SimpleNamespace
from unittest.mock import Mock, patch

from dubbing.diarization.pyannote import PyannoteCommunityBackend


def test_pyannote_community_pipeline_loads_once_and_returns_anonymous_turns(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    model = tmp_path / "community-1"
    model.mkdir()

    annotation = Mock()
    annotation.itertracks.return_value = [
        (SimpleNamespace(start=0.1, end=1.2), None, "SPEAKER_00")
    ]
    pipeline = Mock(return_value=SimpleNamespace(speaker_diarization=annotation))
    module = SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=Mock(return_value=pipeline)))
    backend = PyannoteCommunityBackend(str(model), device="cpu")
    with patch("dubbing.diarization.pyannote.importlib.import_module", return_value=module):
        first = backend.diarize(audio)
        second = backend.diarize(audio)
    module.Pipeline.from_pretrained.assert_called_once_with(str(model), token=None)
    assert first.turns[0].speaker == "SPEAKER_00"
    assert first.turns[0].start_ms == 100
    assert second.to_dict() == first.to_dict()
