from unittest.mock import Mock, patch

import pytest

from dubbing.diarization.models import DiarizationError
from dubbing.diarization.precision2 import Precision2Authorization, Precision2Backend


def test_precision2_is_disabled_without_per_recording_authority(tmp_path):
    audio = tmp_path / "private.wav"
    audio.write_bytes(b"private")
    transport = Mock()
    backend = Precision2Backend(
        Precision2Authorization("a" * 64, "operator-1"),
        transport=transport,
        receipt_dir=tmp_path / "receipts",
    )
    with pytest.raises(DiarizationError, match="cloud_allowed=false"):
        backend.diarize(audio)
    transport.assert_not_called()


def test_precision2_authorized_mock_records_receipt(tmp_path):
    audio = tmp_path / "span.wav"
    audio.write_bytes(b"span")
    transport = Mock(
        return_value={
            "turns": [{"start": 0, "end": 1, "speaker": "S1", "confidence": 0.9}],
            "request_receipt": "offline",
            "retention_mode": "mock-no-retention",
        }
    )
    backend = Precision2Backend(
        Precision2Authorization("a" * 64, "operator-1", cloud_allowed=True),
        transport=transport,
        receipt_dir=tmp_path / "receipts",
    )
    with patch.dict("os.environ", {"PYANNOTEAI_API_KEY": "local"}):
        result = backend.diarize(audio)
    assert result.turns[0].speaker == "S1"
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 1
