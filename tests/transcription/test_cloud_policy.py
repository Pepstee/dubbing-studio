from unittest.mock import patch

import pytest

from dubbing.transcription.cloud import (
    CloudAdjudicator,
    CloudAuthorization,
    CloudSpan,
    OfflineMockTransport,
)
from dubbing.transcription.models import TranscriptionError


def test_cloud_is_denied_by_default_without_transport_call(tmp_path):
    audio = tmp_path / "span.wav"
    audio.write_bytes(b"private")
    transport = OfflineMockTransport()
    adjudicator = CloudAdjudicator(tmp_path / "receipts", transport=transport)
    with pytest.raises(TranscriptionError, match="cloud_allowed=false"):
        adjudicator.adjudicate(
            CloudSpan(audio, 0, 20_000),
            CloudAuthorization("a" * 64, "deepgram", "operator-1"),
        )
    assert transport.calls == []


def test_authorized_failed_span_records_hash_bound_offline_receipt(tmp_path):
    audio = tmp_path / "span.wav"
    audio.write_bytes(b"private")
    transport = OfflineMockTransport()
    adjudicator = CloudAdjudicator(tmp_path / "receipts", transport=transport)
    authorization = CloudAuthorization(
        "a" * 64,
        "deepgram",
        "operator-1",
        cloud_allowed=True,
    )
    with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "locally-configured"}):
        candidate, receipt = adjudicator.adjudicate(
            CloudSpan(audio, 5_000, 30_000), authorization
        )
    assert candidate["text"] == "mock candidate"
    assert receipt.is_file()
    assert len(transport.calls) == 1


def test_whole_recording_and_wrong_span_lengths_are_rejected(tmp_path):
    audio = tmp_path / "span.wav"
    audio.write_bytes(b"private")
    adjudicator = CloudAdjudicator(tmp_path / "receipts", transport=OfflineMockTransport())
    authorization = CloudAuthorization(
        "a" * 64, "deepgram", "operator-1", cloud_allowed=True
    )
    with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "local"}), pytest.raises(
        TranscriptionError, match="20 and 60"
    ):
        adjudicator.adjudicate(CloudSpan(audio, 0, 61_000), authorization)
