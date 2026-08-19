import hashlib
import wave
from unittest.mock import patch

import pytest

from dubbing.transcription.cloud import (
    CloudAdjudicator,
    CloudAuthorization,
    CloudSpan,
    OfflineMockTransport,
    OpenAIHTTPTransport,
    TargetedCloudAdjudication,
    context_packet_interval,
    plan_context_packets,
    unresolved_intervals,
)
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)


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


def _silent_wave(path, duration_seconds=30):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 16_000 * duration_seconds)


def _local_result(source, *, digest=None):
    digest = digest or hashlib.sha256(source.read_bytes()).hexdigest()
    segments = (
        TranscriptSegment(0, 5_000, "This is clean speech."),
        TranscriptSegment(
            5_000,
            6_000,
            "[UNCERTAIN: INDEPENDENT TRANSCRIPTIONS DISAGREE]",
            uncertain=True,
        ),
        TranscriptSegment(6_000, 30_000, "The rest of this recording is clean speech."),
    )
    return TranscriptionResult(
        segments=segments,
        text=" ".join(item.text for item in segments),
        backend="faster-whisper",
        model="local",
        device="cuda",
        language=None,
        duration_ms=30_000,
        confidence_available=False,
        source_sha256=digest,
    )


def test_unresolved_intervals_and_context_packets_are_automatic(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source)
    result = _local_result(source)
    assert unresolved_intervals(result) == ((5_000, 6_000),)
    assert context_packet_interval((5_000, 6_000), duration_ms=30_000) == (0, 20_000)
    assert plan_context_packets(
        ((5_000, 6_000), (7_000, 8_000)), duration_ms=30_000
    ) == ((0, 20_000, ((5_000, 6_000), (7_000, 8_000))),)


def test_targeted_cloud_adjudication_uploads_only_context_and_preserves_receipts(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source)
    result = _local_result(source)
    transport = OfflineMockTransport(
        {
            "text": "Yes.",
            "segments": [
                {"start_ms": 5_000, "end_ms": 6_000, "text": "Yes.", "speaker": "A"}
            ],
        }
    )
    output = tmp_path / "cloud"
    adjudicator = CloudAdjudicator(output / "receipts", transport=transport)
    authorization = CloudAuthorization(
        result.source_sha256,
        "openai",
        "operator-project-4",
        cloud_allowed=True,
    )
    with patch.dict("os.environ", {"OPENAI_API_KEY": "local"}):
        amended, report = TargetedCloudAdjudication(adjudicator, output).run(
            source, result, authorization
        )

    assert len(transport.calls) == 1
    assert transport.calls[0]["request"]["provider"] == "openai"
    assert any(segment.text == "Yes." for segment in amended.segments)
    assert not any(segment.uncertain for segment in amended.segments)
    assert report["decisions"][0]["packet_start_ms"] == 0
    assert report["decisions"][0]["packet_end_ms"] == 20_000
    assert report["giga_admission_allowed"] is False
    assert (output / "result.json").is_file()
    assert (output / "adjudication-report.json").is_file()
    assert len(tuple((output / "receipts").glob("*.json"))) == 1


def test_targeted_cloud_adjudication_rejects_cross_source_before_upload(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source)
    result = _local_result(source, digest="b" * 64)
    transport = OfflineMockTransport()
    adjudicator = CloudAdjudicator(tmp_path / "receipts", transport=transport)
    authorization = CloudAuthorization(
        "a" * 64,
        "openai",
        "operator-project-4",
        cloud_allowed=True,
    )
    with pytest.raises(TranscriptionError, match="does not match source"):
        TargetedCloudAdjudication(adjudicator, tmp_path / "cloud").run(
            source, result, authorization
        )
    assert transport.calls == []


def test_empty_cloud_target_fails_closed_and_keeps_uncertainty(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source)
    result = _local_result(source)
    transport = OfflineMockTransport(
        {
            "text": "context only",
            "segments": [
                {"start_ms": 0, "end_ms": 1_000, "text": "context only"}
            ],
        }
    )
    adjudicator = CloudAdjudicator(tmp_path / "cloud" / "receipts", transport=transport)
    authorization = CloudAuthorization(
        result.source_sha256,
        "openai",
        "operator-project-4",
        cloud_allowed=True,
    )
    with patch.dict("os.environ", {"OPENAI_API_KEY": "local"}):
        amended, report = TargetedCloudAdjudication(
            adjudicator, tmp_path / "cloud"
        ).run(source, result, authorization)
    assert any(segment.uncertain for segment in amended.segments)
    assert report["admission_status"] == "FAIL_CLOSED"
    assert report["decisions"][0]["status"] == "REJECTED_EMPTY_TARGET"


def test_openai_response_parser_requires_timestamped_diarized_segments():
    candidate = OpenAIHTTPTransport._candidate(
        {
            "text": "Привет.",
            "segments": [
                {"start": 1.25, "end": 2.5, "text": "Привет.", "speaker": "A"}
            ],
        }
    )
    assert candidate == {
        "text": "Привет.",
        "segments": [
            {
                "start_ms": 1_250,
                "end_ms": 2_500,
                "text": "Привет.",
                "speaker": "A",
            }
        ],
    }
    with pytest.raises(TranscriptionError, match="timestamped segments"):
        OpenAIHTTPTransport._candidate({"text": "untimed"})


def test_contiguous_long_uncertainty_is_split_into_bounded_targets(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source)
    result = _local_result(source)
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(0, 60_000, "[UNCERTAIN: FIRST]", uncertain=True),
            TranscriptSegment(60_000, 70_000, "[UNCERTAIN: SECOND]", uncertain=True),
        ),
        text="[UNCERTAIN: FIRST] [UNCERTAIN: SECOND]",
        backend=result.backend,
        model=result.model,
        device=result.device,
        language=None,
        duration_ms=70_000,
        confidence_available=False,
        source_sha256=result.source_sha256,
    )
    assert unresolved_intervals(result) == ((0, 60_000), (60_000, 70_000))


def test_pathological_cloud_repetition_cannot_unlock_admission(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source)
    result = _local_result(source)
    loop = " ".join(["loops"] * 40)
    transport = OfflineMockTransport(
        {
            "text": loop,
            "segments": [
                {"start_ms": 5_000, "end_ms": 6_000, "text": loop}
            ],
        }
    )
    output = tmp_path / "cloud"
    adjudicator = CloudAdjudicator(output / "receipts", transport=transport)
    authorization = CloudAuthorization(
        result.source_sha256,
        "openai",
        "operator-project-4",
        cloud_allowed=True,
    )
    with patch.dict("os.environ", {"OPENAI_API_KEY": "local"}):
        _, report = TargetedCloudAdjudication(adjudicator, output).run(
            source, result, authorization
        )
    assert report["admission_status"] == "FAIL_CLOSED"
    assert report["quality"]["status"] != "PASS"
    assert any("repetition" in item["code"] for item in report["quality"]["issues"])
