import hashlib
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
import threading
import wave
from unittest.mock import patch

import pytest

from dubbing.transcription.cloud import (
    CloudAdjudicator,
    CloudAuthorization,
    CloudSpan,
    ElevenLabsHTTPTransport,
    OfflineMockTransport,
    OpenAIHTTPTransport,
    TargetedCloudAdjudication,
    context_packet_interval,
    plan_context_packets,
    unresolved_intervals,
)
from dubbing.transcription.teacher import CloudTeacherPolicy, CloudTeacherRunner
from dubbing.transcription.teacher_cli import _macos_keychain_credential
from dubbing.transcription.compaction import CompactionPacket, CompactionSlice
from dubbing.transcription.speech_regions import SpeechRegion, SpeechRegionPlan
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


def _teacher_policy(**overrides):
    document = {
        "schema_version": "dubbing.cloud-teacher-policy.v1",
        "programme_id": "month-one-2026-08",
        "provider": "elevenlabs",
        "begins_at": "2026-08-20T00:00:00Z",
        "ends_at": "2026-09-19T00:00:00Z",
        "operator_authorization_id": "operator-month-one",
        "cloud_allowed": True,
        "training_corpus_allowed": True,
        "privacy": {
            "model_improvement_opt_out_attested": True,
            "retention_mode": "provider-storage; model-improvement-opt-out",
        },
        "limits": {
            "max_total_audio_seconds": 1_260_000,
            "max_estimated_cost_usd": 100,
            "estimated_price_per_hour_usd": 0.22,
            "chunk_seconds": 3600,
        },
        "compaction": {
            "enabled": True,
            "padding_ms": 1500,
            "merge_gap_ms": 2000,
            "separator_ms": 500,
            "maximum_slices_per_packet": 1,
            "preserve_diarization_context": True,
        },
        "training": {
            "maximum_agreement_wer": 0.05,
            "maximum_agreement_cer": 0.08,
        },
    }
    for key, value in overrides.items():
        document[key] = value
    return document


def test_teacher_policy_requires_privacy_opt_out_attestation():
    document = _teacher_policy()
    document["privacy"]["model_improvement_opt_out_attested"] = False
    with pytest.raises(ValueError, match="opt-out"):
        CloudTeacherPolicy.from_dict(document)


def test_macos_keychain_lookup_keeps_secret_out_of_command_arguments():
    completed = SimpleNamespace(returncode=0, stdout=b"keychain-only-secret\n")
    with patch.dict(
        "os.environ", {"ELEVENLABS_API_KEY": "environment-only-secret"}
    ), patch(
        "dubbing.transcription.teacher_cli.sys.platform", "darwin"
    ), patch(
        "dubbing.transcription.teacher_cli.subprocess.run", return_value=completed
    ) as run:
        credential = _macos_keychain_credential(
            "dubbing-studio-elevenlabs", "operator"
        )

    assert credential == "keychain-only-secret"
    command = run.call_args.args[0]
    assert command == [
        "/usr/bin/security",
        "find-generic-password",
        "-w",
        "-s",
        "dubbing-studio-elevenlabs",
        "-a",
        "operator",
    ]
    assert credential not in command
    assert run.call_args.kwargs["stderr"] is not None
    assert "ELEVENLABS_API_KEY" not in run.call_args.kwargs["env"]


def test_macos_keychain_lookup_fails_closed_without_leaking_provider_error():
    completed = SimpleNamespace(returncode=44, stdout=b"")
    with patch(
        "dubbing.transcription.teacher_cli.sys.platform", "darwin"
    ), patch(
        "dubbing.transcription.teacher_cli.subprocess.run", return_value=completed
    ), pytest.raises(ValueError, match="could not read") as error:
        _macos_keychain_credential("service", "account")

    assert "ElevenLabs API key" not in str(error.value)


def test_teacher_policy_forbids_discontinuous_diarization_packets():
    document = _teacher_policy()
    document["compaction"]["maximum_slices_per_packet"] = 64
    with pytest.raises(ValueError, match="exactly one source slice"):
        CloudTeacherPolicy.from_dict(document)


def test_programme_lock_serializes_budget_decisions(tmp_path):
    runner = CloudTeacherRunner(
        CloudTeacherPolicy.from_dict(_teacher_policy()),
        tmp_path / "teacher",
        tmp_path / "programme-usage.json",
        transport=OfflineMockTransport(),
        now=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    started = threading.Event()
    acquired = threading.Event()

    def contender():
        started.set()
        with runner._programme_lock():
            acquired.set()

    with runner._programme_lock():
        thread = threading.Thread(target=contender)
        thread.start()
        assert started.wait(1)
        assert not acquired.wait(0.05)
    assert acquired.wait(1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_uploading_reservation_counts_against_cost_cap(tmp_path):
    document = _teacher_policy()
    document["limits"]["max_estimated_cost_usd"] = 1.0
    runner = CloudTeacherRunner(
        CloudTeacherPolicy.from_dict(document),
        tmp_path / "teacher",
        tmp_path / "programme-usage.json",
        transport=OfflineMockTransport(),
        now=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    usage = {
        "chunks": {
            "possibly-paid": {
                "status": "uploading",
                "duration_ms": 1_000,
                "estimated_cost_usd": 0.99,
            }
        }
    }
    with pytest.raises(TranscriptionError, match="cost cap"):
        runner._assert_budget(usage, 3_600_000)


def test_elevenlabs_parser_preserves_word_evidence():
    candidate = ElevenLabsHTTPTransport._candidate(
        {
            "language_code": "rus",
            "language_probability": 0.97,
            "text": "Привет мир",
            "words": [
                {
                    "type": "word",
                    "text": "Привет",
                    "start": 0.1,
                    "end": 0.6,
                    "speaker_id": "speaker_0",
                    "logprob": -0.1,
                },
                {"type": "spacing", "text": " "},
                {
                    "type": "word",
                    "text": "мир",
                    "start": 0.7,
                    "end": 1.0,
                    "speaker_id": "speaker_0",
                    "logprob": -0.2,
                },
            ],
        }
    )
    assert candidate["language"] == "ru"
    assert [item["text"] for item in candidate["segments"]] == ["Привет", "мир"]
    assert candidate["segments"][0]["speaker"] == "speaker_0"
    assert candidate["segments"][0]["confidence"] > 0.8


def test_cloud_teacher_builds_only_agreement_silver_and_reuses_checkpoint(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source, duration_seconds=5)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    local = TranscriptionResult(
        segments=(TranscriptSegment(500, 3_500, "hello careful world", language="en"),),
        text="hello careful world",
        backend="faster-whisper",
        model="large-v3",
        device="cuda",
        language="en",
        duration_ms=5_000,
        confidence_available=False,
        source_sha256=digest,
    )
    transport = OfflineMockTransport(
        {
            "text": "hello careful world",
            "segments": [
                {
                    "start_ms": 500,
                    "end_ms": 1_100,
                    "text": "hello",
                    "speaker": "speaker_0",
                    "language": "en",
                    "confidence": 0.95,
                },
                {
                    "start_ms": 1_200,
                    "end_ms": 2_100,
                    "text": "careful",
                    "speaker": "speaker_0",
                    "language": "en",
                    "confidence": 0.94,
                },
                {
                    "start_ms": 2_200,
                    "end_ms": 3_500,
                    "text": "world",
                    "speaker": "speaker_0",
                    "language": "en",
                    "confidence": 0.96,
                },
                {
                    "start_ms": 4_000,
                    "end_ms": 4_500,
                    "text": "uncorroborated",
                    "speaker": "speaker_0",
                    "language": "en",
                    "confidence": 0.92,
                },
            ],
        }
    )

    def fake_extract(_source, _packet, destination):
        destination.write_bytes(b"lossless-flac-fixture")

    class Detector:
        @property
        def identity(self):
            return "test-sensitive-vad"

        def detect(self, _source):
            return SpeechRegionPlan(
                self.identity,
                5_000,
                (SpeechRegion(500, 3_500, "strict"),),
                (SpeechRegion(400, 3_600, "sensitive"),),
                (SpeechRegion(500, 3_500, "strict"),),
            )

    policy = CloudTeacherPolicy.from_dict(_teacher_policy())
    runner = CloudTeacherRunner(
        policy,
        tmp_path / "teacher",
        tmp_path / "programme-usage.json",
        transport=transport,
        speech_region_detector=Detector(),
        now=datetime(2026, 8, 21, tzinfo=timezone.utc),
        credential="keychain-only",
    )
    with patch.dict("os.environ", {}, clear=True), patch(
        "dubbing.transcription.teacher.extract_compacted_flac", side_effect=fake_extract
    ):
        report = runner.run(source, local)
        # Simulate a crash after the provider-bound checkpoint was committed but
        # before the programme ledger update became durable.
        (tmp_path / "programme-usage.json").unlink()
        replay = runner.run(source, local)
        assert "ELEVENLABS_API_KEY" not in os.environ

    assert len(transport.calls) == 1
    assert report["local_transcript_overwritten"] is False
    assert report["giga_admission_allowed"] is False
    assert report["silver_corpus"]["accepted_count"] == 1
    assert report["silver_corpus"]["excluded_count"] == 1
    assert replay["chunks"][0]["reused"] is True
    assert "keychain-only" not in json.dumps(report)
    for evidence in (tmp_path / "teacher").rglob("*.json*"):
        assert "keychain-only" not in evidence.read_text(encoding="utf-8")
    accepted = (tmp_path / "teacher" / "silver-corpus" / "accepted.jsonl").read_text()
    row = json.loads(accepted)
    assert row["label_class"] == "CONSENSUS_SILVER"
    assert row["split"] in {"train", "validation", "locked_test"}
    excluded = json.loads(
        (tmp_path / "teacher" / "silver-corpus" / "excluded.jsonl").read_text()
    )
    assert excluded["label_class"] == "CLOUD_ONLY_UNVERIFIED"
    assert excluded["target_text"] is None
    assert report["compaction"]["uploaded_ms"] == 5_000
    assert report["chunks"][0]["mapping_sha256"]
    repaired_usage = json.loads((tmp_path / "programme-usage.json").read_text())
    assert next(iter(repaired_usage["chunks"].values()))["status"] == "completed"


def test_unknown_upload_outcome_remains_reserved_and_cannot_be_retried(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source, duration_seconds=5)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    local = TranscriptionResult(
        segments=(TranscriptSegment(500, 3_500, "hello careful world", language="en"),),
        text="hello careful world",
        backend="faster-whisper",
        model="large-v3",
        device="cuda",
        language="en",
        duration_ms=5_000,
        confidence_available=False,
        source_sha256=digest,
    )

    class Detector:
        @property
        def identity(self):
            return "test-sensitive-vad"

        def detect(self, _source):
            speech = SpeechRegion(500, 3_500, "sensitive")
            return SpeechRegionPlan(self.identity, 5_000, (), (speech,), (speech,))

    class UnknownOutcomeTransport:
        def __init__(self):
            self.calls = 0

        def __call__(self, _request, _audio, _credential):
            self.calls += 1
            raise TranscriptionError("network outcome is unknown")

    transport = UnknownOutcomeTransport()
    runner = CloudTeacherRunner(
        CloudTeacherPolicy.from_dict(_teacher_policy()),
        tmp_path / "teacher",
        tmp_path / "programme-usage.json",
        transport=transport,
        speech_region_detector=Detector(),
        now=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )

    def fake_extract(_source, _packet, destination):
        destination.write_bytes(b"lossless-flac-fixture")

    with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "local-only"}), patch(
        "dubbing.transcription.teacher.extract_compacted_flac", side_effect=fake_extract
    ):
        with pytest.raises(TranscriptionError, match="network outcome is unknown"):
            runner.run(source, local)
        usage = json.loads((tmp_path / "programme-usage.json").read_text())
        reservation = next(iter(usage["chunks"].values()))
        assert reservation["status"] == "uploading"
        assert reservation["estimated_cost_usd"] > 0
        with pytest.raises(TranscriptionError, match="unknown outcome"):
            runner.run(source, local)

    assert transport.calls == 1


def test_cloud_teacher_restores_compacted_words_to_original_recording_time(tmp_path):
    source = tmp_path / "recording.wav"
    _silent_wave(source, duration_seconds=20)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    local = TranscriptionResult(
        segments=(TranscriptSegment(5_000, 8_000, "hello careful world", language="en"),),
        text="hello careful world",
        backend="faster-whisper",
        model="large-v3",
        device="cuda",
        language="en",
        duration_ms=20_000,
        confidence_available=False,
        source_sha256=digest,
    )
    transport = OfflineMockTransport(
        {
            "text": "hello careful world",
            "segments": [
                {"start_ms": 500, "end_ms": 1_000, "text": "hello", "language": "en"},
                {
                    "start_ms": 1_100,
                    "end_ms": 2_000,
                    "text": "careful",
                    "language": "en",
                },
                {"start_ms": 2_100, "end_ms": 3_000, "text": "world", "language": "en"},
            ],
        }
    )

    class Detector:
        @property
        def identity(self):
            return "test-sensitive-vad"

        def detect(self, _source):
            speech = SpeechRegion(5_000, 8_000, "sensitive")
            return SpeechRegionPlan(self.identity, 20_000, (), (speech,), (speech,))

    policy_document = _teacher_policy()
    policy_document["compaction"] = {
        "enabled": True,
        "padding_ms": 500,
        "merge_gap_ms": 0,
        "separator_ms": 500,
        "maximum_slices_per_packet": 1,
        "preserve_diarization_context": True,
    }
    runner = CloudTeacherRunner(
        CloudTeacherPolicy.from_dict(policy_document),
        tmp_path / "teacher",
        tmp_path / "programme-usage.json",
        transport=transport,
        speech_region_detector=Detector(),
        now=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )

    def fake_extract(_source, _packet, destination):
        destination.write_bytes(b"lossless-flac-fixture")

    with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "local-only"}), patch(
        "dubbing.transcription.teacher.extract_compacted_flac", side_effect=fake_extract
    ):
        report = runner.run(source, local)

    cloud = json.loads(
        (tmp_path / "teacher" / "cloud-transcript.json").read_text(encoding="utf-8")
    )
    assert [(item["start_ms"], item["end_ms"]) for item in cloud["segments"]] == [
        (5_000, 5_500),
        (5_600, 6_500),
        (6_600, 7_500),
    ]
    assert report["compaction"]["source_duration_ms"] == 20_000
    assert report["compaction"]["uploaded_ms"] == 4_000
    assert report["compaction"]["removed_ms"] == 16_000
    usage = json.loads((tmp_path / "programme-usage.json").read_text())
    assert next(iter(usage["chunks"].values()))["duration_ms"] == 4_000


def test_provider_speech_in_inserted_silence_fails_compact_packet_quality(tmp_path):
    packet = CompactionPacket(
        0,
        (
            CompactionSlice(1_000, 2_000, 0, 1_000),
            CompactionSlice(10_000, 11_000, 1_500, 2_500),
        ),
        2_500,
        500,
    )
    candidate = {
        "text": "real hallucinated",
        "segments": [
            {"start_ms": 100, "end_ms": 500, "text": "real"},
            {"start_ms": 1_100, "end_ms": 1_300, "text": "hallucinated"},
        ],
    }
    runner = CloudTeacherRunner(
        CloudTeacherPolicy.from_dict(_teacher_policy()),
        tmp_path / "teacher",
        tmp_path / "usage.json",
        transport=OfflineMockTransport(),
        now=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    segments, unmapped = runner._candidate_segments(candidate, packet)
    quality = runner._candidate_packet_quality(
        candidate, packet, unmapped_count=unmapped
    )
    assert [item.text for item in segments] == ["real"]
    assert unmapped == 1
    assert quality["status"] == "REPROCESS_REQUIRED"
    assert any(
        issue["code"] == "speech_in_compaction_separator"
        for issue in quality["issues"]
    )
