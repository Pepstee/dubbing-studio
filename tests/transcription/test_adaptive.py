import json
import hashlib
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dubbing.transcription.adaptive import (
    AdaptiveChunk,
    AdaptiveChunkPlanner,
    AdaptiveLongFormCoordinator,
    AudioStream,
    MediaProbe,
    detect_silence_intervals,
    probe_media,
    reconcile_chunks,
)
from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.audio_candidates import AudioCandidate, AudioCandidateSet
from dubbing.transcription.models import (
    DecodeDiagnostics,
    TranscriptSegment,
    TranscriptWord,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)
from dubbing.transcription.speech_regions import SpeechRegion, SpeechRegionPlan


def _pcm_wave(path, samples, sample_rate=16000):
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(array("h", samples).tobytes())


def test_native_pcm_wave_probe_silence_and_extraction_without_ffmpeg(tmp_path):
    source = tmp_path / "source.wav"
    _pcm_wave(source, [10000, -10000] * 4000 + [0] * 16000 + [10000, -10000] * 4000)

    with (
        patch("dubbing.transcription.adaptive.shutil.which", return_value=None),
        patch("dubbing.transcription.adaptive.ffmpeg_executable", return_value=None),
    ):
        probe = probe_media(source)
        silences = detect_silence_intervals(source)
        output = tmp_path / "chunk.wav"
        AdaptiveLongFormCoordinator._extract(
            source,
            probe.audio_streams,
            AdaptiveChunk(0, 500, 1500, 500, 1500, "silence"),
            output,
        )

    assert probe.duration_ms == 2000
    assert probe.audio_streams[0].codec == "pcm_s16le"
    assert silences == ((500, 1500),)
    with wave.open(str(output), "rb") as extracted:
        assert extracted.getnframes() == 16000


def test_ffmpeg_silence_detection_maps_only_the_first_audio_stream(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    completed = SimpleNamespace(
        returncode=0,
        stderr="silence_start: 1.0\nsilence_end: 2.0 | silence_duration: 1.0\n",
    )

    with (
        patch("dubbing.transcription.adaptive.ffmpeg_executable", return_value="ffmpeg"),
        patch("dubbing.transcription.adaptive.subprocess.run", return_value=completed) as run,
    ):
        assert detect_silence_intervals(source) == ((1000, 2000),)

    command = run.call_args.args[0]
    assert command[command.index("-map") + 1] == "0:a:0"
    assert "-vn" in command
    assert "-sn" in command
    assert "-dn" in command


def test_chunk_extraction_merges_every_audio_stream_as_discrete_channels(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    output = tmp_path / "chunk.wav"
    streams = (
        AudioStream(1, "aac", 2, 48_000),
        AudioStream(2, "aac", 1, 44_100),
    )
    completed = SimpleNamespace(returncode=0, stderr="")

    with (
        patch("dubbing.transcription.adaptive.ffmpeg_executable", return_value="ffmpeg"),
        patch("dubbing.transcription.adaptive.subprocess.run", return_value=completed) as run,
    ):
        AdaptiveLongFormCoordinator._extract(
            source,
            streams,
            AdaptiveChunk(0, 0, 10_000, 0, 10_000, "end-of-media"),
            output,
        )

    command = run.call_args.args[0]
    assert command[command.index("-filter_complex") + 1] == ("[0:1][0:2]amerge=inputs=2[all_audio]")
    assert command[command.index("-ac") + 1] == "3"
    assert command[command.index("-ar") + 1] == "48000"


def test_reconciliation_resorts_words_collapsed_by_boundary_clamping():
    chunk = AdaptiveChunk(0, 10, 30, 0, 30, "end-of-media")
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(
                9,
                20,
                "two words",
                words=(
                    TranscriptWord(9, 20, " wide"),
                    TranscriptWord(10, 11, " short"),
                ),
            ),
        ),
        text="two words",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=30,
        confidence_available=False,
    )

    reconciled = reconcile_chunks([(chunk, result)])

    assert [(word.start_ms, word.end_ms) for word in reconciled[0].words] == [
        (10, 11),
        (10, 20),
    ]


class _RetryingBackend(TranscriptionBackend):
    def __init__(self):
        self.calls = 0

    @property
    def identity(self):
        return "fixture:persistent"

    def transcribe(self, audio, options=None):
        self.calls += 1
        text = " ".join(["Loops"] * 10) if self.calls == 1 else "clean ordinary phrase"
        return TranscriptionResult(
            segments=(TranscriptSegment(0, 10_000, text),),
            text=text,
            backend="fixture",
            model="persistent",
            device="test",
            language=options.language or "en",
            duration_ms=10_000,
            confidence_available=False,
        )


class _IndependentBackend(TranscriptionBackend):
    def __init__(self, text="clean ordinary phrase"):
        self.calls = 0
        self.text = text

    @property
    def identity(self):
        return "fixture:independent"

    def transcribe(self, audio, options=None):
        self.calls += 1
        return TranscriptionResult(
            segments=(TranscriptSegment(0, 10_000, self.text),),
            text=self.text,
            backend="fixture",
            model="independent",
            device="test",
            language=options.language or "en",
            duration_ms=10_000,
            confidence_available=False,
        )


class _SpeechDetector:
    identity = "fixture:sensitive-vad"

    def __init__(self, regions=()):
        self.calls = 0
        self.regions = regions

    def detect(self, audio):
        self.calls += 1
        duration_ms = 65_407
        return SpeechRegionPlan(
            self.identity,
            duration_ms,
            tuple(item for item in self.regions if item.source_pass == "strict"),
            tuple(item for item in self.regions if item.source_pass == "sensitive"),
            self.regions,
        )


def _lesson_silence_chunk():
    return AdaptiveChunk(
        index=15,
        start_ms=962_400,
        end_ms=1_023_807,
        extract_start_ms=960_400,
        extract_end_ms=1_025_807,
        boundary_reason="long-silence",
    )


def _single_chunk_planner(chunk):
    return SimpleNamespace(
        plan=lambda duration_ms, silence_centres, silence_intervals=(): (chunk,),
        to_dict=lambda: {"fixture": "single-chunk"},
    )


def test_dual_evidence_silence_skips_asr_and_replays_empty_checkpoint(tmp_path):
    source = tmp_path / "lesson.mov"
    source.write_bytes(b"source")
    chunk = _lesson_silence_chunk()
    backend = _IndependentBackend()
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        planner=_single_chunk_planner(chunk),
        speech_region_detector=detector,
    )

    def extract(source, streams, selected_chunk, output):
        output.write_bytes(b"confirmed silence")

    with (
        patch(
            "dubbing.transcription.adaptive.probe_media",
            return_value=MediaProbe(
                1_034_499,
                6,
                (AudioStream(0, "pcm", 1, 16_000),),
            ),
        ),
        patch(
            "dubbing.transcription.adaptive.detect_silence_intervals",
            return_value=((955_118, 1_034_499),),
        ),
        patch.object(coordinator, "_extract", side_effect=extract),
    ):
        result, _ = coordinator.run(source)
        replay, _ = coordinator.run(source)

    assert result.segments == ()
    assert result.text == ""
    assert replay == result
    assert backend.calls == 0
    assert detector.calls == 1
    manifest = json.loads((tmp_path / "job" / "manifest.json").read_text())
    assert manifest["silence_verification_detector_identity"] == detector.identity
    assert manifest["targeted_retry_region_detector_identity"] == detector.identity
    chunk_document = json.loads((tmp_path / "job" / "chunks" / "000015.json").read_text())
    assert chunk_document["segments"] == []
    assert chunk_document["provenance"]["classification"] == "confirmed_silence"
    receipt = json.loads((tmp_path / "job" / "receipts" / "000015.json").read_text())
    assert receipt["selected_attempt"] == 0
    assert receipt["targeted_retry_exhausted"] is False
    assert receipt["attempts"][0]["status"] == "CONFIRMED_SILENCE"
    assert receipt["attempts"][0]["energy_silence_coverage_ms"] == 65_407
    assert not any(item["kind"] == "targeted-span-redecode" for item in receipt["attempts"])


def test_silence_short_of_full_extracted_range_preserves_asr(tmp_path):
    source = tmp_path / "lesson.mov"
    source.write_bytes(b"source")
    chunk = _lesson_silence_chunk()
    backend = _IndependentBackend()
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        planner=_single_chunk_planner(chunk),
        speech_region_detector=detector,
    )

    with (
        patch(
            "dubbing.transcription.adaptive.probe_media",
            return_value=MediaProbe(
                1_034_499,
                6,
                (AudioStream(0, "pcm", 1, 16_000),),
            ),
        ),
        patch(
            "dubbing.transcription.adaptive.detect_silence_intervals",
            return_value=((955_118, chunk.extract_end_ms - 1),),
        ),
        patch.object(
            coordinator,
            "_extract",
            side_effect=lambda source, streams, selected_chunk, output: output.write_bytes(
                b"not fully covered"
            ),
        ),
    ):
        coordinator.run(source)

    assert backend.calls == 1
    assert detector.calls == 0


def test_semantic_vad_speech_overrides_full_energy_silence(tmp_path):
    chunk = _lesson_silence_chunk()
    backend = _IndependentBackend()
    detector = _SpeechDetector((SpeechRegion(20_000, 21_000, "sensitive"),))
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        speech_region_detector=detector,
    )
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"quiet speech")

    result, receipt = coordinator._decode_chunk(
        audio,
        chunk,
        TranscriptionOptions(),
        energy_silence_coverage_ms=65_407,
    )

    assert result.text == "clean ordinary phrase"
    assert backend.calls == 1
    assert detector.calls == 1
    assert receipt["attempts"][0]["status"] == "SEMANTIC_SPEECH_DETECTED"


def test_energy_silence_without_configured_detector_preserves_asr(tmp_path):
    chunk = _lesson_silence_chunk()
    backend = _IndependentBackend()
    targeted_detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        silence_verification_detector=None,
        targeted_retry_region_detector=targeted_detector,
    )
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"unverified silence")

    result, _ = coordinator._decode_chunk(
        audio,
        chunk,
        TranscriptionOptions(),
        energy_silence_coverage_ms=65_407,
    )

    assert result.text == "clean ordinary phrase"
    assert backend.calls == 1
    assert targeted_detector.calls == 0


class _ConstantBackend(TranscriptionBackend):
    def __init__(self, identity):
        self._identity = identity

    @property
    def identity(self):
        return self._identity

    def transcribe(self, audio, options=None):
        return TranscriptionResult(
            segments=(TranscriptSegment(0, 500, "agreed speech"),),
            text="agreed speech",
            backend="fixture",
            model=self._identity,
            device="test",
            language=options.language or "en",
            duration_ms=500,
            confidence_available=False,
        )


class _AudioCandidateBackend(TranscriptionBackend):
    def __init__(self, identity, texts):
        self._identity = identity
        self.texts = texts

    @property
    def identity(self):
        return self._identity

    def transcribe(self, audio, options=None):
        text = self.texts[Path(audio).name]
        return TranscriptionResult(
            segments=(TranscriptSegment(0, 20_000, text),),
            text=text,
            backend="fixture",
            model=self.identity,
            device="test",
            language=options.language or "en",
            duration_ms=20_000,
            confidence_available=False,
        )


def _audio_candidate_set(tmp_path):
    candidates = []
    for index, identifier in enumerate(("raw", "channel-0", "channel-1")):
        path = tmp_path / f"{identifier}.wav"
        path.write_bytes(identifier.encode())
        candidates.append(
            AudioCandidate(
                identifier,
                path,
                "0" * 64,
                f"{index + 1:064x}",
                {"kind": identifier},
            )
        )
    return AudioCandidateSet(tuple(candidates), 2)


def test_planner_chooses_silence_near_target_and_bounds_chunks():
    planner = AdaptiveChunkPlanner(
        target_seconds=120, minimum_seconds=60, maximum_seconds=180, overlap_seconds=2
    )
    chunks = planner.plan(400_000, (110_000, 250_000))
    assert [item.end_ms for item in chunks] == [110_000, 250_000, 400_000]
    assert all(item.end_ms - item.start_ms <= 180_000 for item in chunks)
    assert chunks[1].extract_start_ms == 108_000


def test_planner_supports_utterance_scale_language_boundaries():
    planner = AdaptiveChunkPlanner(
        target_seconds=8, minimum_seconds=2, maximum_seconds=16, overlap_seconds=0.25
    )
    chunks = planner.plan(42_000, (7_000, 20_000, 28_000, 40_000))

    assert [item.end_ms for item in chunks] == [7_000, 20_000, 28_000, 40_000, 42_000]
    assert all(item.boundary_reason == "silence" for item in chunks[:-1])


def test_planner_splits_multilingual_tail_near_target():
    planner = AdaptiveChunkPlanner(
        target_seconds=8, minimum_seconds=2, maximum_seconds=30, overlap_seconds=0.25
    )
    chunks = planner.plan(23_000, (10_000, 17_000))

    assert [item.end_ms for item in chunks] == [10_000, 17_000, 23_000]
    assert chunks[0].boundary_reason == "silence"
    assert chunks[1].boundary_reason == "silence"


def test_planner_prefers_long_inter_utterance_silence_over_nearby_short_pause():
    planner = AdaptiveChunkPlanner(
        target_seconds=8, minimum_seconds=2, maximum_seconds=30, overlap_seconds=0.25
    )
    intervals = (
        (20_796, 21_358),
        (22_600, 25_885),
        (30_744, 32_571),
    )
    centres = tuple(round((start + end) / 2) for start, end in intervals)

    chunks = planner.plan(
        42_410,
        centres,
        silence_intervals=intervals,
    )

    assert chunks[0].end_ms == 24_242
    assert chunks[0].extract_end_ms == 24_492
    assert chunks[1].extract_start_ms == 23_992


def test_planner_does_not_cross_a_second_long_silence():
    planner = AdaptiveChunkPlanner(
        target_seconds=8, minimum_seconds=2, maximum_seconds=30, overlap_seconds=0.25
    )
    intervals = (
        (6_000, 8_000),
        (13_000, 16_000),
        (22_000, 24_000),
    )
    centres = tuple(round((start + end) / 2) for start, end in intervals)

    chunks = planner.plan(
        30_000,
        centres,
        silence_intervals=intervals,
    )

    assert [chunk.end_ms for chunk in chunks] == [7_000, 14_500, 23_000, 30_000]
    assert all(chunk.boundary_reason == "long-silence" for chunk in chunks[:-1])


def test_planner_treats_a_700ms_turn_pause_as_a_hard_boundary():
    planner = AdaptiveChunkPlanner(
        target_seconds=8, minimum_seconds=2, maximum_seconds=30, overlap_seconds=0.25
    )
    intervals = ((7_500, 8_300), (14_000, 14_600))
    centres = tuple(round((start + end) / 2) for start, end in intervals)

    chunks = planner.plan(
        20_000,
        centres,
        silence_intervals=intervals,
    )

    assert chunks[0].end_ms == 7_900
    assert chunks[0].boundary_reason == "long-silence"


def test_nondefault_silence_policy_is_checkpoint_bound(tmp_path):
    coordinator = AdaptiveLongFormCoordinator(
        _RetryingBackend(), tmp_path, minimum_silence_seconds=0.5
    )
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    probe = MediaProbe(10_000, 6, (AudioStream(0, "pcm", 1, 16_000),))
    chunks = (AdaptiveChunk(0, 0, 10_000, 0, 10_000, "end-of-media"),)

    manifest = coordinator._manifest(source, "0" * 64, probe, chunks)

    assert manifest["silence_detection"]["minimum_silence_seconds"] == 0.5


def test_language_retry_policy_is_checkpoint_bound(tmp_path):
    coordinator = AdaptiveLongFormCoordinator(
        _RetryingBackend(), tmp_path, language_retry_policy={"ko": "always"}
    )
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    probe = MediaProbe(10_000, 6, (AudioStream(0, "pcm", 1, 16_000),))
    chunks = (AdaptiveChunk(0, 0, 10_000, 0, 10_000, "end-of-media"),)

    manifest = coordinator._manifest(source, "0" * 64, probe, chunks)

    assert manifest["language_retry_policy"] == {"ko": "always"}


def test_targeted_retry_spans_are_vad_bounded_between_20_and_60_seconds():
    spans = AdaptiveLongFormCoordinator._bounded_retry_spans(
        [(10_000, 160_000)],
        duration_ms=200_000,
        silence_centres=(55_000, 100_000, 145_000),
    )
    assert spans == (
        (8_000, 55_000),
        (55_000, 100_000),
        (100_000, 142_000),
        (142_000, 162_000),
    )
    assert all(20_000 <= end - start <= 60_000 for start, end in spans)


def test_quality_issue_timestamps_drive_targeted_retry_not_the_whole_recording():
    text = " ".join(["I don't know what to do"] * 10)
    result = TranscriptionResult(
        (
            TranscriptSegment(0, 1000, "ordinary"),
            TranscriptSegment(120_000, 130_000, text),
            TranscriptSegment(240_000, 241_000, "ordinary again"),
        ),
        f"ordinary {text} ordinary again",
        "fixture",
        "fixture",
        "test",
        "en",
        300_000,
        False,
    )
    from dubbing.transcription.quality import evaluate_transcript_quality

    quality = evaluate_transcript_quality(result, expected_duration_ms=300_000)

    assert AdaptiveLongFormCoordinator._critical_failure_intervals(result, quality) == [
        (120_000, 130_000)
    ]


def test_v3_checkpoint_manifest_migrates_only_when_other_contract_fields_match(tmp_path):
    backend = _RetryingBackend()
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    probe = MediaProbe(60_000, 6, (AudioStream(0, "pcm", 2, 48_000),))
    chunks = (AdaptiveChunk(0, 0, 60_000, 0, 60_000, "end-of-media"),)
    expected = coordinator._manifest(source, "0" * 64, probe, chunks)
    existing = dict(expected)
    existing["coordinator_version"] = "adaptive-long-form-v3"
    existing.pop("targeted_retry")
    manifest = tmp_path / "job" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(existing), encoding="utf-8")

    coordinator._admit_manifest(expected)

    assert json.loads(manifest.read_text(encoding="utf-8")) == expected


def test_detector_role_change_invalidates_adaptive_checkpoint_manifest(tmp_path):
    detector = _SpeechDetector()
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    probe = MediaProbe(60_000, 6, (AudioStream(0, "pcm", 2, 48_000),))
    chunks = (AdaptiveChunk(0, 0, 60_000, 0, 60_000, "end-of-media"),)
    both_roles = AdaptiveLongFormCoordinator(
        _RetryingBackend(),
        tmp_path / "job",
        speech_region_detector=detector,
    )
    both_roles._admit_manifest(both_roles._manifest(source, "0" * 64, probe, chunks))
    silence_only = AdaptiveLongFormCoordinator(
        _RetryingBackend(),
        tmp_path / "job",
        silence_verification_detector=detector,
        targeted_retry_region_detector=None,
    )

    with pytest.raises(
        TranscriptionError,
        match="adaptive checkpoint does not match source or configuration",
    ):
        silence_only._admit_manifest(silence_only._manifest(source, "0" * 64, probe, chunks))


def test_overlap_reconciliation_drops_duplicate_boundary_segment():
    chunks = [
        AdaptiveChunk(0, 0, 10_000, 0, 12_000, "silence"),
        AdaptiveChunk(1, 10_000, 20_000, 8_000, 20_000, "end-of-media"),
    ]
    first = TranscriptionResult(
        (TranscriptSegment(9_000, 10_500, "same boundary"),),
        "same boundary",
        "x",
        "x",
        "x",
        "en",
        12_000,
        False,
    )
    second = TranscriptionResult(
        (TranscriptSegment(1_000, 2_500, "same boundary"),),
        "same boundary",
        "x",
        "x",
        "x",
        "en",
        12_000,
        False,
    )
    assert len(reconcile_chunks(list(zip(chunks, (first, second))))) == 1


def test_overlap_context_is_clipped_to_non_overlapping_logical_chunks():
    chunks = [
        AdaptiveChunk(0, 0, 10_000, 0, 12_000, "silence"),
        AdaptiveChunk(1, 10_000, 20_000, 8_000, 20_000, "end-of-media"),
    ]
    first = TranscriptionResult(
        (TranscriptSegment(8_500, 10_500, "before boundary"),),
        "before boundary",
        "x",
        "x",
        "x",
        "en",
        12_000,
        False,
    )
    second = TranscriptionResult(
        (TranscriptSegment(1_500, 3_000, "after boundary"),),
        "after boundary",
        "x",
        "x",
        "x",
        "en",
        12_000,
        False,
    )

    segments = reconcile_chunks(list(zip(chunks, (first, second))))

    assert [(item.start_ms, item.end_ms) for item in segments] == [
        (8_500, 10_000),
        (10_000, 11_000),
    ]


def test_coordinator_retries_only_failed_span_and_checkpoints(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    backend = _RetryingBackend()
    retry_backend = _IndependentBackend()
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        retry_backend=retry_backend,
        planner=AdaptiveChunkPlanner(
            target_seconds=60, minimum_seconds=60, maximum_seconds=60, overlap_seconds=0
        ),
    )

    def extract(source, stream, chunk, output):
        output.write_bytes(b"span")

    def extract_retry(source, segment, destination):
        destination.write_bytes(b"retry span")

    with (
        patch(
            "dubbing.transcription.adaptive.probe_media",
            return_value=MediaProbe(60_000, 6, (AudioStream(0, "pcm", 2, 48_000),)),
        ),
        patch("dubbing.transcription.adaptive.detect_silence_centres", return_value=()),
        patch.object(coordinator, "_extract", side_effect=extract),
        patch.object(coordinator, "_extract_language_span", side_effect=extract_retry),
    ):
        result, quality = coordinator.run(source)
        coordinator.run(source)
    assert backend.calls == 6
    assert retry_backend.calls == 5
    assert result.text == "clean ordinary phrase"
    assert quality["status"] == "PASS"
    assert (tmp_path / "job" / "chunks" / "000000.json").is_file()
    receipt = json.loads((tmp_path / "job" / "receipts" / "000000.json").read_text())
    targeted = [
        attempt
        for attempt in receipt["attempts"]
        if attempt.get("kind") == "targeted-span-redecode"
    ]
    assert len(targeted) == 10
    assert targeted[0]["source_end_ms"] == 20_000
    adjudications = [
        attempt
        for attempt in receipt["attempts"]
        if attempt.get("kind") == "targeted-span-adjudication"
    ]
    assert len(adjudications) == 1
    assert adjudications[0]["status"] == "CONSENSUS_PASS"
    assert adjudications[0]["selected_audio_candidate"] == "raw"
    assert adjudications[0]["raw_consensus_available"] is True
    assert adjudications[0]["divergent_audio_candidates"] is False
    assert adjudications[0]["selected_uncertain"] is False


def test_targeted_retry_without_independent_backend_fails_closed(tmp_path):
    backend = _RetryingBackend()
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")
    attempts = []
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (0, 20_000),
            "Loops Loops Loops",
            TranscriptionOptions(),
            attempts,
        )

    assert replacement is None
    assert backend.calls == 0
    assert attempts[-1]["status"] == "NO_INDEPENDENT_BACKEND_CONFIGURED"


def test_audio_candidate_can_rescue_raw_only_with_independent_consensus(tmp_path):
    primary = _AudioCandidateBackend(
        "fixture:primary",
        {
            "raw.wav": "raw primary transcript",
            "channel-0.wav": "rescued channel speech",
            "channel-1.wav": "Loops " * 10,
        },
    )
    independent = _AudioCandidateBackend(
        "fixture:independent",
        {
            "raw.wav": "unrelated independent words",
            "channel-0.wav": "rescued channel speech",
            "channel-1.wav": "Loops " * 10,
        },
    )
    coordinator = AdaptiveLongFormCoordinator(
        primary,
        tmp_path / "job",
        retry_backend=independent,
    )
    attempts = []
    with (
        patch.object(
            coordinator,
            "_extract_language_span",
            side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
        ),
        patch(
            "dubbing.transcription.adaptive.build_audio_candidates",
            return_value=_audio_candidate_set(tmp_path),
        ),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (0, 20_000),
            "failed original words",
            TranscriptionOptions(),
            attempts,
        )

    assert " ".join(item.text for item in replacement) == "rescued channel speech"
    adjudication = attempts[-1]
    assert adjudication["status"] == "CONSENSUS_PASS"
    assert adjudication["raw_consensus_available"] is False
    assert adjudication["selected_audio_candidate"] == "channel-0"


def test_divergent_channel_consensuses_fail_closed(tmp_path):
    texts = {
        "raw.wav": "raw primary transcript",
        "channel-0.wav": "first rescued phrase",
        "channel-1.wav": "second alternate words",
    }
    primary = _AudioCandidateBackend("fixture:primary", texts)
    independent = _AudioCandidateBackend(
        "fixture:independent",
        {**texts, "raw.wav": "unrelated independent words"},
    )
    coordinator = AdaptiveLongFormCoordinator(
        primary,
        tmp_path / "job",
        retry_backend=independent,
    )
    attempts = []
    with (
        patch.object(
            coordinator,
            "_extract_language_span",
            side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
        ),
        patch(
            "dubbing.transcription.adaptive.build_audio_candidates",
            return_value=_audio_candidate_set(tmp_path),
        ),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (0, 20_000),
            "failed original words",
            TranscriptionOptions(),
            attempts,
        )

    assert replacement[0].uncertain is True
    assert replacement[0].text == "[UNCERTAIN: INDEPENDENT TRANSCRIPTIONS DISAGREE]"
    adjudication = attempts[-1]
    assert adjudication["status"] == "AUDIO_CANDIDATE_DISAGREEMENT"
    assert adjudication["divergent_audio_candidates"] is True


def test_raw_consensus_wins_even_when_processed_candidates_disagree(tmp_path):
    primary_texts = {
        "raw.wav": "trustworthy raw recording transcript",
        "channel-0.wav": "first processed interpretation",
        "channel-1.wav": "second processed alternative",
    }
    primary = _AudioCandidateBackend("fixture:primary", primary_texts)
    independent = _AudioCandidateBackend("fixture:independent", primary_texts)
    coordinator = AdaptiveLongFormCoordinator(
        primary,
        tmp_path / "job",
        retry_backend=independent,
    )
    attempts = []
    with (
        patch.object(
            coordinator,
            "_extract_language_span",
            side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
        ),
        patch(
            "dubbing.transcription.adaptive.build_audio_candidates",
            return_value=_audio_candidate_set(tmp_path),
        ),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (0, 20_000),
            "failed original words",
            TranscriptionOptions(),
            attempts,
        )

    assert " ".join(item.text for item in replacement) == primary_texts["raw.wav"]
    adjudication = attempts[-1]
    assert adjudication["status"] == "CONSENSUS_PASS"
    assert adjudication["selected_audio_candidate"] == "raw"
    assert adjudication["raw_consensus_available"] is True
    assert adjudication["divergent_audio_candidates"] is False


def test_independent_disagreement_is_preserved_as_uncertain(tmp_path):
    primary = _RetryingBackend()
    primary.calls = 1
    coordinator = AdaptiveLongFormCoordinator(
        primary,
        tmp_path / "job",
        retry_backend=_IndependentBackend("entirely different testimony today"),
    )
    attempts = []
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (0, 20_000),
            "Loops Loops Loops",
            TranscriptionOptions(),
            attempts,
        )

    assert replacement is not None
    assert all(segment.uncertain for segment in replacement)
    assert [segment.text for segment in replacement] == [
        "[UNCERTAIN: INDEPENDENT TRANSCRIPTIONS DISAGREE]"
    ]
    assert attempts[-1]["status"] == "INDEPENDENT_DISAGREEMENT"
    assert attempts[-1]["agreement"] == 0.0


def test_exact_low_confidence_agreement_cannot_become_clean_text(tmp_path):
    class _LowConfidenceBackend(_ConstantBackend):
        def transcribe(self, audio, options=None):
            return TranscriptionResult(
                segments=(
                    TranscriptSegment(
                        0,
                        500,
                        "Oh",
                        confidence=0.08,
                    ),
                ),
                text="Oh",
                backend="fixture",
                model=self.identity,
                device="test",
                language=options.language or "en",
                duration_ms=500,
                confidence_available=True,
            )

    coordinator = AdaptiveLongFormCoordinator(
        _LowConfidenceBackend("fixture:primary"),
        tmp_path / "job",
        retry_backend=_LowConfidenceBackend("fixture:independent"),
    )
    attempts = []
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (0, 2000),
            "failed hallucination",
            TranscriptionOptions(),
            attempts,
        )

    assert replacement is not None
    assert [item.text for item in replacement] == [
        "[UNCERTAIN: INDEPENDENT TRANSCRIPTIONS DISAGREE]"
    ]
    assert attempts[-1]["status"] == "INDEPENDENT_DISAGREEMENT"
    assert attempts[-1]["agreement"] == 1.0
    assert attempts[-1]["acoustic_confidence_floor"] == 0.08


def test_exact_single_token_agreement_remains_uncertain_even_at_high_confidence(tmp_path):
    class _SingleTokenBackend(_ConstantBackend):
        def transcribe(self, audio, options=None):
            return TranscriptionResult(
                segments=(TranscriptSegment(0, 300, "응?", confidence=0.9),),
                text="응?",
                backend="fixture",
                model=self.identity,
                device="test",
                language="ko",
                duration_ms=300,
                confidence_available=True,
            )

    coordinator = AdaptiveLongFormCoordinator(
        _SingleTokenBackend("fixture:primary"),
        tmp_path / "job",
        retry_backend=_SingleTokenBackend("fixture:independent"),
    )
    attempts = []
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (0, 1000),
            "failed hallucination",
            TranscriptionOptions(),
            attempts,
        )

    assert replacement is not None
    assert replacement[0].uncertain is True
    assert attempts[-1]["status"] == "INDEPENDENT_DISAGREEMENT"
    assert attempts[-1]["agreement"] == 1.0
    assert attempts[-1]["acoustic_confidence_floor"] == 0.9
    assert attempts[-1]["consensus_token_count"] == 1


def test_failed_text_language_only_prioritizes_and_never_limits_retry_languages(tmp_path):
    coordinator = AdaptiveLongFormCoordinator(_RetryingBackend(), tmp_path / "job")

    assert coordinator._target_languages("English hallucination") == (
        "en",
        "ro",
        "ru",
        "ko",
    )
    assert coordinator._target_languages("ошибочная расшифровка") == (
        "ru",
        "en",
        "ro",
        "ko",
    )


def test_targeted_retry_microdecodes_regions_and_marks_long_uncovered_intervals(tmp_path):
    class _Detector:
        identity = "fixture:speech-regions"

        def detect(self, audio):
            regions = (
                SpeechRegion(2000, 4000, "strict"),
                SpeechRegion(8000, 9000, "sensitive"),
            )
            return SpeechRegionPlan(
                self.identity,
                20_000,
                regions[:1],
                regions[1:],
                regions,
            )

    coordinator = AdaptiveLongFormCoordinator(
        _ConstantBackend("fixture:primary"),
        tmp_path / "job",
        retry_backend=_ConstantBackend("fixture:independent"),
        targeted_retry_region_detector=_Detector(),
    )
    attempts = []
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (10_000, 30_000),
            "failed hallucination",
            TranscriptionOptions(),
            attempts,
        )

    assert replacement is not None
    assert [(item.start_ms, item.end_ms, item.text, item.uncertain) for item in replacement] == [
        (10_000, 12_000, "[UNCERTAIN: SPEECH REGION NOT ISOLATED]", True),
        (12_000, 12_500, "agreed speech", False),
        (12_500, 14_000, "[UNCERTAIN: SPEECH REGION NOT ISOLATED]", True),
        (14_000, 18_000, "[UNCERTAIN: SPEECH REGION NOT ISOLATED]", True),
        (18_000, 18_500, "agreed speech", False),
        (19_000, 30_000, "[UNCERTAIN: SPEECH REGION NOT ISOLATED]", True),
    ]
    assert attempts[0]["kind"] == "targeted-speech-region-plan"
    assert attempts[-1]["kind"] == "targeted-speech-region-result"


def test_explicit_none_disables_retry_isolation_but_keeps_silence_detector(tmp_path):
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        _ConstantBackend("fixture:primary"),
        tmp_path / "job",
        retry_backend=_ConstantBackend("fixture:independent"),
        silence_verification_detector=detector,
        targeted_retry_region_detector=None,
    )
    attempts = []
    audio_candidates = _audio_candidate_set(tmp_path)
    with (
        patch.object(
            coordinator,
            "_extract_language_span",
            side_effect=lambda source, segment, destination: destination.write_bytes(b"span"),
        ),
        patch(
            "dubbing.transcription.adaptive.build_audio_candidates",
            return_value=audio_candidates,
        ),
    ):
        replacement = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (10_000, 10_500),
            "failed hallucination",
            TranscriptionOptions(),
            attempts,
        )

    assert replacement is not None
    assert detector.calls == 0
    assert not any(item["kind"] == "targeted-speech-region-plan" for item in attempts)


def test_code_switch_mismatch_redecodes_only_uncertain_turn(tmp_path):
    class _LanguageBackend(TranscriptionBackend):
        def __init__(self):
            self.languages = []

        @property
        def identity(self):
            return "fixture:languages"

        def transcribe(self, audio, options=None):
            self.languages.append(options.language)
            language = options.language or "en"
            return TranscriptionResult(
                (TranscriptSegment(0, 2000, "привет мир", language=language),),
                "привет мир",
                "fixture",
                "languages",
                "test",
                language,
                2000,
                False,
            )

    audio = tmp_path / "span.wav"
    audio.write_bytes(b"span")
    backend = _LanguageBackend()
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")

    def extract(source, segment, destination):
        destination.write_bytes(b"turn")

    with patch.object(coordinator, "_extract_language_span", side_effect=extract):
        result, receipt = coordinator._decode_chunk(
            audio,
            AdaptiveChunk(0, 0, 2000, 0, 2000, "end-of-media"),
            TranscriptionOptions(),
        )
    assert backend.languages == [None, "ru"]
    assert result.segments[0].language == "ru"
    assert any(item.get("kind") == "turn-language-redecode" for item in receipt["attempts"])


def test_short_language_retry_uses_context_but_admits_only_target_turn(tmp_path):
    class _ContextBackend(TranscriptionBackend):
        @property
        def identity(self):
            return "fixture:context-language"

        def transcribe(self, audio, options=None):
            return TranscriptionResult(
                (
                    TranscriptSegment(
                        0,
                        2900,
                        "outside before привет мир outside after",
                        language="ru",
                        words=(
                            TranscriptWord(0, 700, " outside before"),
                            TranscriptWord(1000, 1400, " привет"),
                            TranscriptWord(1400, 1800, " мир"),
                            TranscriptWord(2300, 2900, " outside after"),
                        ),
                    ),
                ),
                "outside before привет мир outside after",
                "fixture",
                "context-language",
                "test",
                "ru",
                3000,
                False,
            )

    original = TranscriptionResult(
        (TranscriptSegment(1000, 2000, "привет мир", language="en", uncertain=True),),
        "привет мир",
        "fixture",
        "context-language",
        "test",
        "en",
        4000,
        False,
    )
    coordinator = AdaptiveLongFormCoordinator(_ContextBackend(), tmp_path / "job")
    extracted = []

    def extract(source, segment, destination):
        extracted.append((segment.start_ms, segment.end_ms))
        destination.write_bytes(b"context")

    attempts = []
    with patch.object(coordinator, "_extract_language_span", side_effect=extract):
        resolved = coordinator._resolve_uncertain_turns(
            tmp_path / "audio.wav",
            original,
            coordinator.backend,
            TranscriptionOptions(),
            attempts,
        )

    assert extracted == [(0, 3000)]
    assert [(item.start_ms, item.end_ms, item.text) for item in resolved.segments] == [
        (1000, 1800, "привет мир")
    ]
    assert resolved.segments[0].language == "ru"
    assert resolved.segments[0].uncertain is False
    assert attempts[0]["context_start_ms"] == 0
    assert attempts[0]["context_end_ms"] == 3000


def test_language_retry_rejects_context_without_target_turn(tmp_path):
    class _NeighbourOnlyBackend(TranscriptionBackend):
        @property
        def identity(self):
            return "fixture:neighbour-only"

        def transcribe(self, audio, options=None):
            return TranscriptionResult(
                (TranscriptSegment(0, 700, "соседняя речь", language="ru"),),
                "соседняя речь",
                "fixture",
                "neighbour-only",
                "test",
                "ru",
                3000,
                False,
            )

    original_segment = TranscriptSegment(1000, 2000, "неясно", language="en", uncertain=True)
    original = TranscriptionResult(
        (original_segment,),
        original_segment.text,
        "fixture",
        "neighbour-only",
        "test",
        "en",
        4000,
        False,
    )
    coordinator = AdaptiveLongFormCoordinator(_NeighbourOnlyBackend(), tmp_path / "job")
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, segment, destination: destination.write_bytes(b"context"),
    ):
        resolved = coordinator._resolve_uncertain_turns(
            tmp_path / "audio.wav",
            original,
            coordinator.backend,
            TranscriptionOptions(),
            [],
        )

    assert resolved == original


def test_language_retry_preserves_turn_beyond_audio_duration_as_uncertain(tmp_path):
    class _MustNotRunBackend(TranscriptionBackend):
        @property
        def identity(self):
            return "fixture:must-not-run"

        def transcribe(self, audio, options=None):
            raise AssertionError("out-of-audio language retry must not run")

    original_segment = TranscriptSegment(
        74_100,
        74_120,
        "Продолжение следует...",
        language="en",
        uncertain=True,
        words=(TranscriptWord(74_100, 74_120, " следует..."),),
    )
    original = TranscriptionResult(
        (original_segment,),
        original_segment.text,
        "fixture",
        "real-like-overflow",
        "test",
        "en",
        65_407,
        False,
    )
    coordinator = AdaptiveLongFormCoordinator(_MustNotRunBackend(), tmp_path / "job")
    attempts = []

    resolved = coordinator._resolve_uncertain_turns(
        tmp_path / "audio.wav",
        original,
        coordinator.backend,
        TranscriptionOptions(),
        attempts,
    )

    assert resolved == original
    assert attempts == [
        {
            "kind": "turn-language-redecode-skipped",
            "source_start_ms": 74_100,
            "source_end_ms": 74_120,
            "audio_duration_ms": 65_407,
            "reason": "TARGET_OUTSIDE_AUDIO_DURATION",
        }
    ]


def test_detected_korean_chunk_uses_always_forced_retry(tmp_path):
    class _KoreanBackend(TranscriptionBackend):
        @property
        def identity(self):
            return "fixture:korean"

        def transcribe(self, audio, options=None):
            forced = options.language == "ko"
            text = "강제 한국어" if forced else "자동 한국어"
            return TranscriptionResult(
                (
                    TranscriptSegment(
                        0,
                        2000,
                        text,
                        language="ko",
                        diagnostics=DecodeDiagnostics(avg_log_probability=-0.2 if forced else -0.1),
                    ),
                ),
                text,
                "fixture",
                "korean",
                "test",
                "ko",
                2000,
                False,
            )

    coordinator = AdaptiveLongFormCoordinator(
        _KoreanBackend(),
        tmp_path / "job",
        language_retry_policy={"ko": "always"},
    )
    audio = tmp_path / "span.wav"
    audio.write_bytes(b"span")

    result, receipt = coordinator._decode_chunk(
        audio,
        AdaptiveChunk(0, 0, 2000, 0, 2000, "end-of-media"),
        TranscriptionOptions(),
    )

    assert result.text == "강제 한국어"
    retry = next(
        item for item in receipt["attempts"] if item["kind"] == "detected-chunk-language-redecode"
    )
    assert retry["selected"] is True


def test_unknown_latin_turn_stays_uncertain_instead_of_trying_every_language(tmp_path):
    class _UnknownBackend(TranscriptionBackend):
        def __init__(self):
            self.calls = 0

        @property
        def identity(self):
            return "fixture:unknown"

        def transcribe(self, audio, options=None):
            self.calls += 1
            return TranscriptionResult(
                (TranscriptSegment(0, 2000, "ambiguous latin words", language="nn"),),
                "ambiguous latin words",
                "fixture",
                "unknown",
                "test",
                "nn",
                2000,
                False,
            )

    backend = _UnknownBackend()
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")
    audio = tmp_path / "span.wav"
    audio.write_bytes(b"span")
    result, _ = coordinator._decode_chunk(
        audio,
        AdaptiveChunk(0, 0, 2000, 0, 2000, "end-of-media"),
        TranscriptionOptions(),
    )
    assert backend.calls == 1
    assert result.segments[0].uncertain


def test_processing_cap_binds_full_source_and_limits_plan_result_and_replay(tmp_path):
    source = tmp_path / "lesson.mov"
    source.write_bytes(b"full source including unrelated tail")
    backend = _IndependentBackend()
    observed = {}

    def plan(duration_ms, silence_centres, *, silence_intervals=()):
        observed["duration_ms"] = duration_ms
        observed["silence_centres"] = silence_centres
        observed["silence_intervals"] = silence_intervals
        return (AdaptiveChunk(0, 0, duration_ms, 0, duration_ms, "end-of-media"),)

    planner = SimpleNamespace(
        plan=plan,
        to_dict=lambda: {"fixture": "processing-cap"},
    )
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        planner=planner,
    )

    with (
        patch(
            "dubbing.transcription.adaptive.probe_media",
            return_value=MediaProbe(
                10_000,
                source.stat().st_size,
                (AudioStream(0, "pcm", 1, 16_000),),
            ),
        ),
        patch(
            "dubbing.transcription.adaptive.detect_silence_intervals",
            return_value=((5_000, 8_000), (9_000, 10_000)),
        ),
        patch.object(
            coordinator,
            "_extract",
            side_effect=lambda source, streams, chunk, output: output.write_bytes(
                b"processed prefix"
            ),
        ),
    ):
        result, quality = coordinator.run(source, processing_end_ms=6_000)
        replay, _ = coordinator.run(source, processing_end_ms=6_000)

    assert observed == {
        "duration_ms": 6_000,
        "silence_centres": (5_500,),
        "silence_intervals": ((5_000, 6_000),),
    }
    assert result == replay
    assert result.duration_ms == 6_000
    assert max(segment.end_ms for segment in result.segments) == 6_000
    assert result.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.provenance["processing_scope"] == {
        "start_ms": 0,
        "end_ms": 6_000,
        "processed_duration_ms": 6_000,
        "full_source_duration_ms": 10_000,
        "full_source_sha256_bound": True,
    }
    assert quality["metrics"]["duration_ms"] == 6_000
    assert backend.calls == 1
    manifest = json.loads((tmp_path / "job" / "manifest.json").read_text())
    assert manifest["probe"]["duration_ms"] == 10_000
    assert manifest["processing_scope"]["end_ms"] == 6_000
    assert manifest["chunks"][0]["extract_end_ms"] == 6_000


def test_processing_cap_change_or_full_source_change_rejects_checkpoint(tmp_path):
    source = tmp_path / "lesson.mov"
    source.write_bytes(b"source with tail")
    coordinator = AdaptiveLongFormCoordinator(
        _IndependentBackend(),
        tmp_path / "job",
    )
    probe = MediaProbe(
        10_000,
        source.stat().st_size,
        (AudioStream(0, "pcm", 1, 16_000),),
    )

    with (
        patch("dubbing.transcription.adaptive.probe_media", return_value=probe),
        patch(
            "dubbing.transcription.adaptive.detect_silence_intervals",
            return_value=(),
        ),
        patch.object(
            coordinator,
            "_extract",
            side_effect=lambda source, streams, chunk, output: output.write_bytes(b"prefix"),
        ),
    ):
        coordinator.run(source, processing_end_ms=6_000)
        with pytest.raises(TranscriptionError, match="does not match"):
            coordinator.run(source, processing_end_ms=5_000)
        source.write_bytes(b"changed full source tail")
        with pytest.raises(TranscriptionError, match="does not match"):
            coordinator.run(source, processing_end_ms=6_000)


@pytest.mark.parametrize("processing_end_ms", [0, -1, 10_001, 1.5, True])
def test_processing_cap_must_fit_probed_source(tmp_path, processing_end_ms):
    source = tmp_path / "lesson.mov"
    source.write_bytes(b"source")
    coordinator = AdaptiveLongFormCoordinator(_IndependentBackend(), tmp_path / "job")
    with patch(
        "dubbing.transcription.adaptive.probe_media",
        return_value=MediaProbe(
            10_000,
            source.stat().st_size,
            (AudioStream(0, "pcm", 1, 16_000),),
        ),
    ):
        with pytest.raises(TranscriptionError, match="processing_end_ms"):
            coordinator.run(source, processing_end_ms=processing_end_ms)
