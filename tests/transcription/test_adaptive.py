import hashlib
import json
import shutil
import wave
from array import array
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import dubbing.transcription.adaptive as adaptive_module
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
from dubbing.transcription.job import create_source_binding
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
    def __init__(self, text="clean ordinary phrase", *, duration_ms=10_000):
        self.calls = 0
        self.text = text
        self.duration_ms = duration_ms

    @property
    def identity(self):
        return "fixture:independent"

    def transcribe(self, audio, options=None):
        self.calls += 1
        return TranscriptionResult(
            segments=(TranscriptSegment(0, self.duration_ms, self.text),),
            text=self.text,
            backend="fixture",
            model="independent",
            device="test",
            language=options.language or "en",
            duration_ms=self.duration_ms,
            confidence_available=False,
        )


class _OutOfAudioBackend(TranscriptionBackend):
    def __init__(self):
        self.calls = 0

    @property
    def identity(self):
        return "fixture:out-of-audio"

    def transcribe(self, audio, options=None):
        self.calls += 1
        return TranscriptionResult(
            segments=(TranscriptSegment(74_100, 74_120, "continuation"),),
            text="continuation",
            backend="fixture",
            model="out-of-audio",
            device="test",
            language=options.language or "en",
            duration_ms=65_407,
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


def test_real_six_gap_near_silence_stops_after_empty_bounded_primary(tmp_path):
    source = tmp_path / "lesson.mov"
    source.write_bytes(b"source")
    chunk = _lesson_silence_chunk()
    backend = _OutOfAudioBackend()
    retry_backend = _IndependentBackend()
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        planner=_single_chunk_planner(chunk),
        retry_backend=retry_backend,
        silence_verification_detector=detector,
        targeted_retry_region_detector=None,
    )
    real_silence_intervals = (
        (960_400, 961_715),
        (961_735, 963_064),
        (963_084, 1_019_110),
        (1_019_623, 1_022_101),
        (1_023_308, 1_024_306),
        (1_024_327, 1_025_491),
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
            return_value=real_silence_intervals,
        ),
        patch.object(
            coordinator,
            "_extract",
            side_effect=lambda source, streams, selected_chunk, output: output.write_bytes(
                b"near silence"
            ),
        ),
    ):
        result, _ = coordinator.run(source)
        replay, _ = coordinator.run(source)

    assert result.segments == ()
    assert result.text == ""
    assert replay == result
    assert backend.calls == 1
    assert retry_backend.calls == 0
    assert detector.calls == 1
    receipt = json.loads((tmp_path / "job" / "receipts" / "000015.json").read_text())
    assert [item["kind"] for item in receipt["attempts"]] == [
        "out-of-audio-segment-rejection",
        "chunk-silence-classification",
    ]
    classification = receipt["attempts"][-1]
    assert classification["status"] == "CONFIRMED_SILENCE_AFTER_EMPTY_ASR"
    assert classification["energy_silence_coverage_ms"] == 63_310
    assert classification["energy_silence_coverage_ratio"] == 0.967939
    assert classification["total_uncovered_ms"] == 2_097
    assert classification["maximum_uncovered_gap_ms"] == 1_207
    assert receipt["selected_attempt"] == 1
    manifest = json.loads((tmp_path / "job" / "manifest.json").read_text())
    policy = manifest["silence_admission"]["near_silence_after_empty_asr"]
    assert policy["minimum_energy_coverage_ratio"] == 0.96
    assert policy["maximum_total_uncovered_ms"] == 2_500
    assert policy["maximum_uncovered_gap_ms"] == 1_500
    turn_policy = manifest["silence_admission"]["uncertain_turn_after_empty_language_retry"]
    assert turn_policy == {
        "policy": "full-energy-plus-empty-forced-language-plus-empty-sensitive-vad-v1",
        "required_energy_coverage_ratio": 1.0,
        "forced_candidate_valid_segment_count_required": 0,
        "forced_candidate_failure_codes": ["malformed_or_empty_output"],
        "sensitive_vad_target_overlap_ms_required": 0,
    }


def test_near_silence_rechecks_after_confirmed_no_speech_turn_is_removed(tmp_path):
    class _InBoundsAndOutOfBoundsStockHallucinationBackend(TranscriptionBackend):
        def __init__(self):
            self.calls = 0

        @property
        def identity(self):
            return "fixture:in-and-out-of-bounds-stock-hallucination"

        def transcribe(self, audio, options=None):
            self.calls += 1
            if options.language == "ru":
                return TranscriptionResult(
                    (),
                    "",
                    "fixture",
                    "in-and-out-of-bounds-stock-hallucination",
                    "test",
                    "ru",
                    65_407,
                    False,
                )
            text = "Продолжение следует..."
            return TranscriptionResult(
                (
                    TranscriptSegment(
                        29_100,
                        29_120,
                        text,
                        language="en",
                        uncertain=True,
                    ),
                    TranscriptSegment(
                        74_100,
                        74_120,
                        text,
                        language="en",
                        uncertain=True,
                    ),
                ),
                f"{text} {text}",
                "fixture",
                "in-and-out-of-bounds-stock-hallucination",
                "test",
                "en",
                65_407,
                False,
            )

    chunk = _lesson_silence_chunk()
    backend = _InBoundsAndOutOfBoundsStockHallucinationBackend()
    retry_backend = _IndependentBackend()
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        retry_backend=retry_backend,
        silence_verification_detector=detector,
        targeted_retry_region_detector=None,
    )
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"tascam-shaped near silence")
    intervals = (
        (0, 1_315),
        (1_335, 2_664),
        (2_684, 58_710),
        (59_223, 61_700),
        (62_908, 63_906),
        (63_928, 65_091),
    )

    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, marker, destination: destination.write_bytes(
            b"hash-bound-context"
        ),
    ):
        result, receipt = coordinator._decode_chunk(
            audio,
            chunk,
            TranscriptionOptions(),
            energy_silence_coverage_ms=63_308,
            energy_silence_intervals=intervals,
        )

    assert result.segments == ()
    assert result.text == ""
    assert result.provenance["classification"] == "confirmed_silence_after_empty_asr"
    assert backend.calls == 2
    assert retry_backend.calls == 0
    assert detector.calls == 2
    assert [item["kind"] for item in receipt["attempts"]] == [
        "out-of-audio-segment-rejection",
        "turn-language-redecode",
        "uncertain-turn-silence-adjudication",
        "chunk-silence-classification",
    ]
    assert receipt["attempts"][2]["status"] == "DROPPED_CONFIRMED_NO_SPEECH"
    classification = receipt["attempts"][3]
    assert classification["status"] == "CONFIRMED_SILENCE_AFTER_EMPTY_ASR"
    assert classification["energy_silence_coverage_ms"] == 63_308
    assert classification["total_uncovered_ms"] == 2_099
    assert classification["maximum_uncovered_gap_ms"] == 1_208
    assert receipt["selected_attempt"] == 3
    assert receipt["targeted_retry_exhausted"] is False
    assert not any(item["kind"] == "targeted-span-redecode" for item in receipt["attempts"])


def test_near_silence_sensitive_vad_speech_preserves_fail_closed_retry(tmp_path):
    chunk = _lesson_silence_chunk()
    backend = _OutOfAudioBackend()
    detector = _SpeechDetector((SpeechRegion(62_900, 63_200, "sensitive"),))
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        retry_backend=_IndependentBackend(),
        silence_verification_detector=detector,
        targeted_retry_region_detector=None,
    )
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"quiet speech")
    intervals = (
        (0, 1_315),
        (1_335, 2_664),
        (2_684, 58_710),
        (59_223, 61_701),
        (62_908, 63_906),
        (63_927, 65_091),
    )

    with patch.object(
        coordinator,
        "_repair_rejected_spans",
        side_effect=lambda audio, result, quality, duration, options, attempts: (
            result,
            quality,
        ),
    ) as repair:
        result, receipt = coordinator._decode_chunk(
            audio,
            chunk,
            TranscriptionOptions(),
            energy_silence_coverage_ms=63_310,
            energy_silence_intervals=intervals,
        )

    assert result.segments == ()
    assert detector.calls == 1
    assert repair.call_count == 1
    classifications = [
        item for item in receipt["attempts"] if item["kind"] == "chunk-silence-classification"
    ]
    assert classifications[0]["status"] == "SEMANTIC_SPEECH_DETECTED_AFTER_EMPTY_ASR"
    assert receipt["targeted_retry_exhausted"] is True


@pytest.mark.parametrize(
    ("coverage_ms", "intervals"),
    (
        (62_000, ((0, 62_000),)),
        (63_500, ((0, 60_000), (61_907, 65_407))),
    ),
)
def test_near_silence_unsafe_energy_shape_preserves_fail_closed_retry(
    tmp_path, coverage_ms, intervals
):
    chunk = _lesson_silence_chunk()
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        _OutOfAudioBackend(),
        tmp_path / "job",
        retry_backend=_IndependentBackend(),
        silence_verification_detector=detector,
        targeted_retry_region_detector=None,
    )
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"not safely silent")

    with patch.object(
        coordinator,
        "_repair_rejected_spans",
        side_effect=lambda audio, result, quality, duration, options, attempts: (
            result,
            quality,
        ),
    ) as repair:
        coordinator._decode_chunk(
            audio,
            chunk,
            TranscriptionOptions(),
            energy_silence_coverage_ms=coverage_ms,
            energy_silence_intervals=intervals,
        )

    assert detector.calls == 0
    assert repair.call_count == 1


def test_valid_primary_speech_cannot_be_reclassified_as_near_silence(tmp_path):
    chunk = _lesson_silence_chunk()
    detector = _SpeechDetector()
    backend = _IndependentBackend()
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        silence_verification_detector=detector,
        targeted_retry_region_detector=None,
    )
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"valid quiet speech")

    result, receipt = coordinator._decode_chunk(
        audio,
        chunk,
        TranscriptionOptions(),
        energy_silence_coverage_ms=63_310,
        energy_silence_intervals=(
            (0, 1_315),
            (1_335, 2_664),
            (2_684, 58_710),
            (59_223, 61_701),
            (62_908, 63_906),
            (63_927, 65_091),
        ),
    )

    assert result.text == "clean ordinary phrase"
    assert backend.calls == 1
    assert detector.calls == 0
    assert not any(item["kind"] == "chunk-silence-classification" for item in receipt["attempts"])


def test_near_silence_without_detector_preserves_fail_closed_retry(tmp_path):
    chunk = _lesson_silence_chunk()
    coordinator = AdaptiveLongFormCoordinator(
        _OutOfAudioBackend(),
        tmp_path / "job",
        retry_backend=_IndependentBackend(),
        silence_verification_detector=None,
        targeted_retry_region_detector=None,
    )
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"unverified near silence")

    with patch.object(
        coordinator,
        "_repair_rejected_spans",
        side_effect=lambda audio, result, quality, duration, options, attempts: (
            result,
            quality,
        ),
    ) as repair:
        _, receipt = coordinator._decode_chunk(
            audio,
            chunk,
            TranscriptionOptions(),
            energy_silence_coverage_ms=63_310,
            energy_silence_intervals=(
                (0, 1_315),
                (1_335, 2_664),
                (2_684, 58_710),
                (59_223, 61_701),
                (62_908, 63_906),
                (63_927, 65_091),
            ),
        )

    assert repair.call_count == 1
    assert not any(item["kind"] == "chunk-silence-classification" for item in receipt["attempts"])


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
        self.calls = []

    @property
    def identity(self):
        return self._identity

    def transcribe(self, audio, options=None):
        self.calls.append((Path(audio).name, options.language))
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

    manifest = coordinator._manifest(source, create_source_binding(source), probe, chunks)

    assert manifest["silence_detection"]["minimum_silence_seconds"] == 0.5


def test_language_retry_policy_is_checkpoint_bound(tmp_path):
    coordinator = AdaptiveLongFormCoordinator(
        _RetryingBackend(), tmp_path, language_retry_policy={"ko": "always"}
    )
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    probe = MediaProbe(10_000, 6, (AudioStream(0, "pcm", 1, 16_000),))
    chunks = (AdaptiveChunk(0, 0, 10_000, 0, 10_000, "end-of-media"),)

    manifest = coordinator._manifest(source, create_source_binding(source), probe, chunks)

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


def test_legacy_checkpoint_manifest_fails_closed(tmp_path):
    backend = _RetryingBackend()
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    probe = MediaProbe(60_000, 6, (AudioStream(0, "pcm", 2, 48_000),))
    chunks = (AdaptiveChunk(0, 0, 60_000, 0, 60_000, "end-of-media"),)
    expected = coordinator._manifest(source, create_source_binding(source), probe, chunks)
    existing = dict(expected)
    existing["coordinator_version"] = "adaptive-long-form-v3"
    existing.pop("targeted_retry")
    manifest = tmp_path / "job" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(TranscriptionError, match="does not match"):
        coordinator._admit_manifest(expected)

    assert json.loads(manifest.read_text(encoding="utf-8")) == existing


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
    binding = create_source_binding(source)
    both_roles._admit_manifest(both_roles._manifest(source, binding, probe, chunks))
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
        silence_only._admit_manifest(silence_only._manifest(source, binding, probe, chunks))


def test_near_silence_policy_change_invalidates_adaptive_checkpoint_manifest(tmp_path):
    coordinator = AdaptiveLongFormCoordinator(_RetryingBackend(), tmp_path / "job")
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    probe = MediaProbe(60_000, 6, (AudioStream(0, "pcm", 2, 48_000),))
    chunks = (AdaptiveChunk(0, 0, 60_000, 0, 60_000, "end-of-media"),)
    expected = coordinator._manifest(source, create_source_binding(source), probe, chunks)
    existing = json.loads(json.dumps(expected))
    existing["silence_admission"]["near_silence_after_empty_asr"]["maximum_uncovered_gap_ms"] = (
        1_501
    )
    manifest = tmp_path / "job" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(existing), encoding="utf-8")

    with pytest.raises(
        TranscriptionError,
        match="adaptive checkpoint does not match source or configuration",
    ):
        coordinator._admit_manifest(expected)


def test_targeted_retry_budget_is_checkpoint_bound(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    probe = MediaProbe(60_000, 6, (AudioStream(0, "pcm", 2, 48_000),))
    chunks = (AdaptiveChunk(0, 0, 60_000, 0, 60_000, "end-of-media"),)
    coordinator = AdaptiveLongFormCoordinator(
        _RetryingBackend(),
        tmp_path / "job",
        maximum_targeted_decode_attempts_per_chunk=8,
    )
    binding = create_source_binding(source)
    expected = coordinator._manifest(source, binding, probe, chunks)

    assert expected["targeted_retry"]["maximum_decode_attempts_per_chunk"] == 8
    coordinator._admit_manifest(expected)

    changed = AdaptiveLongFormCoordinator(
        _RetryingBackend(),
        tmp_path / "job",
        maximum_targeted_decode_attempts_per_chunk=10,
    )
    with pytest.raises(
        TranscriptionError,
        match="adaptive checkpoint does not match source or configuration",
    ):
        changed._admit_manifest(changed._manifest(source, binding, probe, chunks))


@pytest.mark.parametrize("limit", (1, 1_025))
def test_targeted_retry_budget_rejects_unsafe_limits(tmp_path, limit):
    with pytest.raises(
        ValueError,
        match="maximum targeted decode attempts per chunk must be between 2 and 1024",
    ):
        AdaptiveLongFormCoordinator(
            _RetryingBackend(),
            tmp_path / "job",
            maximum_targeted_decode_attempts_per_chunk=limit,
        )


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
    assert {item[0] for item in primary.calls + independent.calls} == {
        "raw.wav",
        "channel-0.wav",
        "channel-1.wav",
    }
    assert not any(
        item["kind"] == "targeted-audio-escalation-short-circuit"
        for item in attempts
    )


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
    assert {item[0] for item in primary.calls + independent.calls} == {
        "raw.wav",
        "channel-0.wav",
        "channel-1.wav",
    }


def test_raw_consensus_short_circuits_processed_audio_escalation(tmp_path):
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
    assert {item[0] for item in primary.calls + independent.calls} == {"raw.wav"}
    short_circuit = next(
        item for item in attempts if item["kind"] == "targeted-audio-escalation-short-circuit"
    )
    assert short_circuit["status"] == "RAW_CONSENSUS_FINAL"
    assert short_circuit["skipped_audio_candidates"] == ["channel-0", "channel-1"]
    assert short_circuit["avoided_scheduled_decode_count"] == 2 * len(
        primary.calls + independent.calls
    )


def test_local_retry_budget_stops_before_partial_audio_candidate_and_fails_closed(tmp_path):
    primary = _AudioCandidateBackend(
        "fixture:primary",
        {
            "raw.wav": "raw primary transcript",
            "channel-0.wav": "processed candidate consensus",
            "channel-1.wav": "processed candidate consensus",
        },
    )
    independent = _AudioCandidateBackend(
        "fixture:independent",
        {
            "raw.wav": "unrelated independent testimony",
            "channel-0.wav": "processed candidate consensus",
            "channel-1.wav": "processed candidate consensus",
        },
    )
    coordinator = AdaptiveLongFormCoordinator(
        primary,
        tmp_path / "job",
        retry_backend=independent,
        candidate_languages=("en", "ru"),
        maximum_targeted_decode_attempts_per_chunk=8,
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

    targeted = [item for item in attempts if item["kind"] == "targeted-span-redecode"]
    assert len(targeted) == 8
    assert {item["audio_candidate"] for item in targeted} == {"raw"}
    assert len(primary.calls) + len(independent.calls) == 8
    assert replacement is not None
    assert [(item.text, item.uncertain) for item in replacement] == [
        ("[UNCERTAIN: INDEPENDENT TRANSCRIPTIONS DISAGREE]", True)
    ]
    budget = next(
        item
        for item in attempts
        if item["kind"] == "targeted-local-retry-budget-exhausted"
    )
    assert budget["completed_decode_attempts"] == 8
    assert budget["required_next_audio_candidate_decode_count"] == 8
    assert budget["next_audio_candidate"] == "channel-0"
    assert attempts[-1]["status"] == "LOCAL_RETRY_BUDGET_EXHAUSTED"
    assert attempts[-1]["processed_consensus_count"] == 0
    assert attempts[-1]["selected"] is False


def test_raw_consensus_can_finish_at_local_retry_budget(tmp_path):
    texts = {
        "raw.wav": "trustworthy raw recording transcript",
        "channel-0.wav": "unused processed interpretation",
        "channel-1.wav": "unused processed alternative",
    }
    primary = _AudioCandidateBackend("fixture:primary", texts)
    independent = _AudioCandidateBackend("fixture:independent", texts)
    coordinator = AdaptiveLongFormCoordinator(
        primary,
        tmp_path / "job",
        retry_backend=independent,
        candidate_languages=("en", "ru"),
        maximum_targeted_decode_attempts_per_chunk=8,
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

    assert " ".join(item.text for item in replacement) == texts["raw.wav"]
    assert len(primary.calls) + len(independent.calls) == 8
    assert attempts[-1]["status"] == "CONSENSUS_PASS"
    assert attempts[-1]["raw_consensus_available"] is True
    assert not any(
        item["kind"] == "targeted-local-retry-budget-exhausted" for item in attempts
    )


def test_local_retry_budget_is_shared_across_spans_in_one_chunk(tmp_path):
    primary = _AudioCandidateBackend(
        "fixture:primary",
        {
            "raw.wav": "raw primary transcript",
            "channel-0.wav": "processed candidate consensus",
            "channel-1.wav": "processed candidate consensus",
        },
    )
    independent = _AudioCandidateBackend(
        "fixture:independent",
        {
            "raw.wav": "unrelated independent testimony",
            "channel-0.wav": "processed candidate consensus",
            "channel-1.wav": "processed candidate consensus",
        },
    )
    coordinator = AdaptiveLongFormCoordinator(
        primary,
        tmp_path / "job",
        retry_backend=independent,
        candidate_languages=("en", "ru"),
        maximum_targeted_decode_attempts_per_chunk=8,
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
        first = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (0, 20_000),
            "failed original words",
            TranscriptionOptions(),
            attempts,
        )
        calls_after_first = len(primary.calls) + len(independent.calls)
        second = coordinator._decode_target_span(
            tmp_path / "source.wav",
            (20_000, 40_000),
            "another failed original",
            TranscriptionOptions(),
            attempts,
        )

    assert first is not None and first[0].uncertain is True
    assert second is None
    assert calls_after_first == 8
    assert len(primary.calls) + len(independent.calls) == calls_after_first
    budget_events = [
        item
        for item in attempts
        if item["kind"] == "targeted-local-retry-budget-exhausted"
    ]
    assert len(budget_events) == 2
    assert budget_events[-1]["source_start_ms"] == 20_000
    assert budget_events[-1]["completed_decode_attempts"] == 8


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


def test_unsupported_japanese_script_is_never_clean_in_english_result():
    from dubbing.transcription.language import annotate_transcript_languages
    from dubbing.transcription.quality import (
        TranscriptQualityStatus,
        evaluate_transcript_quality,
    )

    result = TranscriptionResult(
        segments=(
            TranscriptSegment(
                0,
                10_000,
                "ご視聴ありがとうございました",
                language="en",
            ),
        ),
        text="ご視聴ありがとうございました",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=10_000,
        confidence_available=False,
    )

    annotated = annotate_transcript_languages(result)
    report = evaluate_transcript_quality(annotated, expected_duration_ms=10_000)

    assert annotated.segments[0].uncertain is True
    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert "unsupported_script_language_mismatch" in {
        item.code for item in report.issues
    }


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


class _PostAudioBackend(TranscriptionBackend):
    def __init__(self, segments):
        self.calls = 0
        self.segments = segments

    @property
    def identity(self):
        return "fixture:post-audio"

    def transcribe(self, audio, options=None):
        self.calls += 1
        return TranscriptionResult(
            segments=self.segments,
            text=" ".join(segment.text for segment in self.segments),
            backend="fixture",
            model="post-audio",
            device="test",
            language="en",
            duration_ms=74_980,
            confidence_available=False,
        )


def test_wholly_post_audio_segment_is_rejected_without_targeted_retry(tmp_path):
    backend = _PostAudioBackend(
        (
            TranscriptSegment(0, 10_000, "real speech"),
            TranscriptSegment(
                74_180,
                74_980,
                "And...",
                words=(TranscriptWord(74_180, 74_980, " And..."),),
            ),
        )
    )
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"audio")

    result, receipt = coordinator._decode_chunk(
        audio,
        AdaptiveChunk(0, 0, 64_202, 0, 64_202, "fixture"),
        TranscriptionOptions(),
    )

    assert backend.calls == 1
    assert result.text == "real speech"
    assert len(result.segments) == 1
    assert (result.segments[0].start_ms, result.segments[0].end_ms) == (0, 10_000)
    assert result.segments[0].text == "real speech"
    assert receipt["targeted_retry_exhausted"] is False
    rejection = next(
        item for item in receipt["attempts"] if item["kind"] == "out-of-audio-segment-rejection"
    )
    assert rejection["audio_duration_ms"] == 64_202
    assert rejection["raw_quality"]["status"] == "REPROCESS_REQUIRED"
    assert rejection["changes"] == [
        {
            "action": "DROPPED_WHOLE_SEGMENT",
            "segment_index": 1,
            "observed_start_ms": 74_180,
            "observed_end_ms": 74_980,
        }
    ]
    assert not any(item["kind"] == "targeted-span-redecode" for item in receipt["attempts"])


def test_segment_overlapping_audio_end_is_clamped_by_word_midpoint(tmp_path):
    backend = _PostAudioBackend(
        (
            TranscriptSegment(
                63_800,
                65_000,
                "kept dropped",
                words=(
                    TranscriptWord(63_800, 64_100, " kept"),
                    TranscriptWord(64_300, 65_000, " dropped"),
                ),
            ),
        )
    )
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"audio")

    result, receipt = coordinator._decode_chunk(
        audio,
        AdaptiveChunk(0, 0, 64_202, 0, 64_202, "fixture"),
        TranscriptionOptions(),
    )

    assert backend.calls == 1
    assert result.text == "kept"
    assert result.segments[0].end_ms == 64_202
    assert result.segments[0].words == (TranscriptWord(63_800, 64_100, " kept"),)
    assert receipt["targeted_retry_exhausted"] is False
    rejection = next(
        item for item in receipt["attempts"] if item["kind"] == "out-of-audio-segment-rejection"
    )
    assert rejection["changes"][0]["action"] == "CLAMPED_OVERLAPPING_SEGMENT"
    assert rejection["changes"][0]["dropped_word_count"] == 1


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


class _EmptyLanguageRetryBackend(TranscriptionBackend):
    @property
    def identity(self):
        return "fixture:empty-language-retry"

    def transcribe(self, audio, options=None):
        return TranscriptionResult(
            (),
            "",
            "fixture",
            "empty-language-retry",
            "test",
            options.language,
            3_000,
            False,
        )


@pytest.mark.parametrize(
    ("start_ms", "end_ms", "text"),
    [
        (50_200, 51_600, "зелень"),
        (29_100, 29_120, "Продолжение следует..."),
        (167_400, 167_420, "Продолжение следует..."),
    ],
    ids=("greenery", "continuation-middle", "continuation-tail"),
)
def test_empty_language_retry_drops_exact_full_silence_zero_vad_patterns(
    tmp_path, start_ms, end_ms, text
):
    segment = TranscriptSegment(start_ms, end_ms, text, language="en", uncertain=True)
    original = TranscriptionResult(
        (segment,),
        text,
        "fixture",
        "empty-language-retry",
        "test",
        "en",
        200_000,
        False,
        provenance={"fixture": True},
    )
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        _EmptyLanguageRetryBackend(),
        tmp_path / "job",
        silence_verification_detector=detector,
    )
    attempts = []
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, marker, destination: destination.write_bytes(
            b"hash-bound-context"
        ),
    ):
        resolved = coordinator._resolve_uncertain_turns(
            tmp_path / "audio.wav",
            original,
            coordinator.backend,
            TranscriptionOptions(),
            attempts,
            energy_silence_intervals=((start_ms, end_ms),),
        )

    assert resolved.segments == ()
    assert resolved.text == ""
    assert detector.calls == 1
    adjudication = attempts[-1]
    assert adjudication["kind"] == "uncertain-turn-silence-adjudication"
    assert adjudication["status"] == "DROPPED_CONFIRMED_NO_SPEECH"
    assert adjudication["source_start_ms"] == start_ms
    assert adjudication["source_end_ms"] == end_ms
    assert adjudication["target_energy_silence_coverage_ms"] == end_ms - start_ms
    assert adjudication["sensitive_vad_target_overlap_ms"] == 0
    assert adjudication["context_audio_sha256"] == hashlib.sha256(b"hash-bound-context").hexdigest()
    assert adjudication["original_text_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert adjudication["forced_retry_quality_sha256"]
    assert adjudication["speech_region_plan_sha256"]
    provenance = resolved.provenance["uncertain_turn_silence_adjudications"]
    assert len(provenance) == 1
    assert provenance[0]["context_audio_sha256"] == adjudication["context_audio_sha256"]


def test_uncertain_turn_silence_adjudication_preserves_sensitive_vad_speech(tmp_path):
    segment = TranscriptSegment(1_000, 2_000, "неясно", language="en", uncertain=True)
    original = TranscriptionResult(
        (segment,), segment.text, "fixture", "empty", "test", "en", 4_000, False
    )
    detector = _SpeechDetector((SpeechRegion(900, 1_100, "sensitive"),))
    coordinator = AdaptiveLongFormCoordinator(
        _EmptyLanguageRetryBackend(),
        tmp_path / "job",
        silence_verification_detector=detector,
    )
    attempts = []
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, marker, destination: destination.write_bytes(b"context"),
    ):
        resolved = coordinator._resolve_uncertain_turns(
            tmp_path / "audio.wav",
            original,
            coordinator.backend,
            TranscriptionOptions(),
            attempts,
            energy_silence_intervals=((1_000, 2_000),),
        )

    assert resolved == original
    assert attempts[-1]["status"] == "PRESERVED_SENSITIVE_SPEECH"
    assert attempts[-1]["sensitive_vad_target_overlap_ms"] == 100


@pytest.mark.parametrize(
    ("detector_configured", "energy_intervals"),
    [
        (False, ((1_000, 2_000),)),
        (True, ((1_000, 1_999),)),
    ],
    ids=("detector-absent", "partial-energy-silence"),
)
def test_uncertain_turn_silence_adjudication_requires_full_energy_and_detector(
    tmp_path, detector_configured, energy_intervals
):
    segment = TranscriptSegment(1_000, 2_000, "неясно", language="en", uncertain=True)
    original = TranscriptionResult(
        (segment,), segment.text, "fixture", "empty", "test", "en", 4_000, False
    )
    detector = _SpeechDetector() if detector_configured else None
    coordinator = AdaptiveLongFormCoordinator(
        _EmptyLanguageRetryBackend(),
        tmp_path / "job",
        silence_verification_detector=detector,
    )
    attempts = []
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, marker, destination: destination.write_bytes(b"context"),
    ):
        resolved = coordinator._resolve_uncertain_turns(
            tmp_path / "audio.wav",
            original,
            coordinator.backend,
            TranscriptionOptions(),
            attempts,
            energy_silence_intervals=energy_intervals,
        )

    assert resolved == original
    assert not any(item["kind"] == "uncertain-turn-silence-adjudication" for item in attempts)
    if detector is not None:
        assert detector.calls == 0


def test_uncertain_turn_silence_adjudication_preserves_nonempty_failed_retry(tmp_path):
    class _NonemptyFailedBackend(TranscriptionBackend):
        @property
        def identity(self):
            return "fixture:nonempty-failed-language-retry"

        def transcribe(self, audio, options=None):
            text = " ".join(["повтор"] * 20)
            return TranscriptionResult(
                (TranscriptSegment(1_000, 2_000, text, language="ru"),),
                text,
                "fixture",
                "nonempty-failed-language-retry",
                "test",
                "ru",
                3_000,
                False,
            )

    segment = TranscriptSegment(1_000, 2_000, "неясно", language="en", uncertain=True)
    original = TranscriptionResult(
        (segment,), segment.text, "fixture", "failed", "test", "en", 4_000, False
    )
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        _NonemptyFailedBackend(),
        tmp_path / "job",
        silence_verification_detector=detector,
    )
    with patch.object(
        coordinator,
        "_extract_language_span",
        side_effect=lambda source, marker, destination: destination.write_bytes(b"context"),
    ):
        resolved = coordinator._resolve_uncertain_turns(
            tmp_path / "audio.wav",
            original,
            coordinator.backend,
            TranscriptionOptions(),
            [],
            energy_silence_intervals=((1_000, 2_000),),
        )

    assert resolved == original
    assert detector.calls == 0


def test_uncertain_turn_silence_adjudication_preserves_mixed_script_without_retry(
    tmp_path,
):
    class _MustNotRunBackend(TranscriptionBackend):
        @property
        def identity(self):
            return "fixture:mixed-must-not-run"

        def transcribe(self, audio, options=None):
            raise AssertionError("mixed-script turn must remain for independent evidence")

    segment = TranscriptSegment(1_000, 2_000, "Или yes", language="en", uncertain=True)
    original = TranscriptionResult(
        (segment,), segment.text, "fixture", "mixed", "test", "en", 4_000, False
    )
    detector = _SpeechDetector()
    coordinator = AdaptiveLongFormCoordinator(
        _MustNotRunBackend(),
        tmp_path / "job",
        silence_verification_detector=detector,
    )

    resolved = coordinator._resolve_uncertain_turns(
        tmp_path / "audio.wav",
        original,
        coordinator.backend,
        TranscriptionOptions(),
        [],
        energy_silence_intervals=((1_000, 2_000),),
    )

    assert resolved == original
    assert detector.calls == 0


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
    backend = _IndependentBackend(duration_ms=6_000)
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
    assert manifest["source_integrity"]["policy"] == (
        "trusted-binding-plus-concurrent-full-rehash-v1"
    )
    assert result.provenance["source_integrity"]["source_sha256"] == (result.source_sha256)
    progress = json.loads((tmp_path / "job/progress.json").read_text())
    assert progress["status"] == "COMPLETE"


def test_processing_cap_change_or_full_source_change_rejects_checkpoint(tmp_path):
    source = tmp_path / "lesson.mov"
    source.write_bytes(b"source with tail")
    coordinator = AdaptiveLongFormCoordinator(
        _IndependentBackend(duration_ms=6_000),
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


def test_wrong_legacy_source_digest_fails_before_probe_or_backend(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    backend = _IndependentBackend(duration_ms=1_000)
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")

    with patch("dubbing.transcription.adaptive.probe_media") as probe:
        with pytest.raises(TranscriptionError, match="does not match expected"):
            coordinator.run(source, source_digest="0" * 64)

    probe.assert_not_called()
    assert backend.calls == 0


def test_source_digest_and_binding_are_mutually_exclusive(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    binding = create_source_binding(source)
    coordinator = AdaptiveLongFormCoordinator(_IndependentBackend(), tmp_path / "job")

    with pytest.raises(TranscriptionError, match="mutually exclusive"):
        coordinator.run(
            source,
            source_digest=binding.sha256,
            source_binding=binding,
        )


def test_source_binding_rejects_changed_stat_before_probe(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    binding = create_source_binding(source)
    source.write_bytes(b"changed")
    coordinator = AdaptiveLongFormCoordinator(_IndependentBackend(), tmp_path / "job")

    with patch("dubbing.transcription.adaptive.probe_media") as probe:
        with pytest.raises(TranscriptionError, match="stat identity"):
            coordinator.run(source, source_binding=binding)

    probe.assert_not_called()


def test_concurrent_source_verifier_overlaps_decode_and_is_awaited(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    verifier_started = Event()
    decode_released_verifier = Event()

    class _BarrierBackend(_IndependentBackend):
        def transcribe(self, audio, options=None):
            assert verifier_started.wait(2)
            decode_released_verifier.set()
            return super().transcribe(audio, options)

    backend = _BarrierBackend(duration_ms=1_000)
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")

    def verify(binding):
        verifier_started.set()
        assert decode_released_verifier.wait(2)
        return binding

    with patch("dubbing.transcription.adaptive.verify_source_binding", side_effect=verify):
        result, _ = _run_checkpoint_fixture(
            coordinator,
            source,
            1_000,
            lambda source, streams, chunk, output: output.write_bytes(b"chunk"),
        )

    assert backend.calls == 1
    assert result.provenance["source_integrity"]["concurrent_full_source_rehash_passed"]


def test_tail_mutation_fails_closed_but_preserves_chunks_and_prior_final(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source including unprocessed tail")

    class _MutatingBackend(_IndependentBackend):
        def transcribe(self, audio, options=None):
            result = super().transcribe(audio, options)
            source.write_bytes(b"source with a mutated unprocessed tail")
            return result

    job = tmp_path / "job"
    job.mkdir()
    prior_result = b'{"prior":"valid"}'
    prior_quality = b'{"prior":"valid-quality"}'
    (job / "result.json").write_bytes(prior_result)
    (job / "quality-report.json").write_bytes(prior_quality)
    coordinator = AdaptiveLongFormCoordinator(_MutatingBackend(), job)

    with pytest.raises(TranscriptionError, match="source integrity verification failed"):
        with (
            patch(
                "dubbing.transcription.adaptive.probe_media",
                return_value=MediaProbe(
                    2_000,
                    source.stat().st_size,
                    (AudioStream(0, "pcm", 1, 16_000),),
                ),
            ),
            patch(
                "dubbing.transcription.adaptive.detect_silence_intervals",
                return_value=(),
            ),
            patch.object(
                coordinator,
                "_extract",
                side_effect=lambda source, streams, chunk, output: output.write_bytes(
                    b"processed prefix"
                ),
            ),
        ):
            coordinator.run(source, processing_end_ms=1_000)

    assert (job / "chunks/000000.json").is_file()
    assert (job / "receipts/000000.json").is_file()
    assert (job / "result.json").read_bytes() == prior_result
    assert (job / "quality-report.json").read_bytes() == prior_quality
    progress = json.loads((job / "progress.json").read_text())
    assert progress["status"] == "INTEGRITY_FAILED"


def test_source_mutation_never_publishes_fresh_final_artifacts(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")

    class _MutatingBackend(_IndependentBackend):
        def transcribe(self, audio, options=None):
            result = super().transcribe(audio, options)
            source.write_bytes(b"mutated")
            return result

    coordinator = AdaptiveLongFormCoordinator(_MutatingBackend(), tmp_path / "job")
    with pytest.raises(TranscriptionError, match="source integrity verification failed"):
        _run_checkpoint_fixture(
            coordinator,
            source,
            1_000,
            lambda source, streams, chunk, output: output.write_bytes(b"chunk"),
        )

    assert not (tmp_path / "job/result.json").exists()
    assert not (tmp_path / "job/quality-report.json").exists()
    assert json.loads((tmp_path / "job/progress.json").read_text())["status"] == (
        "INTEGRITY_FAILED"
    )


def test_decode_failure_awaits_verifier_and_writes_terminal_failed_progress(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    verifier_started = Event()
    verifier_release = Event()
    verifier_completed = Event()

    class _FailingBackend(_IndependentBackend):
        def transcribe(self, audio, options=None):
            assert verifier_started.wait(2)
            verifier_release.set()
            raise RuntimeError("decoder failed")

    def verify(binding):
        verifier_started.set()
        assert verifier_release.wait(2)
        verifier_completed.set()
        return binding

    coordinator = AdaptiveLongFormCoordinator(_FailingBackend(), tmp_path / "job")
    with (
        patch("dubbing.transcription.adaptive.verify_source_binding", side_effect=verify),
        pytest.raises(RuntimeError, match="decoder failed"),
    ):
        _run_checkpoint_fixture(
            coordinator,
            source,
            1_000,
            lambda source, streams, chunk, output: output.write_bytes(b"chunk"),
        )

    assert verifier_completed.is_set()
    progress = json.loads((tmp_path / "job/progress.json").read_text())
    assert progress["status"] == "FAILED"
    assert not (tmp_path / "job/result.json").exists()


def test_verifier_start_failure_is_terminal_integrity_failure(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    coordinator = AdaptiveLongFormCoordinator(_IndependentBackend(), tmp_path / "job")

    with (
        patch(
            "dubbing.transcription.adaptive.ThreadPoolExecutor",
            side_effect=RuntimeError("thread unavailable"),
        ),
        pytest.raises(TranscriptionError, match="could not be started"),
    ):
        _run_checkpoint_fixture(
            coordinator,
            source,
            1_000,
            lambda source, streams, chunk, output: output.write_bytes(b"chunk"),
        )

    progress = json.loads((tmp_path / "job/progress.json").read_text())
    assert progress["status"] == "INTEGRITY_FAILED"
    assert not (tmp_path / "job/result.json").exists()


def _fail_one_final_stage_replace(target_name):
    original_replace = adaptive_module.os.replace
    failed = False

    def replace(source, destination):
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if not failed and source_path.suffix == ".stage" and destination_path.name == target_name:
            failed = True
            raise OSError(f"injected {target_name} replacement failure")
        return original_replace(source, destination)

    return replace


@pytest.mark.parametrize("failed_target", ["result.json", "quality-report.json"])
def test_final_pair_failure_restores_both_previous_artifacts(tmp_path, failed_target):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    job = tmp_path / "job"
    backend = _IndependentBackend(duration_ms=1_000)
    coordinator = AdaptiveLongFormCoordinator(backend, job)
    _run_checkpoint_fixture(
        coordinator,
        source,
        1_000,
        lambda source, streams, chunk, output: output.write_bytes(b"chunk"),
    )
    previous_result = b'{"generation":"previous-result"}\n'
    previous_quality = b'{"generation":"previous-quality"}\n'
    (job / "result.json").write_bytes(previous_result)
    (job / "quality-report.json").write_bytes(previous_quality)

    with (
        patch(
            "dubbing.transcription.adaptive.os.replace",
            side_effect=_fail_one_final_stage_replace(failed_target),
        ),
        pytest.raises(TranscriptionError, match="previous generation restored"),
    ):
        _run_checkpoint_fixture(
            coordinator,
            source,
            1_000,
            lambda source, streams, chunk, output: output.write_bytes(b"chunk"),
        )

    assert backend.calls == 1
    assert (job / "result.json").read_bytes() == previous_result
    assert (job / "quality-report.json").read_bytes() == previous_quality
    assert json.loads((job / "progress.json").read_text())["status"] == "FAILED"
    assert not list(job.glob(".*.stage"))
    assert not list(job.glob(".*.backup"))


@pytest.mark.parametrize("failed_target", ["result.json", "quality-report.json"])
def test_first_final_pair_publication_failure_leaves_neither_artifact(
    tmp_path,
    failed_target,
):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    job = tmp_path / "job"
    coordinator = AdaptiveLongFormCoordinator(
        _IndependentBackend(duration_ms=1_000),
        job,
    )

    with (
        patch(
            "dubbing.transcription.adaptive.os.replace",
            side_effect=_fail_one_final_stage_replace(failed_target),
        ),
        pytest.raises(TranscriptionError, match="previous generation restored"),
    ):
        _run_checkpoint_fixture(
            coordinator,
            source,
            1_000,
            lambda source, streams, chunk, output: output.write_bytes(b"chunk"),
        )

    assert not (job / "result.json").exists()
    assert not (job / "quality-report.json").exists()
    assert json.loads((job / "progress.json").read_text())["status"] == "FAILED"
    assert not list(job.glob(".*.stage"))
    assert not list(job.glob(".*.backup"))


def _run_checkpoint_fixture(coordinator, source, duration_ms, extract):
    with (
        patch(
            "dubbing.transcription.adaptive.probe_media",
            return_value=MediaProbe(
                duration_ms,
                source.stat().st_size,
                (AudioStream(0, "pcm", 1, 16_000),),
            ),
        ),
        patch(
            "dubbing.transcription.adaptive.detect_silence_intervals",
            return_value=(),
        ),
        patch.object(coordinator, "_extract", side_effect=extract),
    ):
        return coordinator.run(source)


def _canonical_document_hash(document):
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_valid_replay_reextracts_without_redecoding(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    backend = _IndependentBackend(duration_ms=1_000)
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")
    extractions = []

    def extract(source, streams, chunk, output):
        extractions.append(chunk.index)
        output.write_bytes(b"stable chunk zero")

    first, _ = _run_checkpoint_fixture(coordinator, source, 1_000, extract)
    replay, _ = _run_checkpoint_fixture(coordinator, source, 1_000, extract)

    assert replay == first
    assert backend.calls == 1
    assert extractions == [0, 0]
    receipt = json.loads((tmp_path / "job/receipts/000000.json").read_text())
    assert receipt["schema_version"] == "dubbing.adaptive-chunk-receipt.v1"
    assert receipt["full_source_sha256"] == hashlib.sha256(b"source").hexdigest()
    assert receipt["chunk"]["index"] == 0


@pytest.mark.parametrize("missing", ["checkpoint", "receipt"])
def test_replay_rejects_incomplete_checkpoint_pair(tmp_path, missing):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    backend = _IndependentBackend(duration_ms=1_000)
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")

    def extract(source, streams, chunk, output):
        output.write_bytes(b"stable chunk zero")

    _run_checkpoint_fixture(coordinator, source, 1_000, extract)
    target = (
        tmp_path / "job/chunks/000000.json"
        if missing == "checkpoint"
        else tmp_path / "job/receipts/000000.json"
    )
    target.unlink()

    with pytest.raises(TranscriptionError, match="checkpoint pair is incomplete"):
        _run_checkpoint_fixture(coordinator, source, 1_000, extract)
    assert backend.calls == 1


def test_replay_rejects_checkpoint_document_tampering(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    backend = _IndependentBackend(duration_ms=1_000)
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")

    def extract(source, streams, chunk, output):
        output.write_bytes(b"stable chunk zero")

    _run_checkpoint_fixture(coordinator, source, 1_000, extract)
    checkpoint = tmp_path / "job/chunks/000000.json"
    document = json.loads(checkpoint.read_text())
    document["text"] = "tampered"
    checkpoint.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(TranscriptionError, match="checkpoint document hash mismatch"):
        _run_checkpoint_fixture(coordinator, source, 1_000, extract)
    assert backend.calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "legacy"),
        ("coordinator_version", "forged"),
        ("quality_policy_version", "forged"),
        ("full_source_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("chunk", {"index": 0}),
    ],
)
def test_replay_rejects_receipt_binding_tampering(tmp_path, field, value):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    backend = _IndependentBackend(duration_ms=1_000)
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")

    def extract(source, streams, chunk, output):
        output.write_bytes(b"stable chunk zero")

    _run_checkpoint_fixture(coordinator, source, 1_000, extract)
    receipt_path = tmp_path / "job/receipts/000000.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(TranscriptionError, match=f"binding mismatch: 000000:{field}"):
        _run_checkpoint_fixture(coordinator, source, 1_000, extract)
    assert backend.calls == 1


def test_replay_rejects_copied_equal_duration_chunk_pair(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    backend = _IndependentBackend(duration_ms=1_000)
    chunks = (
        AdaptiveChunk(0, 0, 1_000, 0, 1_000, "fixture"),
        AdaptiveChunk(1, 1_000, 2_000, 1_000, 2_000, "fixture"),
    )
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        planner=_single_chunk_planner(chunks[0]),
    )
    coordinator.planner = SimpleNamespace(
        plan=lambda duration_ms, silence_centres, silence_intervals=(): chunks,
        to_dict=lambda: {"fixture": "two-equal-chunks"},
    )

    def extract(source, streams, chunk, output):
        output.write_bytes(f"chunk-{chunk.index}".encode())

    _run_checkpoint_fixture(coordinator, source, 2_000, extract)
    shutil.copyfile(
        tmp_path / "job/chunks/000000.json",
        tmp_path / "job/chunks/000001.json",
    )
    shutil.copyfile(
        tmp_path / "job/receipts/000000.json",
        tmp_path / "job/receipts/000001.json",
    )

    with pytest.raises(TranscriptionError, match="binding mismatch: 000001:chunk"):
        _run_checkpoint_fixture(coordinator, source, 2_000, extract)
    assert backend.calls == 2


def test_replay_reextracts_and_rejects_forged_audio_binding(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    backend = _IndependentBackend(duration_ms=1_000)
    chunks = (
        AdaptiveChunk(0, 0, 1_000, 0, 1_000, "fixture"),
        AdaptiveChunk(1, 1_000, 2_000, 1_000, 2_000, "fixture"),
    )
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
        planner=SimpleNamespace(
            plan=lambda duration_ms, silence_centres, silence_intervals=(): chunks,
            to_dict=lambda: {"fixture": "two-equal-chunks"},
        ),
    )

    def extract(source, streams, chunk, output):
        output.write_bytes(f"chunk-{chunk.index}".encode())

    _run_checkpoint_fixture(coordinator, source, 2_000, extract)
    checkpoint_zero = json.loads((tmp_path / "job/chunks/000000.json").read_text())
    receipt_zero = json.loads((tmp_path / "job/receipts/000000.json").read_text())
    checkpoint_one = tmp_path / "job/chunks/000001.json"
    receipt_one = tmp_path / "job/receipts/000001.json"
    forged_receipt = json.loads(receipt_one.read_text())
    checkpoint_one.write_text(json.dumps(checkpoint_zero), encoding="utf-8")
    forged_receipt["checkpoint_document_sha256"] = _canonical_document_hash(checkpoint_zero)
    forged_receipt["extracted_audio_sha256"] = receipt_zero["extracted_audio_sha256"]
    receipt_one.write_text(json.dumps(forged_receipt), encoding="utf-8")

    with pytest.raises(TranscriptionError, match="extracted audio hash mismatch: 000001"):
        _run_checkpoint_fixture(coordinator, source, 2_000, extract)
    assert backend.calls == 2


def test_initial_decode_rejects_backend_source_hash_mismatch(tmp_path):
    class _WrongSourceBackend(_IndependentBackend):
        def transcribe(self, audio, options=None):
            result = super().transcribe(audio, options)
            return replace(result, source_sha256="f" * 64)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    backend = _WrongSourceBackend(duration_ms=1_000)
    coordinator = AdaptiveLongFormCoordinator(backend, tmp_path / "job")

    def extract(source, streams, chunk, output):
        output.write_bytes(b"stable chunk zero")

    with pytest.raises(TranscriptionError, match="backend source hash mismatch"):
        _run_checkpoint_fixture(coordinator, source, 1_000, extract)
    assert not (tmp_path / "job/chunks/000000.json").exists()
    assert not (tmp_path / "job/receipts/000000.json").exists()


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
