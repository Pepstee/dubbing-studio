import json
import wave
from array import array
from unittest.mock import patch

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
from dubbing.transcription.models import (
    DecodeDiagnostics,
    TranscriptSegment,
    TranscriptionOptions,
    TranscriptionResult,
)



def _pcm_wave(path, samples, sample_rate=16000):
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(array("h", samples).tobytes())


def test_native_pcm_wave_probe_silence_and_extraction_without_ffmpeg(tmp_path):
    source = tmp_path / "source.wav"
    _pcm_wave(source, [10000, -10000] * 4000 + [0] * 16000 + [10000, -10000] * 4000)

    with patch("dubbing.transcription.adaptive.shutil.which", return_value=None), patch(
        "dubbing.transcription.adaptive.ffmpeg_executable", return_value=None
    ):
        probe = probe_media(source)
        silences = detect_silence_intervals(source)
        output = tmp_path / "chunk.wav"
        AdaptiveLongFormCoordinator._extract(
            source,
            probe.audio_streams[0],
            AdaptiveChunk(0, 500, 1500, 500, 1500, "silence"),
            output,
        )

    assert probe.duration_ms == 2000
    assert probe.audio_streams[0].codec == "pcm_s16le"
    assert silences == ((500, 1500),)
    with wave.open(str(output), "rb") as extracted:
        assert extracted.getnframes() == 16000


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


def test_overlap_reconciliation_drops_duplicate_boundary_segment():
    chunks = [
        AdaptiveChunk(0, 0, 10_000, 0, 12_000, "silence"),
        AdaptiveChunk(1, 10_000, 20_000, 8_000, 20_000, "end-of-media"),
    ]
    first = TranscriptionResult(
        (TranscriptSegment(9_000, 10_500, "same boundary"),),
        "same boundary", "x", "x", "x", "en", 12_000, False,
    )
    second = TranscriptionResult(
        (TranscriptSegment(1_000, 2_500, "same boundary"),),
        "same boundary", "x", "x", "x", "en", 12_000, False,
    )
    assert len(reconcile_chunks(list(zip(chunks, (first, second))))) == 1


def test_overlap_context_is_clipped_to_non_overlapping_logical_chunks():
    chunks = [
        AdaptiveChunk(0, 0, 10_000, 0, 12_000, "silence"),
        AdaptiveChunk(1, 10_000, 20_000, 8_000, 20_000, "end-of-media"),
    ]
    first = TranscriptionResult(
        (TranscriptSegment(8_500, 10_500, "before boundary"),),
        "before boundary", "x", "x", "x", "en", 12_000, False,
    )
    second = TranscriptionResult(
        (TranscriptSegment(1_500, 3_000, "after boundary"),),
        "after boundary", "x", "x", "x", "en", 12_000, False,
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

    with patch(
        "dubbing.transcription.adaptive.probe_media",
        return_value=MediaProbe(60_000, 6, (AudioStream(0, "pcm", 2, 48_000),)),
    ), patch(
        "dubbing.transcription.adaptive.detect_silence_centres", return_value=()
    ), patch.object(coordinator, "_extract", side_effect=extract), patch.object(
        coordinator, "_extract_language_span", side_effect=extract_retry
    ):
        result, quality = coordinator.run(source)
        coordinator.run(source)
    assert backend.calls == 4
    assert retry_backend.calls == 3
    assert result.text == "clean ordinary phrase"
    assert quality["status"] == "PASS"
    assert (tmp_path / "job" / "chunks" / "000000.json").is_file()
    receipt = json.loads(
        (tmp_path / "job" / "receipts" / "000000.json").read_text()
    )
    targeted = [
        attempt
        for attempt in receipt["attempts"]
        if attempt.get("kind") == "targeted-span-redecode"
    ]
    assert len(targeted) == 6
    assert targeted[0]["source_end_ms"] == 20_000
    adjudications = [
        attempt
        for attempt in receipt["attempts"]
        if attempt.get("kind") == "targeted-span-adjudication"
    ]
    assert adjudications == [
        {
            "kind": "targeted-span-adjudication",
            "source_start_ms": 0,
            "source_end_ms": 20_000,
            "status": "CONSENSUS_PASS",
            "primary_backend": "fixture:persistent",
            "selected_backend": "fixture:independent",
            "selected_language": None,
            "independent_backend_count": 1,
            "agreement": 1.0,
            "agreement_threshold": 0.75,
            "selected_uncertain": False,
        }
    ]


def test_targeted_retry_without_independent_backend_fails_closed(tmp_path):
    coordinator = AdaptiveLongFormCoordinator(
        _RetryingBackend(), tmp_path / "job"
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

    assert replacement is None
    assert attempts[-1]["status"] == "NO_HEALTHY_INDEPENDENT_CANDIDATE"


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
    assert attempts[-1]["status"] == "INDEPENDENT_DISAGREEMENT"
    assert attempts[-1]["agreement"] == 0.0


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
                        diagnostics=DecodeDiagnostics(
                            avg_log_probability=-0.2 if forced else -0.1
                        ),
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
        item
        for item in receipt["attempts"]
        if item["kind"] == "detected-chunk-language-redecode"
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
