import json
from unittest.mock import patch

from dubbing.transcription.adaptive import (
    AdaptiveChunk,
    AdaptiveChunkPlanner,
    AdaptiveLongFormCoordinator,
    AudioStream,
    MediaProbe,
    reconcile_chunks,
)
from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptionOptions,
    TranscriptionResult,
)


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


def test_planner_chooses_silence_near_target_and_bounds_chunks():
    planner = AdaptiveChunkPlanner(
        target_seconds=120, minimum_seconds=60, maximum_seconds=180, overlap_seconds=2
    )
    chunks = planner.plan(400_000, (110_000, 250_000))
    assert [item.end_ms for item in chunks] == [110_000, 250_000, 400_000]
    assert all(item.end_ms - item.start_ms <= 180_000 for item in chunks)
    assert chunks[1].extract_start_ms == 108_000


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


def test_coordinator_retries_only_failed_span_and_checkpoints(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    backend = _RetryingBackend()
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        tmp_path / "job",
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
    assert backend.calls == 2
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
    assert len(targeted) == 1
    assert targeted[0]["source_end_ms"] == 20_000


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
