from __future__ import annotations

import shutil

import pytest

from dubbing.backends.base import TTSBackend
from dubbing.backends.say import SayTTSBackend
from dubbing.models import ProsodyTag, Segment, SRTEntry, TTSResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entry(index: int = 1, start_ms: int = 0, end_ms: int = 1000, text: str = "Hello") -> SRTEntry:
    return SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text)


def make_segment(text: str = "Hello", tags: list[ProsodyTag] | None = None, language: str = "en") -> Segment:
    return Segment(entry=make_entry(text=text), tags=tags or [], language=language)


@pytest.fixture
def backend(monkeypatch):
    # Force silence path so unit tests are fast (no subprocess calls)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    return SayTTSBackend()


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

def test_say_backend_is_tts_backend():
    assert issubclass(SayTTSBackend, TTSBackend)


def test_instance_is_tts_backend(backend):
    assert isinstance(backend, TTSBackend)


# ---------------------------------------------------------------------------
# synthesize() — return type and count
# ---------------------------------------------------------------------------

def test_synthesize_returns_list(backend):
    result = backend.synthesize([make_segment()])
    assert isinstance(result, list)


def test_single_segment_yields_one_result(backend):
    results = backend.synthesize([make_segment()])
    assert len(results) == 1


def test_three_segments_yield_three_results(backend):
    segments = [make_segment(f"text {i}") for i in range(3)]
    results = backend.synthesize(segments)
    assert len(results) == 3


def test_empty_segment_list_yields_empty_list(backend):
    results = backend.synthesize([])
    assert results == []


def test_results_are_ttsresult_instances(backend):
    results = backend.synthesize([make_segment(), make_segment("World")])
    assert all(isinstance(r, TTSResult) for r in results)


# ---------------------------------------------------------------------------
# synthesize() — audio_bytes
# ---------------------------------------------------------------------------

def test_audio_bytes_is_bytes(backend):
    result = backend.synthesize([make_segment()])[0]
    assert isinstance(result.audio_bytes, bytes)


def test_audio_bytes_is_non_empty(backend):
    result = backend.synthesize([make_segment()])[0]
    assert len(result.audio_bytes) > 0


def test_audio_bytes_non_empty_for_all_segments(backend):
    segments = [make_segment("Hello"), make_segment("World"), make_segment("!")]
    for res in backend.synthesize(segments):
        assert len(res.audio_bytes) > 0


# ---------------------------------------------------------------------------
# synthesize() — duration_ms
# ---------------------------------------------------------------------------

def test_duration_ms_positive(backend):
    result = backend.synthesize([make_segment("Hello")])[0]
    assert result.duration_ms > 0


def test_duration_ms_each_segment_positive(backend):
    texts = ["Hi", "Hello there", "Goodbye"]
    segments = [make_segment(t) for t in texts]
    results = backend.synthesize(segments)
    for res in results:
        assert res.duration_ms > 0


# ---------------------------------------------------------------------------
# synthesize() — segment identity
# ---------------------------------------------------------------------------

def test_result_segment_is_input_segment(backend):
    seg = make_segment("test")
    result = backend.synthesize([seg])[0]
    assert result.segment is seg


def test_result_segments_order_matches_input(backend):
    segments = [make_segment(f"text {i}") for i in range(5)]
    results = backend.synthesize(segments)
    for res, seg in zip(results, segments):
        assert res.segment is seg


# ---------------------------------------------------------------------------
# Tags and language are passed through unchanged
# ---------------------------------------------------------------------------

def test_prosody_tags_preserved_in_result(backend):
    tag = ProsodyTag(name="emotion", value="happy")
    seg = Segment(entry=make_entry(text="Hello"), tags=[tag], language="en")
    result = backend.synthesize([seg])[0]
    assert result.segment.tags == [tag]


def test_language_preserved_in_result(backend):
    seg = Segment(entry=make_entry(text="Hola"), tags=[], language="es")
    result = backend.synthesize([seg])[0]
    assert result.segment.language == "es"
