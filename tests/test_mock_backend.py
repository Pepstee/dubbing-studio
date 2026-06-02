from __future__ import annotations

import sys
import unittest.mock

import pytest

from dubbing.backends.mock import MockTTSBackend
from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, SRTEntry, ProsodyTag, TTSResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entry(index: int = 1, start_ms: int = 0, end_ms: int = 1000, text: str = "Hello") -> SRTEntry:
    return SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text)


def make_segment(text: str = "Hello", tags: list[ProsodyTag] | None = None, language: str = "en") -> Segment:
    return Segment(entry=make_entry(text=text), tags=tags or [], language=language)


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

def test_mock_backend_is_tts_backend():
    assert issubclass(MockTTSBackend, TTSBackend)


def test_instance_is_tts_backend():
    backend = MockTTSBackend()
    assert isinstance(backend, TTSBackend)


# ---------------------------------------------------------------------------
# synthesize() — return type and count
# ---------------------------------------------------------------------------

def test_synthesize_returns_list():
    backend = MockTTSBackend()
    result = backend.synthesize([make_segment()])
    assert isinstance(result, list)


def test_single_segment_yields_one_result():
    backend = MockTTSBackend()
    segments = [make_segment()]
    results = backend.synthesize(segments)
    assert len(results) == 1


def test_three_segments_yield_three_results():
    backend = MockTTSBackend()
    segments = [make_segment(f"text {i}") for i in range(3)]
    results = backend.synthesize(segments)
    assert len(results) == 3


def test_empty_segment_list_yields_empty_list():
    backend = MockTTSBackend()
    results = backend.synthesize([])
    assert results == []


def test_results_are_ttsresult_instances():
    backend = MockTTSBackend()
    results = backend.synthesize([make_segment(), make_segment("World")])
    assert all(isinstance(r, TTSResult) for r in results)


# ---------------------------------------------------------------------------
# synthesize() — audio_bytes
# ---------------------------------------------------------------------------

def test_audio_bytes_is_bytes():
    backend = MockTTSBackend()
    result = backend.synthesize([make_segment()])[0]
    assert isinstance(result.audio_bytes, bytes)


def test_audio_bytes_is_empty_bytes():
    backend = MockTTSBackend()
    result = backend.synthesize([make_segment()])[0]
    assert result.audio_bytes == b""


def test_audio_bytes_empty_for_all_segments():
    backend = MockTTSBackend()
    segments = [make_segment("Hello"), make_segment("World"), make_segment("!")]
    for res in backend.synthesize(segments):
        assert res.audio_bytes == b""


# ---------------------------------------------------------------------------
# synthesize() — duration_ms
# ---------------------------------------------------------------------------

_MS_PER_CHAR = 60  # mirrors the implementation constant


def test_duration_ms_matches_char_count():
    text = "Hello"
    backend = MockTTSBackend()
    result = backend.synthesize([make_segment(text)])[0]
    assert result.duration_ms == len(text) * _MS_PER_CHAR


def test_duration_ms_zero_for_empty_text():
    backend = MockTTSBackend()
    result = backend.synthesize([make_segment("")])[0]
    assert result.duration_ms == 0


def test_duration_ms_single_char():
    backend = MockTTSBackend()
    result = backend.synthesize([make_segment("A")])[0]
    assert result.duration_ms == _MS_PER_CHAR


def test_duration_ms_long_text():
    text = "x" * 100
    backend = MockTTSBackend()
    result = backend.synthesize([make_segment(text)])[0]
    assert result.duration_ms == 100 * _MS_PER_CHAR


def test_duration_ms_unicode_text():
    text = "héllo"
    backend = MockTTSBackend()
    result = backend.synthesize([make_segment(text)])[0]
    assert result.duration_ms == len(text) * _MS_PER_CHAR


def test_duration_ms_each_segment_independent():
    backend = MockTTSBackend()
    texts = ["Hi", "Hello there", "Goodbye"]
    segments = [make_segment(t) for t in texts]
    results = backend.synthesize(segments)
    for res, text in zip(results, texts):
        assert res.duration_ms == len(text) * _MS_PER_CHAR


# ---------------------------------------------------------------------------
# synthesize() — segment identity
# ---------------------------------------------------------------------------

def test_result_segment_is_input_segment():
    backend = MockTTSBackend()
    seg = make_segment("test")
    result = backend.synthesize([seg])[0]
    assert result.segment is seg


def test_result_segments_order_matches_input():
    backend = MockTTSBackend()
    segments = [make_segment(f"text {i}") for i in range(5)]
    results = backend.synthesize(segments)
    for i, (res, seg) in enumerate(zip(results, segments)):
        assert res.segment is seg


# ---------------------------------------------------------------------------
# No filesystem or network calls
# ---------------------------------------------------------------------------

def test_synthesize_does_not_open_files(monkeypatch):
    calls = []
    original_open = open

    def patched_open(*args, **kwargs):
        calls.append(args)
        return original_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)
    backend = MockTTSBackend()
    backend.synthesize([make_segment("test")])
    assert calls == [], f"synthesize() opened files unexpectedly: {calls}"


def test_synthesize_does_not_use_socket(monkeypatch):
    import socket
    original_connect = socket.socket.connect
    attempts = []

    def patched_connect(self, *args, **kwargs):
        attempts.append(args)
        return original_connect(self, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", patched_connect)
    backend = MockTTSBackend()
    backend.synthesize([make_segment("test")])
    assert attempts == [], "synthesize() attempted a network connection"


def test_synthesize_does_not_call_subprocess(monkeypatch):
    import subprocess
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = MockTTSBackend()
    backend.synthesize([make_segment("test")])
    assert calls == [], "synthesize() called subprocess.run unexpectedly"


# ---------------------------------------------------------------------------
# Tags and language are passed through unchanged
# ---------------------------------------------------------------------------

def test_prosody_tags_preserved_in_result():
    tag = ProsodyTag(name="emotion", value="happy")
    seg = Segment(entry=make_entry(text="Hello"), tags=[tag], language="en")
    backend = MockTTSBackend()
    result = backend.synthesize([seg])[0]
    assert result.segment.tags == [tag]


def test_language_preserved_in_result():
    seg = Segment(entry=make_entry(text="Hola"), tags=[], language="es")
    backend = MockTTSBackend()
    result = backend.synthesize([seg])[0]
    assert result.segment.language == "es"
