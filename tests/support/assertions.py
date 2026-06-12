from __future__ import annotations

import io
import wave
from typing import Sequence

import pytest

from dubbing.aligner import TimedSegment
from dubbing.models import ProsodyTag


def _decode_samples(data: bytes) -> tuple[list[int], int]:
    with wave.open(io.BytesIO(data)) as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = [
        int.from_bytes(raw[i: i + 2], "little", signed=True)
        for i in range(0, len(raw), 2)
    ]
    return samples, rate


def assert_wav_duration(data: bytes, expected_ms: int, tolerance_ms: int = 2) -> None:
    """Assert *data* is a valid WAV whose duration is within *tolerance_ms* of *expected_ms*."""
    with wave.open(io.BytesIO(data)) as wf:
        actual_ms = int(wf.getnframes() * 1000 / wf.getframerate())
    assert abs(actual_ms - expected_ms) <= tolerance_ms, (
        f"WAV duration {actual_ms}ms not within {tolerance_ms}ms of expected {expected_ms}ms"
    )


def assert_wav_format(
    data: bytes,
    *,
    channels: int = 1,
    sampwidth: int = 2,
    rate: int = 22050,
) -> None:
    """Assert exact PCM format parameters on a WAV buffer."""
    with wave.open(io.BytesIO(data)) as wf:
        assert wf.getnchannels() == channels, (
            f"channels: expected {channels}, got {wf.getnchannels()}"
        )
        assert wf.getsampwidth() == sampwidth, (
            f"sampwidth: expected {sampwidth}, got {wf.getsampwidth()}"
        )
        assert wf.getframerate() == rate, (
            f"framerate: expected {rate}, got {wf.getframerate()}"
        )


def assert_silence_region(data: bytes, start_s: float, end_s: float) -> None:
    """Assert every sample in the half-open interval [start_s, end_s) seconds is zero."""
    samples, rate = _decode_samples(data)
    lo, hi = int(rate * start_s), int(rate * end_s)
    region = samples[lo:hi]
    assert region, f"region [{start_s}s, {end_s}s) is empty — check offsets against WAV length"
    assert all(s == 0 for s in region), (
        f"expected silence in [{start_s}s, {end_s}s) but found non-zero samples"
    )


def assert_audio_region(data: bytes, start_s: float, end_s: float) -> None:
    """Assert at least one non-zero sample exists in [start_s, end_s) seconds."""
    samples, rate = _decode_samples(data)
    lo, hi = int(rate * start_s), int(rate * end_s)
    region = samples[lo:hi]
    assert region, f"region [{start_s}s, {end_s}s) is empty — check offsets against WAV length"
    assert any(s != 0 for s in region), (
        f"expected non-zero audio in [{start_s}s, {end_s}s) but found only silence"
    )


def assert_timed_segment(
    ts: TimedSegment,
    *,
    start_ms: int,
    end_ms: int,
    stretch_ratio: float | None = None,
) -> None:
    """Assert exact boundary values on a TimedSegment."""
    assert ts.start_ms == start_ms, f"start_ms: expected {start_ms}, got {ts.start_ms}"
    assert ts.end_ms == end_ms, f"end_ms: expected {end_ms}, got {ts.end_ms}"
    if stretch_ratio is not None:
        assert ts.stretch_ratio == pytest.approx(stretch_ratio), (
            f"stretch_ratio: expected {stretch_ratio}, got {ts.stretch_ratio}"
        )


def assert_prosody_tags(
    tags: list[ProsodyTag],
    *,
    names: Sequence[str],
    values: dict[str, str] | None = None,
) -> None:
    """Assert *tags* contains exactly the given prosody tag names (and optional per-name values)."""
    actual_names = {t.name for t in tags}
    expected_names = set(names)
    assert actual_names == expected_names, (
        f"prosody tag names: expected {sorted(expected_names)}, got {sorted(actual_names)}"
    )
    if values:
        by_name = {t.name: t.value for t in tags}
        for name, val in values.items():
            assert by_name.get(name) == val, (
                f"prosody tag '{name}': expected value {val!r}, got {by_name.get(name)!r}"
            )
