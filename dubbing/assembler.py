"""Assemble synthesised segments into one timeline-true WAV.

Each segment's audio is placed at its SRT start time. Gaps between
subtitles become silence. Audio longer than its subtitle window is
time-compressed (linear resampling) to fit the window exactly; audio
shorter than the window plays at natural speed and the remainder of
the window is filled with silence. The output therefore stays in sync
with the source timeline from the first millisecond to the last.
"""

from __future__ import annotations

import io
import sys
import wave
from array import array

from dubbing.aligner import TimedSegment
from dubbing.models import TTSResult

DEFAULT_SAMPLE_RATE = 22050
_SAMPLE_WIDTH = 2  # 16-bit PCM
_CHANNELS = 1


def _ms_to_frames(ms: int, sample_rate: int) -> int:
    return int(round(ms * sample_rate / 1000))


def _decode_wav(data: bytes) -> tuple[array, int]:
    """Decode WAV bytes to mono 16-bit samples plus the source frame rate."""
    try:
        with wave.open(io.BytesIO(data)) as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"segment audio is not valid WAV data: {exc}") from exc

    if width != _SAMPLE_WIDTH:
        raise ValueError(f"unsupported WAV sample width: {width * 8}-bit (expected 16-bit PCM)")

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()

    if channels > 1:
        mono = array("h", bytes(_SAMPLE_WIDTH * (len(samples) // channels)))
        for i in range(len(mono)):
            mono[i] = sum(samples[i * channels:(i + 1) * channels]) // channels
        samples = mono

    return samples, rate


def _fit(samples: array, target_frames: int) -> array:
    """Linearly resample `samples` to exactly `target_frames` frames."""
    n = len(samples)
    if target_frames <= 0 or n == 0:
        return array("h")
    if n == target_frames:
        return samples

    out = array("h", bytes(_SAMPLE_WIDTH * target_frames))
    step = (n - 1) / (target_frames - 1) if target_frames > 1 else 0.0
    for i in range(target_frames):
        pos = i * step
        j = int(pos)
        frac = pos - j
        a = samples[j]
        b = samples[j + 1] if j + 1 < n else a
        out[i] = int(a + (b - a) * frac)
    return out


def _resample(samples: array, src_rate: int, dst_rate: int) -> array:
    if src_rate == dst_rate:
        return samples
    return _fit(samples, int(round(len(samples) * dst_rate / src_rate)))


def _encode_wav(samples: array, sample_rate: int) -> bytes:
    if sys.byteorder == "big":
        samples = array("h", samples)
        samples.byteswap()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _silence(frames: int) -> array:
    return array("h", bytes(_SAMPLE_WIDTH * max(0, frames)))


def assemble_timeline(
    timed: list[TimedSegment],
    results: list[TTSResult],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """Render aligned segments onto a single timeline-true WAV.

    Args:
        timed:   Aligned segments (same order as `results`).
        results: TTS render per segment, carrying the audio bytes.
        sample_rate: Output frame rate.

    Returns:
        WAV bytes (mono, 16-bit) spanning the full subtitle timeline.

    Raises:
        ValueError: on timed/results length mismatch, a segment with no
            audio, or audio that is not decodable 16-bit WAV.
    """
    if len(timed) != len(results):
        raise ValueError(
            f"timed and results must have the same length ({len(timed)} vs {len(results)})"
        )

    out = array("h")
    for ts, res in zip(timed, results):
        if not res.audio_bytes:
            raise ValueError(f"segment {ts.segment.entry.index} produced no audio data")

        samples, src_rate = _decode_wav(res.audio_bytes)
        samples = _resample(samples, src_rate, sample_rate)

        start_frame = _ms_to_frames(ts.start_ms, sample_rate)
        if start_frame > len(out):
            out.extend(_silence(start_frame - len(out)))

        window_ms = ts.end_ms - ts.start_ms
        if window_ms > 0:
            window_frames = _ms_to_frames(window_ms, sample_rate)
            if len(samples) > window_frames:
                # Overlong render: compress to fit the subtitle window.
                samples = _fit(samples, window_frames)
            out.extend(samples)
            # Short render: pad with silence to the window's absolute end.
            end_frame = start_frame + window_frames
            if end_frame > len(out):
                out.extend(_silence(end_frame - len(out)))
        else:
            # Degenerate window (plain-text fallback): natural duration.
            out.extend(samples)

    return _encode_wav(out, sample_rate)
