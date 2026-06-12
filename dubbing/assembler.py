"""Assemble synthesised segments into one timeline-true WAV.

Each segment's audio is anchored at its SRT start time. Gaps between
subtitles become silence. Audio longer than its subtitle window is
time-compressed (linear resampling) to fit the window exactly; audio
shorter than the window plays at natural speed and the remainder of
the window is filled with silence. Where subtitle windows overlap
(two speakers at once), the overlapping audio is mixed — summed and
clamped to the 16-bit range — never shifted later, so the output
stays in sync with the source timeline from the first millisecond to
the last and never outlasts the final subtitle.
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

#: Hard ceiling on the rendered timeline. A hostile SRT a few hundred bytes
#: long can carry a 99-hour timestamp; without this cap the renderer would
#: try to allocate ~16 GB of silence for it (memory-exhaustion DoS through
#: the web upload, which limits *file size*, not *timestamp values*).
#: Four hours comfortably covers any real film or episode.
MAX_TIMELINE_MS = 4 * 60 * 60 * 1000


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


_PCM_MAX = 32767
_PCM_MIN = -32768


def _mix_into(out: array, start_frame: int, samples: array) -> None:
    """Write `samples` into `out` at the absolute `start_frame`, extending
    `out` with silence as needed. Frames that already carry audio (an
    overlapping subtitle window) are mixed by summation, clamped to the
    16-bit range — the timeline position of every segment is preserved."""
    end_frame = start_frame + len(samples)
    if end_frame > len(out):
        out.extend(_silence(end_frame - len(out)))
    for i, sample in enumerate(samples):
        j = start_frame + i
        mixed = out[j] + sample
        if mixed > _PCM_MAX:
            mixed = _PCM_MAX
        elif mixed < _PCM_MIN:
            mixed = _PCM_MIN
        out[j] = mixed


def assemble_timeline(
    timed: list[TimedSegment],
    results: list[TTSResult],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_timeline_ms: int = MAX_TIMELINE_MS,
) -> bytes:
    """Render aligned segments onto a single timeline-true WAV.

    Args:
        timed:   Aligned segments (same order as `results`).
        results: TTS render per segment, carrying the audio bytes.
        sample_rate: Output frame rate.
        max_timeline_ms: Reject timelines beyond this duration BEFORE any
            audio buffer is allocated (hostile-timestamp DoS guard).

    Returns:
        WAV bytes (mono, 16-bit) spanning the full subtitle timeline.

    Raises:
        ValueError: on timed/results length mismatch, a segment with no
            audio, audio that is not decodable 16-bit WAV, a negative
            start time, or a timeline exceeding `max_timeline_ms`.
    """
    if len(timed) != len(results):
        raise ValueError(
            f"timed and results must have the same length ({len(timed)} vs {len(results)})"
        )

    # Validate the timeline bounds up front, before decoding or allocating
    # anything: timestamps come straight from the (untrusted) SRT file.
    for ts in timed:
        if ts.start_ms < 0:
            raise ValueError(
                f"segment {ts.segment.entry.index} has a negative start time ({ts.start_ms} ms)"
            )
        if max(ts.start_ms, ts.end_ms) > max_timeline_ms:
            raise ValueError(
                f"segment {ts.segment.entry.index} ends at "
                f"{max(ts.start_ms, ts.end_ms)} ms, beyond the maximum supported "
                f"timeline of {max_timeline_ms} ms"
            )

    # Render in timeline order: SRT files are not guaranteed sorted, and an
    # out-of-order entry would otherwise be appended at the current write
    # head instead of its own start time, desynchronising everything after it.
    out = array("h")
    for ts, res in sorted(zip(timed, results), key=lambda pair: pair[0].start_ms):
        if not res.audio_bytes:
            raise ValueError(f"segment {ts.segment.entry.index} produced no audio data")

        samples, src_rate = _decode_wav(res.audio_bytes)
        samples = _resample(samples, src_rate, sample_rate)

        # Anchor at the absolute SRT start, NEVER the current write head:
        # overlapping subtitle windows (two speakers at once) mix in place
        # rather than shifting later and stretching the timeline.
        start_frame = _ms_to_frames(ts.start_ms, sample_rate)

        window_ms = ts.end_ms - ts.start_ms
        if window_ms > 0:
            window_frames = _ms_to_frames(window_ms, sample_rate)
            if len(samples) > window_frames:
                # Overlong render: compress to fit the subtitle window.
                samples = _fit(samples, window_frames)
            _mix_into(out, start_frame, samples)
            # Short render: pad with silence to the window's absolute end.
            end_frame = start_frame + window_frames
            if end_frame > len(out):
                out.extend(_silence(end_frame - len(out)))
        else:
            # Degenerate window (plain-text fallback): natural duration.
            _mix_into(out, start_frame, samples)

    return _encode_wav(out, sample_rate)
