from __future__ import annotations

import io
import wave

from dubbing.models import ProsodyTag, Segment, SRTEntry, TTSResult


def make_srt_entry(
    index: int = 1,
    start_ms: int = 0,
    end_ms: int = 1000,
    text: str = "Hello",
) -> SRTEntry:
    return SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text)


def make_segment(
    index: int = 1,
    start_ms: int = 0,
    end_ms: int = 1000,
    text: str = "Hello",
    tags: list[ProsodyTag] | None = None,
    language: str = "",
) -> Segment:
    return Segment(
        entry=make_srt_entry(index=index, start_ms=start_ms, end_ms=end_ms, text=text),
        tags=tags or [],
        language=language,
    )


def make_wav(duration_ms: int, sample_value: int = 1000, rate: int = 22050) -> bytes:
    """Return real 16-bit mono WAV bytes of exactly *duration_ms* milliseconds."""
    frames = int(rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(sample_value.to_bytes(2, "little", signed=True) * frames)
    return buf.getvalue()


def make_tts_result(
    segment: Segment | None = None,
    duration_ms: int = 1000,
    sample_value: int = 1000,
) -> TTSResult:
    seg = segment if segment is not None else make_segment(end_ms=duration_ms)
    return TTSResult(
        segment=seg,
        audio_bytes=make_wav(duration_ms, sample_value=sample_value),
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# SRT text fixtures (importable constants)
# ---------------------------------------------------------------------------

SRT_SINGLE = """\
1
00:00:01,000 --> 00:00:03,000
Hello world

"""

SRT_MULTI = """\
1
00:00:00,000 --> 00:00:02,000
First subtitle

2
00:00:03,000 --> 00:00:05,000
Second subtitle

3
00:00:06,000 --> 00:00:08,000
Third subtitle

"""

SRT_PROSODY = """\
1
00:00:00,000 --> 00:00:02,000
<emotion:happy>Hello world

"""

SRT_MULTI_TAG = """\
1
00:00:00,000 --> 00:00:03,000
<rate:slow><emotion:sad>Goodbye

"""
