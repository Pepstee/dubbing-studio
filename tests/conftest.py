from __future__ import annotations

import pytest

from dubbing.models import Segment

from tests.support.builders import (
    SRT_MULTI,
    SRT_MULTI_TAG,
    SRT_PROSODY,
    SRT_SINGLE,
    make_segment,
    make_wav,
)


@pytest.fixture()
def srt_single() -> str:
    return SRT_SINGLE


@pytest.fixture()
def srt_multi() -> str:
    return SRT_MULTI


@pytest.fixture()
def srt_prosody() -> str:
    return SRT_PROSODY


@pytest.fixture()
def srt_multi_tag() -> str:
    return SRT_MULTI_TAG


@pytest.fixture()
def single_segment() -> Segment:
    return make_segment(start_ms=0, end_ms=2000)


@pytest.fixture()
def three_segments() -> list[Segment]:
    return [
        make_segment(index=i, start_ms=(i - 1) * 2000, end_ms=i * 2000, text=f"Segment {i}")
        for i in range(1, 4)
    ]


@pytest.fixture()
def silent_wav_1s() -> bytes:
    return make_wav(1000, sample_value=0)


@pytest.fixture()
def audio_wav_1s() -> bytes:
    return make_wav(1000, sample_value=1000)
