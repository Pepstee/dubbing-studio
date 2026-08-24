from __future__ import annotations

import io
import wave

import pytest

from dubbing.aligner import TimelineAligner
from dubbing.assembler import assemble_timeline
from dubbing.models import Segment, SRTEntry, TTSResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wav(duration_ms: int, value: int = 1000, rate: int = 22050) -> bytes:
    """Real 16-bit mono WAV filled with a constant non-zero sample value."""
    frames = int(rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(value.to_bytes(2, "little", signed=True) * frames)
    return buf.getvalue()


def _segment(index: int, start_ms: int, end_ms: int, text: str = "x") -> Segment:
    return Segment(
        entry=SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text),
        tags=[],
        language="",
    )


def _timed_and_results(spec: list[tuple[int, int, int]]) -> tuple[list, list]:
    """spec: (start_ms, end_ms, tts_duration_ms) per segment."""
    segments = [_segment(i + 1, s, e) for i, (s, e, _) in enumerate(spec)]
    durations = [d for _, _, d in spec]
    timed = TimelineAligner().align(segments, durations)
    results = [
        TTSResult(segment=seg, audio_bytes=_wav(d), duration_ms=d)
        for seg, d in zip(segments, durations)
    ]
    return timed, results


def _decode(data: bytes) -> tuple[list[int], int]:
    with wave.open(io.BytesIO(data)) as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = [
        int.from_bytes(raw[i:i + 2], "little", signed=True)
        for i in range(0, len(raw), 2)
    ]
    return samples, rate


def _duration_ms(data: bytes) -> int:
    samples, rate = _decode(data)
    return int(len(samples) * 1000 / rate)


# ---------------------------------------------------------------------------
# Timeline span
# ---------------------------------------------------------------------------

class TestTimelineSpan:
    def test_output_spans_to_last_subtitle_end(self):
        timed, results = _timed_and_results([(0, 1000, 800), (5000, 7000, 1500)])
        assert abs(_duration_ms(assemble_timeline(timed, results)) - 7000) <= 2

    def test_leading_offset_becomes_silence(self):
        timed, results = _timed_and_results([(3000, 4000, 1000)])
        samples, rate = _decode(assemble_timeline(timed, results))
        lead = samples[: int(rate * 2.9)]  # first 2.9s must be silent
        assert all(s == 0 for s in lead)

    def test_gap_between_subtitles_is_silence(self):
        timed, results = _timed_and_results([(0, 1000, 1000), (4000, 5000, 1000)])
        samples, rate = _decode(assemble_timeline(timed, results))
        gap = samples[int(rate * 1.1): int(rate * 3.9)]  # 1.1s–3.9s is the gap
        assert all(s == 0 for s in gap)

    def test_audio_present_inside_subtitle_window(self):
        timed, results = _timed_and_results([(2000, 3000, 1000)])
        samples, rate = _decode(assemble_timeline(timed, results))
        window = samples[int(rate * 2.1): int(rate * 2.9)]
        assert any(s != 0 for s in window)

    def test_empty_input_yields_zero_length_wav(self):
        data = assemble_timeline([], [])
        assert _duration_ms(data) == 0

    def test_output_is_valid_mono_16bit_wav(self):
        timed, results = _timed_and_results([(0, 1000, 500)])
        with wave.open(io.BytesIO(assemble_timeline(timed, results))) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2


# ---------------------------------------------------------------------------
# Stretch / pad behaviour
# ---------------------------------------------------------------------------

class TestStretchAndPad:
    def test_overlong_audio_is_compressed_into_window(self):
        # 3s of speech in a 1s window: output still ends at 1000ms.
        timed, results = _timed_and_results([(0, 1000, 3000)])
        assert abs(_duration_ms(assemble_timeline(timed, results)) - 1000) <= 2

    def test_overlong_audio_does_not_bleed_into_next_window(self):
        timed, results = _timed_and_results([(0, 1000, 5000), (2000, 3000, 500)])
        samples, rate = _decode(assemble_timeline(timed, results))
        gap = samples[int(rate * 1.1): int(rate * 1.9)]
        assert all(s == 0 for s in gap)

    def test_short_audio_plays_naturally_then_silence(self):
        # 500ms of speech in a 2s window: speech first, silence after.
        timed, results = _timed_and_results([(0, 2000, 500)])
        samples, rate = _decode(assemble_timeline(timed, results))
        speech = samples[: int(rate * 0.4)]
        tail = samples[int(rate * 0.6):]
        assert any(s != 0 for s in speech)
        assert all(s == 0 for s in tail)

    def test_short_audio_window_still_fully_spanned(self):
        timed, results = _timed_and_results([(0, 2000, 500)])
        assert abs(_duration_ms(assemble_timeline(timed, results)) - 2000) <= 2

    def test_exact_fit_audio_preserved(self):
        timed, results = _timed_and_results([(0, 1000, 1000)])
        samples, _ = _decode(assemble_timeline(timed, results))
        assert any(s != 0 for s in samples)


# ---------------------------------------------------------------------------
# Degenerate windows and resampling
# ---------------------------------------------------------------------------

class TestDegenerateAndResample:
    def test_zero_window_appends_natural_duration(self):
        # Plain-text fallback: window 0 → audio at natural length.
        timed, results = _timed_and_results([(0, 0, 1500)])
        assert abs(_duration_ms(assemble_timeline(timed, results)) - 1500) <= 2

    def test_mismatched_source_rate_is_resampled(self):
        seg = _segment(1, 0, 1000)
        timed = TimelineAligner().align([seg], [1000])
        results = [TTSResult(segment=seg, audio_bytes=_wav(1000, rate=44100), duration_ms=1000)]
        data = assemble_timeline(timed, results)
        samples, rate = _decode(data)
        assert rate == 22050
        assert abs(_duration_ms(data) - 1000) <= 2


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------

class TestAssemblerErrors:
    def test_length_mismatch_raises_value_error(self):
        timed, results = _timed_and_results([(0, 1000, 500)])
        with pytest.raises(ValueError, match="same length"):
            assemble_timeline(timed, [])

    def test_empty_audio_raises_value_error(self):
        seg = _segment(1, 0, 1000)
        timed = TimelineAligner().align([seg], [1000])
        results = [TTSResult(segment=seg, audio_bytes=b"", duration_ms=1000)]
        with pytest.raises(ValueError, match="no audio"):
            assemble_timeline(timed, results)

    def test_invalid_wav_bytes_raise_value_error(self):
        seg = _segment(1, 0, 1000)
        timed = TimelineAligner().align([seg], [1000])
        results = [TTSResult(segment=seg, audio_bytes=b"not a wav", duration_ms=1000)]
        with pytest.raises(ValueError, match="not valid WAV"):
            assemble_timeline(timed, results)


# ---------------------------------------------------------------------------
# Out-of-order SRT entries — rendered in timeline order, not file order
# ---------------------------------------------------------------------------

class TestOutOfOrderEntries:
    def test_unsorted_entries_render_at_their_own_start_times(self):
        """A file listing the 5s entry before the 0s entry must still place
        each segment at its own timestamp, not append the early one late."""
        timed, results = _timed_and_results([(5000, 6000, 800), (0, 1000, 800)])
        samples, rate = _decode(assemble_timeline(timed, results))
        early = samples[int(rate * 0.1): int(rate * 0.7)]
        gap = samples[int(rate * 2.0): int(rate * 4.0)]
        late = samples[int(rate * 5.1): int(rate * 5.7)]
        assert any(s != 0 for s in early), "0–1s segment must play at 0s"
        assert all(s == 0 for s in gap), "the 1–5s gap must be silence"
        assert any(s != 0 for s in late), "5–6s segment must play at 5s"

    def test_unsorted_entries_total_span_matches_last_end(self):
        timed, results = _timed_and_results([(5000, 6000, 800), (0, 1000, 800)])
        assert abs(_duration_ms(assemble_timeline(timed, results)) - 6000) <= 2


# ---------------------------------------------------------------------------
# Overlapping subtitle windows — anchored at SRT start and mixed, never
# shifted later (the timeline-true guarantee must hold under overlap)
# ---------------------------------------------------------------------------

class TestOverlappingEntries:
    def test_overlap_does_not_stretch_timeline(self):
        """0–4s and 3–6s windows must yield exactly 6s of audio, not 7s
        (the second segment must not be appended at the 4s write head)."""
        timed, results = _timed_and_results([(0, 4000, 4000), (3000, 6000, 3000)])
        assert abs(_duration_ms(assemble_timeline(timed, results)) - 6000) <= 2

    def test_overlapping_segment_anchored_at_its_own_start(self):
        """The second speaker must be audible from 3s, not from 4s."""
        timed, results = _timed_and_results([(0, 4000, 4000), (3000, 6000, 3000)])
        samples, rate = _decode(assemble_timeline(timed, results))
        overlap = samples[int(rate * 3.1): int(rate * 3.9)]
        tail = samples[int(rate * 4.1): int(rate * 5.9)]
        assert any(s != 0 for s in overlap), "second segment must start at 3s"
        assert any(s != 0 for s in tail), "second segment must continue past 4s"

    def test_overlap_region_mixes_both_signals(self):
        """Where both windows carry audio the samples are summed (1000+1000),
        outside the overlap each plays alone (1000)."""
        timed, results = _timed_and_results([(0, 4000, 4000), (3000, 6000, 3000)])
        samples, rate = _decode(assemble_timeline(timed, results))
        solo_a = samples[int(rate * 1.0): int(rate * 2.0)]
        mixed = samples[int(rate * 3.2): int(rate * 3.8)]
        solo_b = samples[int(rate * 4.5): int(rate * 5.5)]
        assert all(s == 1000 for s in solo_a), "first segment alone before 3s"
        assert all(s == 2000 for s in mixed), "3–4s overlap must sum both signals"
        assert all(s == 1000 for s in solo_b), "second segment alone after 4s"

    def test_mixed_overlap_clamps_to_pcm_range(self):
        """Summing two near-full-scale signals must clamp, not wrap around."""
        segments = [_segment(1, 0, 1000), _segment(2, 0, 1000)]
        timed = TimelineAligner().align(segments, [1000, 1000])
        results = [
            TTSResult(segment=seg, audio_bytes=_wav(1000, value=30000), duration_ms=1000)
            for seg in segments
        ]
        samples, _ = _decode(assemble_timeline(timed, results))
        assert max(samples) == 32767, "overflow must clamp at PCM max"
        assert all(s >= 0 for s in samples), "clamped sum must never wrap negative"

    def test_fully_contained_overlap_keeps_outer_window_span(self):
        """A window nested inside another (1–2s inside 0–4s) must not extend
        the output beyond the outer window's end."""
        timed, results = _timed_and_results([(0, 4000, 4000), (1000, 2000, 1000)])
        assert abs(_duration_ms(assemble_timeline(timed, results)) - 4000) <= 2

    def test_identical_windows_mix_in_place(self):
        """Two segments sharing one window must occupy that window only."""
        timed, results = _timed_and_results([(0, 2000, 2000), (0, 2000, 2000)])
        out = assemble_timeline(timed, results)
        assert abs(_duration_ms(out) - 2000) <= 2
        samples, rate = _decode(out)
        body = samples[int(rate * 0.2): int(rate * 1.8)]
        assert all(s == 2000 for s in body), "shared window must carry the mixed sum"
