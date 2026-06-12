"""Targeted tests to kill surviving mutants.

Four mutation categories covered:
  - boundary values   : off-by-one at condition thresholds
  - negated conditions: sign / direction flips in if-guards
  - off-by-one        : index arithmetic, loop endpoints, frame math
  - return-value variants: wrong field, swapped pair, stale literal
"""
from __future__ import annotations

import io
import wave
from array import array

import pytest

from dubbing.aligner import TimedSegment, TimelineAligner, segment_plan
from dubbing.assembler import (
    DEFAULT_SAMPLE_RATE,
    _decode_wav,
    _encode_wav,
    _fit,
    _ms_to_frames,
    _resample,
    _silence,
    assemble_timeline,
)
from dubbing.models import TTSResult
from dubbing.prosody import parse_prosody
from dubbing.srt_parser import parse_srt_string

from tests.support.assertions import (
    assert_audio_region,
    assert_silence_region,
    assert_wav_duration,
)
from tests.support.builders import make_segment, make_tts_result, make_wav


# ===========================================================================
# SECTION A — BOUNDARY VALUES
# Condition thresholds tested at their exact boundary
# ===========================================================================


class TestAlignerWindowBoundaries:
    """window <= 0 guard: minimum positive window must not enter the degenerate path."""

    def test_window_one_is_not_degenerate_exact_fit(self):
        """window == 1, tts == 1: the 1ms window is valid and fits exactly → ratio 1.0."""
        seg = make_segment(start_ms=0, end_ms=1)
        result = TimelineAligner().align([seg], [1])
        assert result[0].stretch_ratio == pytest.approx(1.0)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1

    def test_window_one_overlong_tts_compresses(self):
        """window == 1, tts == 2: 2× over-budget must give ratio 0.5, not 1.0 from degenerate path."""
        seg = make_segment(start_ms=0, end_ms=1)
        result = TimelineAligner().align([seg], [2])
        assert result[0].stretch_ratio == pytest.approx(0.5)
        assert result[0].end_ms == 1  # window preserved

    def test_negative_window_stretch_ratio_is_one(self):
        """Negative window (end < start) is degenerate → stretch_ratio must be 1.0, not negative."""
        seg = make_segment(start_ms=3000, end_ms=1000)  # window = -2000
        result = TimelineAligner().align([seg], [500])
        assert result[0].stretch_ratio == pytest.approx(1.0)

    def test_tts_ms_one_below_window_gives_natural_speed(self):
        """tts_ms == window - 1 is just under budget → ratio must be 1.0, NOT window/(window-1)."""
        window_ms = 1000
        seg = make_segment(start_ms=0, end_ms=window_ms)
        result = TimelineAligner().align([seg], [window_ms - 1])
        assert result[0].stretch_ratio == pytest.approx(1.0)

    def test_tts_ms_one_above_window_triggers_compression(self):
        """tts_ms == window + 1 is just over budget → ratio must be < 1.0."""
        window_ms = 1000
        seg = make_segment(start_ms=0, end_ms=window_ms)
        result = TimelineAligner().align([seg], [window_ms + 1])
        expected = window_ms / (window_ms + 1)
        assert result[0].stretch_ratio == pytest.approx(expected)
        assert result[0].stretch_ratio < 1.0


class TestSrtParserLineBoundaries:
    """len(lines) < 3 guard tested at 2 (skip) and 3 (parse)."""

    def test_two_line_block_is_skipped(self):
        """Exactly 2 lines (index + timecode, no text) must produce zero entries."""
        srt = "1\n00:00:01,000 --> 00:00:03,000"
        assert parse_srt_string(srt) == []

    def test_three_line_block_is_parsed(self):
        """Exactly 3 lines is the minimum valid block — must NOT be skipped."""
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello"
        entries = parse_srt_string(srt)
        assert len(entries) == 1
        assert entries[0].text == "Hello"

    def test_four_line_block_includes_both_text_lines(self):
        """Lines 2 and 3 both belong to subtitle text — lines[2:] must not be truncated."""
        srt = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two"
        entries = parse_srt_string(srt)
        assert entries[0].text == "Line one\nLine two"

    def test_two_line_block_does_not_corrupt_later_valid_block(self):
        """A bad block (2 lines) followed by a good one must not prevent the good one parsing."""
        srt = (
            "1\n00:00:00,000 --> 00:00:01,000\n\n"  # missing text → 2 lines
            "2\n00:00:02,000 --> 00:00:03,000\nGood"
        )
        entries = parse_srt_string(srt)
        assert len(entries) == 1
        assert entries[0].text == "Good"


class TestMsToFramesBoundaries:
    """_ms_to_frames: formula correctness and rounding at key values."""

    def test_zero_ms_gives_zero_frames(self):
        assert _ms_to_frames(0, 22050) == 0

    def test_one_second_gives_exact_sample_rate(self):
        assert _ms_to_frames(1000, 22050) == 22050

    def test_half_second_at_44100hz(self):
        assert _ms_to_frames(500, 44100) == 22050

    def test_divisor_is_1000_not_100(self):
        """100ms at 10 000 Hz = exactly 1 000 frames — fails if divisor is 100."""
        assert _ms_to_frames(100, 10_000) == 1_000

    def test_higher_rate_gives_more_frames(self):
        """Doubling sample rate must double the frame count for the same duration."""
        assert _ms_to_frames(1000, 44100) == 2 * _ms_to_frames(1000, 22050)

    def test_rounding_matches_python_round(self):
        """int(round(x)) must match Python's built-in round() semantics."""
        # 3ms at 22050 Hz = 66.15 frames → int(round(66.15)) = 66
        assert _ms_to_frames(3, 22050) == int(round(3 * 22050 / 1000))


# ===========================================================================
# SECTION B — NEGATED CONDITIONS
# Guards whose inversion would reverse correct/incorrect behaviour
# ===========================================================================


class TestFitNegatedConditions:
    """_fit: target_frames <= 0 and n == 0 guards."""

    def test_target_zero_returns_empty(self):
        """target_frames == 0 must yield an empty array, not raise or return 1 frame."""
        result = _fit(array("h", [1, 2, 3]), 0)
        assert len(result) == 0

    def test_target_negative_returns_empty(self):
        result = _fit(array("h", [1, 2, 3]), -5)
        assert len(result) == 0

    def test_empty_source_with_positive_target_returns_empty(self):
        """n == 0 must always yield empty, regardless of target."""
        result = _fit(array("h"), 10)
        assert len(result) == 0

    def test_identity_when_lengths_equal(self):
        """n == target_frames: the original array is returned unchanged."""
        s = array("h", [10, 20, 30])
        result = _fit(s, 3)
        assert list(result) == [10, 20, 30]


class TestSilenceNegatedConditions:
    """_silence: max(0, frames) guard at the zero boundary."""

    def test_zero_frames_produces_empty_array(self):
        """_silence(0) must be empty — not one frame (max(1, …) mutation)."""
        assert len(_silence(0)) == 0

    def test_negative_frames_produces_empty_array(self):
        assert len(_silence(-1)) == 0

    def test_large_negative_produces_empty(self):
        assert len(_silence(-1000)) == 0

    def test_positive_frames_produces_exact_count(self):
        n = 17
        result = _silence(n)
        assert len(result) == n

    def test_silence_samples_are_zero(self):
        result = _silence(8)
        assert all(v == 0 for v in result)


class TestAssemblerWindowCondition:
    """window_ms > 0 condition: degenerate (plain-text fallback) vs normal path."""

    def test_zero_window_appends_natural_duration_not_nothing(self):
        """Degenerate window (end == start): audio plays at its natural duration — not silenced."""
        seg = make_segment(start_ms=0, end_ms=0)
        ts = TimedSegment(start_ms=0, end_ms=0, segment=seg)
        res = TTSResult(segment=seg, audio_bytes=make_wav(400), duration_ms=400)
        output = assemble_timeline([ts], [res])
        assert_wav_duration(output, 400, tolerance_ms=5)

    def test_one_ms_window_takes_normal_path(self):
        """window_ms == 1 is the minimum positive window — must pad to full 1ms, not use natural length."""
        seg = make_segment(start_ms=0, end_ms=1)
        ts = TimedSegment(start_ms=0, end_ms=1, segment=seg, stretch_ratio=1.0)
        res = TTSResult(segment=seg, audio_bytes=make_wav(50), duration_ms=50)
        output = assemble_timeline([ts], [res])
        # Output must be ~1ms, not ~50ms (natural) — window path dominates
        assert_wav_duration(output, 1, tolerance_ms=2)

    def test_nonzero_window_pads_short_audio_to_window_end(self):
        """Short audio in a 1 000ms window: assembler pads the tail with silence."""
        seg = make_segment(start_ms=0, end_ms=1000)
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg, stretch_ratio=1.0)
        res = TTSResult(segment=seg, audio_bytes=make_wav(300, sample_value=1000), duration_ms=300)
        output = assemble_timeline([ts], [res])
        assert_wav_duration(output, 1000, tolerance_ms=5)
        assert_silence_region(output, 0.4, 0.99)


# ===========================================================================
# SECTION C — OFF-BY-ONE
# Indexing, loop endpoints, frame arithmetic
# ===========================================================================


class TestFitEndpointPreservation:
    """_fit step formula: (n-1)/(target-1) must preserve first and last samples."""

    def test_upsampled_first_sample_preserved(self):
        """Upsampling [500, 0] → 5 frames: first frame must equal 500."""
        s = array("h", [500, 0])
        assert _fit(s, 5)[0] == 500

    def test_upsampled_last_sample_preserved(self):
        """Upsampling [0, 1000] → 5 frames: last frame must equal 1000."""
        s = array("h", [0, 1000])
        result = _fit(s, 5)
        assert result[-1] == 1000

    def test_downsampled_first_sample_preserved(self):
        s = array("h", [300, 200, 100, 50, 25])
        assert _fit(s, 3)[0] == 300

    def test_downsampled_last_sample_preserved(self):
        s = array("h", [0, 100, 200, 300, 400])
        result = _fit(s, 3)
        assert result[-1] == 400

    def test_output_length_is_exactly_target(self):
        """Output length must be exactly target_frames — never ± 1."""
        base = array("h", list(range(50)))
        for target in [1, 2, 3, 7, 10, 49, 50, 51, 100]:
            result = _fit(base, target)
            assert len(result) == target, f"target={target}: got {len(result)}"

    def test_single_source_sample_broadcast(self):
        """1 input sample to N targets: step is 0, all outputs equal the source."""
        s = array("h", [99])
        result = _fit(s, 6)
        assert len(result) == 6
        assert all(v == 99 for v in result)

    def test_single_target_preserves_first_sample(self):
        s = array("h", [42, 100, 200])
        result = _fit(s, 1)
        assert result[0] == 42


class TestResampleDirection:
    """_resample: ratio uses dst/src not src/dst."""

    def test_same_rate_returns_identical_object(self):
        """Same src/dst rate: no resampling — the original array is returned unchanged."""
        s = array("h", [1, 2, 3])
        result = _resample(s, 22050, 22050)
        assert result is s

    def test_double_rate_doubles_frame_count(self):
        """22050 → 44100 should double the sample count."""
        s = array("h", list(range(100)))
        result = _resample(s, 22050, 44100)
        assert len(result) == round(100 * 44100 / 22050)

    def test_half_rate_halves_frame_count(self):
        """44100 → 22050 should halve the sample count."""
        s = array("h", list(range(100)))
        result = _resample(s, 44100, 22050)
        assert len(result) == round(100 * 22050 / 44100)

    def test_rate_conversion_not_inverted(self):
        """Upsampling (22050→44100) must produce MORE frames than downsampling (44100→22050)."""
        s = array("h", list(range(100)))
        up = _resample(s, 22050, 44100)
        down = _resample(s, 44100, 22050)
        assert len(up) > len(down)


class TestDecodeWavReturnValues:
    """_decode_wav: (samples, rate) pair — rate must reflect the actual WAV header."""

    def test_returns_actual_framerate_22050(self):
        audio = make_wav(200, rate=22050)
        _, rate = _decode_wav(audio)
        assert rate == 22050

    def test_returns_actual_framerate_44100(self):
        """Rate must come from the WAV header, not be hardcoded."""
        audio = make_wav(200, rate=44100)
        _, rate = _decode_wav(audio)
        assert rate == 44100

    def test_rate_44100_different_from_22050(self):
        """Ensure the two rates are distinguishable (mutation: always return 22050)."""
        _, r1 = _decode_wav(make_wav(100, rate=22050))
        _, r2 = _decode_wav(make_wav(100, rate=44100))
        assert r1 != r2

    def test_sample_count_matches_wav_frames(self):
        """Number of decoded samples must equal the frame count encoded in the WAV."""
        n_frames = 441
        s = array("h", [i % 32767 for i in range(n_frames)])
        data = _encode_wav(s, 22050)
        samples, _ = _decode_wav(data)
        assert len(samples) == n_frames

    def test_raises_on_garbage_bytes(self):
        with pytest.raises(ValueError, match="valid WAV"):
            _decode_wav(b"this is not a wav file")

    def test_raises_on_8bit_wav(self):
        """8-bit WAV is unsupported; error message must mention bit width."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)  # 8-bit
            wf.setframerate(22050)
            wf.writeframes(bytes(100))
        with pytest.raises(ValueError, match="8-bit"):
            _decode_wav(buf.getvalue())


class TestSrtParserFieldIndexing:
    """parse_srt_string line index assignment: index from [0], timecode from [1], text from [2:]."""

    def test_index_extracted_from_first_line(self):
        """The integer on line 0 becomes entry.index — not line 1 or line 2."""
        srt = "42\n00:00:01,000 --> 00:00:03,000\nText"
        entries = parse_srt_string(srt)
        assert entries[0].index == 42

    def test_start_ms_from_timecode_left_side(self):
        """start_ms must come from groups 1–4 of the timecode regex, not 5–8."""
        srt = "1\n00:00:05,000 --> 00:00:10,000\nText"
        entries = parse_srt_string(srt)
        assert entries[0].start_ms == 5_000

    def test_end_ms_from_timecode_right_side(self):
        """end_ms must come from groups 5–8 of the timecode regex, not 1–4."""
        srt = "1\n00:00:05,000 --> 00:00:10,500\nText"
        entries = parse_srt_string(srt)
        assert entries[0].end_ms == 10_500

    def test_start_and_end_ms_not_swapped(self):
        """start_ms < end_ms — if groups are swapped both would equal end_ms."""
        srt = "1\n00:00:01,000 --> 00:00:05,000\nText"
        entries = parse_srt_string(srt)
        assert entries[0].start_ms == 1_000
        assert entries[0].end_ms == 5_000
        assert entries[0].start_ms < entries[0].end_ms

    def test_text_starts_at_line_two_not_line_one(self):
        """Text must be from lines[2:] — if it were lines[1:], the timecode would appear in text."""
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello world"
        entries = parse_srt_string(srt)
        assert "-->" not in entries[0].text
        assert entries[0].text == "Hello world"

    def test_multiline_text_joined_from_line_two_onwards(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nFirst\nSecond"
        entries = parse_srt_string(srt)
        assert entries[0].text == "First\nSecond"


# ===========================================================================
# SECTION D — RETURN-VALUE VARIANTS
# Wrong field, swapped pair, stale literal
# ===========================================================================


class TestSegmentPlanReturnValues:
    """segment_plan() must map each TimedSegment to a dict with exact field values."""

    def test_all_four_keys_present(self):
        seg = make_segment(start_ms=0, end_ms=1000, text="Hello")
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg, stretch_ratio=1.0)
        plan = segment_plan([ts])
        assert set(plan[0].keys()) == {"start_ms", "end_ms", "stretch_ratio", "text"}

    def test_start_ms_value_correct(self):
        seg = make_segment(start_ms=2000, end_ms=4000)
        ts = TimedSegment(start_ms=2000, end_ms=4000, segment=seg)
        assert segment_plan([ts])[0]["start_ms"] == 2000

    def test_end_ms_value_correct(self):
        seg = make_segment(start_ms=2000, end_ms=4000)
        ts = TimedSegment(start_ms=2000, end_ms=4000, segment=seg)
        assert segment_plan([ts])[0]["end_ms"] == 4000

    def test_start_and_end_not_swapped(self):
        """start_ms and end_ms must not be swapped in the output dict."""
        seg = make_segment(start_ms=100, end_ms=900)
        ts = TimedSegment(start_ms=100, end_ms=900, segment=seg)
        plan = segment_plan([ts])[0]
        assert plan["start_ms"] == 100
        assert plan["end_ms"] == 900

    def test_stretch_ratio_reflects_actual_ratio_not_always_one(self):
        """stretch_ratio in the plan must come from ts.stretch_ratio, not be hardcoded 1.0."""
        seg = make_segment(start_ms=0, end_ms=1000)
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg, stretch_ratio=0.333)
        plan = segment_plan([ts])[0]
        assert plan["stretch_ratio"] == pytest.approx(0.333)

    def test_stretch_ratio_one_is_also_preserved(self):
        seg = make_segment(start_ms=0, end_ms=1000)
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg, stretch_ratio=1.0)
        assert segment_plan([ts])[0]["stretch_ratio"] == pytest.approx(1.0)

    def test_text_from_segment_entry_text(self):
        """text must come from segment.entry.text — not entry.index or a literal ""."""
        seg = make_segment(start_ms=0, end_ms=1000, text="Unique sentence here")
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg)
        assert segment_plan([ts])[0]["text"] == "Unique sentence here"

    def test_order_preserved_across_multiple_segments(self):
        segs = [
            make_segment(index=i, start_ms=i * 1000, end_ms=(i + 1) * 1000, text=f"word{i}")
            for i in range(3)
        ]
        timed = [TimedSegment(start_ms=s.entry.start_ms, end_ms=s.entry.end_ms, segment=s) for s in segs]
        plan = segment_plan(timed)
        assert [p["text"] for p in plan] == ["word0", "word1", "word2"]

    def test_empty_input_returns_empty_list(self):
        assert segment_plan([]) == []

    def test_returns_list_of_dicts(self):
        seg = make_segment()
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg)
        result = segment_plan([ts])
        assert isinstance(result, list)
        assert isinstance(result[0], dict)


class TestParseProsodyReturnValues:
    """parse_prosody() extra checks beyond test_prosody.py: field order and clean-vs-original."""

    def test_clean_text_is_not_original_when_tag_present(self):
        """First element of return must be the cleaned text, NOT the input string."""
        original = "<pitch:high>content"
        clean, _ = parse_prosody(original)
        assert clean != original
        assert clean == "content"

    def test_name_is_left_of_colon_not_right(self):
        """ProsodyTag.name holds the part before ':' — not swapped with .value."""
        _, tags = parse_prosody("<rate:slow>x")
        assert tags[0].name == "rate"
        assert tags[0].value != "rate"

    def test_value_is_right_of_colon_not_left(self):
        _, tags = parse_prosody("<emotion:excited>x")
        assert tags[0].value == "excited"
        assert tags[0].name != "excited"

    def test_multiple_tags_all_stripped_from_clean(self):
        """All occurrences stripped — not just the first (sub vs sub-once)."""
        clean, _ = parse_prosody("<rate:slow><pitch:low>text")
        assert "<rate:slow>" not in clean
        assert "<pitch:low>" not in clean
        assert clean == "text"

    def test_tags_list_not_empty_when_tags_present(self):
        """Second element must be a non-empty list when the text contains tags."""
        _, tags = parse_prosody("<emotion:happy>hello")
        assert len(tags) >= 1

    def test_no_tags_returns_empty_list_not_none(self):
        _, tags = parse_prosody("plain")
        assert tags == []
        assert tags is not None


class TestAlignerReturnValues:
    """TimelineAligner.align() field correctness in TimedSegment."""

    def test_stretch_ratio_for_overlong_is_window_over_tts_not_inverted(self):
        """Overlong TTS: ratio = window/tts, not tts/window (inverted fraction mutant)."""
        window_ms, tts_ms = 1000, 4000
        seg = make_segment(start_ms=0, end_ms=window_ms)
        result = TimelineAligner().align([seg], [tts_ms])
        assert result[0].stretch_ratio == pytest.approx(window_ms / tts_ms)
        assert result[0].stretch_ratio < 1.0  # compression, not expansion

    def test_end_ms_equals_srt_end_for_overlong_tts(self):
        """end_ms must equal srt_end, not srt_start + tts_ms or anything else."""
        seg = make_segment(start_ms=500, end_ms=1500)  # window=1000
        result = TimelineAligner().align([seg], [3000])
        assert result[0].end_ms == 1500

    def test_start_ms_unchanged_from_srt_entry(self):
        seg = make_segment(start_ms=7777, end_ms=8777)
        result = TimelineAligner().align([seg], [500])
        assert result[0].start_ms == 7777

    def test_segment_reference_is_the_same_object(self):
        """The segment field must hold the original Segment object — not a copy."""
        seg = make_segment()
        result = TimelineAligner().align([seg], [500])
        assert result[0].segment is seg

    def test_degenerate_window_has_ratio_one_not_zero(self):
        """Zero-width window takes the degenerate path: stretch_ratio = 1.0 (default)."""
        seg = make_segment(start_ms=500, end_ms=500)
        result = TimelineAligner().align([seg], [1000])
        assert result[0].stretch_ratio == pytest.approx(1.0)

    def test_zero_tts_duration_has_ratio_one_not_zero(self):
        """tts_ms == 0 takes the degenerate path: stretch_ratio = 1.0 (default)."""
        seg = make_segment(start_ms=0, end_ms=2000)
        result = TimelineAligner().align([seg], [0])
        assert result[0].stretch_ratio == pytest.approx(1.0)

    def test_error_message_includes_both_lengths(self):
        """ValueError for mismatched lengths must mention both counts for debuggability."""
        segs = [make_segment(index=i) for i in range(3)]
        with pytest.raises(ValueError, match=r"3"):
            TimelineAligner().align(segs, [100])


class TestAssemblerOutputCorrectness:
    """assemble_timeline() output WAV correctness beyond existing test_assembler.py."""

    def test_output_sample_rate_is_default(self):
        seg = make_segment(start_ms=0, end_ms=500)
        ts = TimedSegment(start_ms=0, end_ms=500, segment=seg)
        res = make_tts_result(segment=seg, duration_ms=300)
        output = assemble_timeline([ts], [res])
        with wave.open(io.BytesIO(output)) as wf:
            assert wf.getframerate() == DEFAULT_SAMPLE_RATE

    def test_overlong_compressed_audio_fills_window(self):
        """Overlong audio compressed to window must NOT leave silence in the window interior."""
        seg = make_segment(start_ms=0, end_ms=1000)
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg, stretch_ratio=0.5)
        long_audio = make_wav(2000, sample_value=1000)
        res = TTSResult(segment=seg, audio_bytes=long_audio, duration_ms=2000)
        output = assemble_timeline([ts], [res])
        assert_wav_duration(output, 1000, tolerance_ms=5)
        assert_audio_region(output, 0.0, 0.9)  # audio throughout, not just first half

    def test_short_audio_silence_in_tail(self):
        """Short audio must be followed by silence to fill the window."""
        seg = make_segment(start_ms=0, end_ms=2000)
        ts = TimedSegment(start_ms=0, end_ms=2000, segment=seg, stretch_ratio=1.0)
        res = TTSResult(
            segment=seg, audio_bytes=make_wav(500, sample_value=1000), duration_ms=500
        )
        output = assemble_timeline([ts], [res])
        assert_wav_duration(output, 2000, tolerance_ms=5)
        assert_silence_region(output, 0.6, 1.95)

    def test_leading_gap_is_silence(self):
        """Samples before the first subtitle start time must all be zero."""
        seg = make_segment(start_ms=1000, end_ms=2000)
        ts = TimedSegment(start_ms=1000, end_ms=2000, segment=seg)
        res = make_tts_result(segment=seg, duration_ms=500, sample_value=1000)
        output = assemble_timeline([ts], [res])
        assert_silence_region(output, 0.0, 0.9)
        assert_audio_region(output, 1.0, 1.4)

    def test_raises_value_error_on_length_mismatch(self):
        seg = make_segment()
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg)
        res = make_tts_result(segment=seg)
        with pytest.raises(ValueError, match="same length"):
            assemble_timeline([ts, ts], [res])

    def test_raises_value_error_on_empty_audio_bytes(self):
        seg = make_segment()
        ts = TimedSegment(start_ms=0, end_ms=1000, segment=seg)
        res = TTSResult(segment=seg, audio_bytes=b"", duration_ms=0)
        with pytest.raises(ValueError):
            assemble_timeline([ts], [res])
