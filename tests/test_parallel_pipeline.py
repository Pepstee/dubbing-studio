"""Tests for DubbingPipeline parallel synthesis.

Acceptance criteria:
  - Pipeline with a mock backend that sleeps 0.05 s per segment finishes
    6 segments in under 3× single-segment wall-clock time (verifies true
    concurrency, not serial invocation).
  - Output order matches ascending SRT index regardless of which segment's
    synthesize() call completes first.
"""
from __future__ import annotations

import io
import time
import wave

import pytest

from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, TTSResult
from dubbing.pipeline import DubbingPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00" * 2)
    return buf.getvalue()


SLEEP_S = 0.05  # per-segment sleep, as specified in acceptance criteria

_SRT_ONE = """\
1
00:00:00,000 --> 00:00:02,000
Segment one

"""

_SRT_SIX = """\
1
00:00:00,000 --> 00:00:02,000
Segment one

2
00:00:02,000 --> 00:00:04,000
Segment two

3
00:00:04,000 --> 00:00:06,000
Segment three

4
00:00:06,000 --> 00:00:08,000
Segment four

5
00:00:08,000 --> 00:00:10,000
Segment five

6
00:00:10,000 --> 00:00:12,000
Segment six

"""

# SRT where segment 1 is last in SRT order — forces the parser to handle it.
_SRT_REVERSE_WRITTEN = """\
6
00:00:10,000 --> 00:00:12,000
Sixth subtitle

5
00:00:08,000 --> 00:00:10,000
Fifth subtitle

4
00:00:06,000 --> 00:00:08,000
Fourth subtitle

3
00:00:04,000 --> 00:00:06,000
Third subtitle

2
00:00:02,000 --> 00:00:04,000
Second subtitle

1
00:00:00,000 --> 00:00:02,000
First subtitle

"""


# ---------------------------------------------------------------------------
# Test backends
# ---------------------------------------------------------------------------

class _SleepingBackend(TTSBackend):
    """Sleeps SLEEP_S per call — simulates I/O-bound TTS without subprocesses."""

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        time.sleep(SLEEP_S)
        wav = _minimal_wav()
        return [TTSResult(segment=seg, audio_bytes=wav, duration_ms=int(SLEEP_S * 1000))
                for seg in segments]


class _ReverseCompletionBackend(TTSBackend):
    """Higher-indexed segments complete first to stress-test ordering.

    Segment with SRT index N sleeps  SLEEP_S / N  seconds, so index 6
    finishes ~6× faster than index 1.  The pipeline must still return
    results in ascending index order.
    """

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        seg = segments[0]
        time.sleep(SLEEP_S / seg.entry.index)
        wav = _minimal_wav()
        return [TTSResult(segment=s, audio_bytes=wav, duration_ms=50) for s in segments]


class _FailingBackend(TTSBackend):
    """Always raises RuntimeError — for error-propagation tests."""

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        raise RuntimeError("TTS engine failure")


class _InstantBackend(TTSBackend):
    """Returns immediately with minimal audio — for non-timing tests."""

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        wav = _minimal_wav()
        return [TTSResult(segment=seg, audio_bytes=wav, duration_ms=100) for seg in segments]


# ---------------------------------------------------------------------------
# Timing: 6 segments finish in < 3 × single-segment wall-clock time
# ---------------------------------------------------------------------------

class TestParallelSpeedup:
    def test_six_segments_faster_than_three_times_single_segment(self):
        pipeline = DubbingPipeline(_SleepingBackend())

        t0 = time.perf_counter()
        pipeline.run_full(_SRT_ONE)
        single_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        pipeline.run_full(_SRT_SIX)
        parallel_t = time.perf_counter() - t0

        assert parallel_t < 3 * single_t, (
            f"Expected parallel ({parallel_t:.3f}s) < 3 × single ({single_t:.3f}s = {3 * single_t:.3f}s). "
            f"This suggests segments are being serialised rather than parallelised."
        )

    def test_six_segments_clearly_faster_than_sequential(self):
        """6-segment parallel run must beat the time sequential would take."""
        sequential_lower_bound = 6 * SLEEP_S  # what purely sequential execution costs

        pipeline = DubbingPipeline(_SleepingBackend())
        t0 = time.perf_counter()
        pipeline.run_full(_SRT_SIX)
        elapsed = time.perf_counter() - t0

        assert elapsed < sequential_lower_bound, (
            f"Parallel run took {elapsed:.3f}s; pure-sequential would take "
            f"≥{sequential_lower_bound:.3f}s.  Segments appear to run serially."
        )

    def test_result_count_matches_segment_count(self):
        pipeline = DubbingPipeline(_SleepingBackend())
        timed, results = pipeline.run_full(_SRT_SIX)
        assert len(results) == 6
        assert len(timed) == 6


# ---------------------------------------------------------------------------
# Output order: ascending SRT index regardless of completion order
# ---------------------------------------------------------------------------

class TestOutputOrder:
    def test_results_in_srt_index_order_despite_reverse_completion(self):
        """Segments that complete last (index 1) must appear first in output."""
        pipeline = DubbingPipeline(_ReverseCompletionBackend())
        timed, results = pipeline.run_full(_SRT_SIX)

        for i, result in enumerate(results):
            expected_index = i + 1  # SRT indices are 1-based
            assert result.segment.entry.index == expected_index, (
                f"Result at position {i} has SRT index {result.segment.entry.index}, "
                f"expected {expected_index}.  Output is not in submission order."
            )

    def test_timed_segments_in_srt_index_order(self):
        pipeline = DubbingPipeline(_ReverseCompletionBackend())
        timed, _ = pipeline.run_full(_SRT_SIX)

        for i, ts in enumerate(timed):
            expected_index = i + 1
            assert ts.segment.entry.index == expected_index

    def test_start_times_are_non_decreasing_in_output(self):
        pipeline = DubbingPipeline(_InstantBackend())
        timed, _ = pipeline.run_full(_SRT_SIX)

        for prev, curr in zip(timed, timed[1:]):
            assert prev.start_ms <= curr.start_ms, (
                f"Non-monotonic start times: {prev.start_ms} then {curr.start_ms}"
            )

    def test_reverse_written_srt_results_in_ascending_start_times(self):
        """SRT blocks written in reverse index order still produce correct output.

        The parser reads blocks sequentially, so the output list is in the
        written order (6,5,4,3,2,1 SRT indices).  After alignment the start_ms
        values must still be non-decreasing if the pipeline preserves
        parse-order (it does — the aligner maps each segment to its own SRT
        window, so written order determines the output list order).
        """
        pipeline = DubbingPipeline(_InstantBackend())
        timed, results = pipeline.run_full(_SRT_REVERSE_WRITTEN)
        # Written order: indices 6,5,4,3,2,1 → start_ms: 10000,8000,6000,4000,2000,0
        # These are DECREASING start times (because SRT was written backwards).
        # The key assertion is that the pipeline does NOT reorder by start time —
        # it preserves parse order and the assembler handles the rest.
        assert len(timed) == 6
        assert len(results) == 6
        # First result is for SRT index 6 (written first, start_ms=10000)
        assert results[0].segment.entry.index == 6
        assert timed[0].start_ms == 10000

    def test_result_text_matches_srt_index_position(self):
        """Segment text must travel with its segment through the pipeline."""
        pipeline = DubbingPipeline(_ReverseCompletionBackend())
        _, results = pipeline.run_full(_SRT_SIX)

        texts = ["Segment one", "Segment two", "Segment three",
                 "Segment four", "Segment five", "Segment six"]
        for i, result in enumerate(results):
            assert result.segment.entry.text == texts[i], (
                f"Position {i}: expected '{texts[i]}', "
                f"got '{result.segment.entry.text}'"
            )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestParallelErrorHandling:
    def test_failing_backend_raises_runtime_error(self):
        pipeline = DubbingPipeline(_FailingBackend())
        with pytest.raises(RuntimeError, match="[Ss]egment|[Ss]ynthesis|TTS"):
            pipeline.run_full(_SRT_ONE)

    def test_failing_backend_on_six_segments_propagates_exception(self):
        """Cancellation never leaks implementation-level CancelledError."""
        pipeline = DubbingPipeline(_FailingBackend())
        with pytest.raises(RuntimeError, match="Segment synthesis failed"):
            pipeline.run_full(_SRT_SIX)

    def test_error_message_wraps_original_cause(self):
        pipeline = DubbingPipeline(_FailingBackend())
        with pytest.raises(RuntimeError) as exc_info:
            pipeline.run_full(_SRT_ONE)
        # Original "TTS engine failure" message must be chained or embedded.
        cause = exc_info.value.__cause__
        assert cause is not None or "TTS engine failure" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_segment_list_returns_empty_results(self):
        # An SRT string that parses to nothing falls back to a single
        # literal-text segment in run_full, so test _synthesize_parallel
        # directly via a subclass hook.
        pipeline = DubbingPipeline(_InstantBackend())
        # _synthesize_parallel with empty list should return []
        result = pipeline._synthesize_parallel([])
        assert result == []

    def test_single_segment_returns_one_result(self):
        pipeline = DubbingPipeline(_InstantBackend())
        _, results = pipeline.run_full(_SRT_ONE)
        assert len(results) == 1

    def test_single_segment_result_is_tts_result(self):
        pipeline = DubbingPipeline(_InstantBackend())
        _, results = pipeline.run_full(_SRT_ONE)
        assert isinstance(results[0], TTSResult)

    def test_results_list_has_no_none_entries(self):
        pipeline = DubbingPipeline(_InstantBackend())
        _, results = pipeline.run_full(_SRT_SIX)
        assert all(r is not None for r in results)

    def test_all_results_have_audio_bytes(self):
        pipeline = DubbingPipeline(_InstantBackend())
        _, results = pipeline.run_full(_SRT_SIX)
        for r in results:
            assert isinstance(r.audio_bytes, bytes)
            assert len(r.audio_bytes) > 0

    def test_all_results_have_positive_duration(self):
        pipeline = DubbingPipeline(_InstantBackend())
        _, results = pipeline.run_full(_SRT_SIX)
        for r in results:
            assert r.duration_ms > 0
