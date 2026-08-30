from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path

from dubbing.aligner import TimedSegment, TimelineAligner
from dubbing.backends.base import TTSBackend
from dubbing.diarization.attribution import attribute_timed_segments
from dubbing.diarization.base import DiarizationBackend
from dubbing.diarization.models import (
    DiarizationResult,
    SegmentAttribution,
    SpeakerConstraints,
)
from dubbing.models import (
    JobConfig,
    Segment,
    SegmentLimitExceeded,
    SRTEntry,
    SynthesisTimeBudgetExceeded,
    TTSResult,
    validate_language,
)
from dubbing.prosody import parse_prosody
from dubbing.srt_parser import parse_srt_string


def _estimated_total_seconds(segments: list[Segment]) -> float:
    return sum(max(0, item.entry.end_ms - item.entry.start_ms) for item in segments) / 1000


class DubbingPipeline:
    def __init__(self, backend: TTSBackend, job_config: JobConfig | None = None) -> None:
        self._backend = backend
        self._aligner = TimelineAligner()
        self._job_config = job_config or JobConfig()

    def run_full(
        self, input: str | Path, language: str = ""
    ) -> tuple[list[TimedSegment], list[TTSResult]]:
        language = validate_language(language)
        if isinstance(input, Path):
            source = input.read_text(encoding="utf-8")
            if "\x00" in source:
                raise ValueError(f"SRT file contains NUL bytes: {input}")
            entries = parse_srt_string(source)
            if not entries:
                raise ValueError(f"SRT file contains no valid subtitle entries: {input}")
        else:
            entries = parse_srt_string(input)
            if not entries:
                entries = [SRTEntry(index=1, start_ms=0, end_ms=0, text=input)]

        segments: list[Segment] = []
        for entry in entries:
            clean_text, tags = parse_prosody(entry.text)
            segments.append(
                Segment(
                    entry=SRTEntry(
                        index=entry.index,
                        start_ms=entry.start_ms,
                        end_ms=entry.end_ms,
                        text=clean_text,
                    ),
                    tags=tags,
                    language=language,
                )
            )

        if (
            self._job_config.max_segments is not None
            and len(segments) > self._job_config.max_segments
        ):
            raise SegmentLimitExceeded(
                f"Segment count {len(segments)} exceeds limit {self._job_config.max_segments}",
                limit=self._job_config.max_segments,
                requested=len(segments),
            )

        if self._job_config.max_synthesis_seconds is not None:
            estimated_seconds = _estimated_total_seconds(segments)
            if estimated_seconds > self._job_config.max_synthesis_seconds:
                raise SynthesisTimeBudgetExceeded(
                    f"Estimated synthesis time {estimated_seconds:.1f}s exceeds limit "
                    f"{self._job_config.max_synthesis_seconds}s",
                    limit=self._job_config.max_synthesis_seconds,
                    requested=estimated_seconds,
                )

        results = self._synthesize_parallel(segments)
        durations = [r.duration_ms for r in results]
        timed = self._aligner.align(segments, durations)
        return timed, results

    def _synthesize_parallel(self, segments: list[Segment]) -> list[TTSResult]:
        if not segments:
            return []

        max_workers = min(len(segments), os.cpu_count() or 4)
        ordered: list[TTSResult | None] = [None] * len(segments)
        first_error: BaseException | None = None
        budget = self._job_config.max_synthesis_seconds
        started_at = time.monotonic()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        timed_out = False
        future_to_idx = {
            executor.submit(self._backend.synthesize, [seg]): i for i, seg in enumerate(segments)
        }
        pending = set(future_to_idx)
        try:
            while pending:
                timeout = None
                if budget is not None:
                    timeout = max(0.0, budget - (time.monotonic() - started_at))
                completed, pending = concurrent.futures.wait(
                    pending,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not completed:
                    timed_out = True
                    elapsed = time.monotonic() - started_at
                    for future in pending:
                        future.cancel()
                    raise SynthesisTimeBudgetExceeded(
                        f"Synthesis exceeded {budget}s wall-clock limit",
                        limit=budget,
                        requested=elapsed,
                    )
                for future in completed:
                    if future.cancelled():
                        continue
                    exc = future.exception()
                    if exc is not None:
                        if first_error is None:
                            first_error = exc
                        for remaining in pending:
                            remaining.cancel()
                    elif first_error is None:
                        index = future_to_idx[future]
                        ordered[index] = future.result()[0]
        finally:
            executor.shutdown(wait=not timed_out, cancel_futures=True)

        if first_error is not None:
            raise RuntimeError(f"Segment synthesis failed: {first_error}") from first_error

        return ordered  # type: ignore[return-value]

    def run(self, input: str | Path, language: str = "") -> list[TimedSegment]:
        timed, _ = self.run_full(input, language=language)
        return timed

    def run_full_with_diarization(
        self,
        input: str | Path,
        source_audio: str | Path,
        diarizer: DiarizationBackend,
        *,
        language: str = "",
        constraints: SpeakerConstraints | None = None,
    ) -> tuple[
        list[TimedSegment],
        list[TTSResult],
        DiarizationResult,
        list[SegmentAttribution],
    ]:
        """Run the existing dubbing core and attribute it from source audio."""
        timed, results = self.run_full(input, language=language)
        diarization = diarizer.diarize(source_audio, constraints=constraints)
        attributions = attribute_timed_segments(timed, diarization.turns)
        return timed, results, diarization, attributions
