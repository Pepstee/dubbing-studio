from __future__ import annotations

import concurrent.futures
import os
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
from dubbing.models import Segment, SRTEntry, TTSResult
from dubbing.prosody import parse_prosody
from dubbing.srt_parser import parse_srt, parse_srt_string


class DubbingPipeline:
    def __init__(self, backend: TTSBackend) -> None:
        self._backend = backend
        self._aligner = TimelineAligner()

    def run_full(
        self, input: str | Path, language: str = ""
    ) -> tuple[list[TimedSegment], list[TTSResult]]:
        if isinstance(input, Path):
            entries = parse_srt(input)
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._backend.synthesize, [seg]): i
                for i, seg in enumerate(segments)
            }
            for fut in concurrent.futures.as_completed(future_to_idx):
                if fut.cancelled():
                    continue
                try:
                    exc = fut.exception()
                except concurrent.futures.CancelledError:
                    continue
                if exc is not None:
                    if first_error is None:
                        first_error = exc
                    for f in future_to_idx:
                        f.cancel()
                elif first_error is None:
                    idx = future_to_idx[fut]
                    ordered[idx] = fut.result()[0]

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
