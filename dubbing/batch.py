from __future__ import annotations

import json
from pathlib import Path

from dubbing.aligner import TimedSegment, segment_plan
from dubbing.assembler import assemble_timeline
from dubbing.backends.base import TTSBackend
from dubbing.models import DEFAULT_MAX_SEGMENTS, DEFAULT_MAX_SYNTHESIS_SECONDS, JobConfig
from dubbing.pipeline import DubbingPipeline


def batch_dub(
    inputs: list[str | Path],
    backend: TTSBackend,
    output_dir: str | Path,
    language: str = "",
    job_config: JobConfig | None = None,
) -> dict[Path, list[TimedSegment]]:
    """Dub every input SRT independently, writing one WAV + JSON per input.

    For each `<stem>.srt`, `<stem>.wav` (timeline-true dubbed audio) and
    `<stem>.json` (the segment plan) are written to `output_dir`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if job_config is None:
        job_config = JobConfig(
            max_segments=DEFAULT_MAX_SEGMENTS,
            max_synthesis_seconds=DEFAULT_MAX_SYNTHESIS_SECONDS,
        )
    pipeline = DubbingPipeline(backend, job_config)
    results: dict[Path, list[TimedSegment]] = {}
    for item in inputs:
        path = Path(item)
        timed, tts_results = pipeline.run_full(path, language=language)
        results[path] = timed
        if timed:
            wav_path = output_dir / f"{path.stem}.wav"
            wav_path.write_bytes(assemble_timeline(timed, tts_results))
        json_path = output_dir / f"{path.stem}.json"
        json_path.write_text(json.dumps(segment_plan(timed), indent=2), encoding="utf-8")
    return results
