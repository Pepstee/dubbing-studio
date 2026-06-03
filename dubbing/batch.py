from __future__ import annotations

from pathlib import Path

from dubbing.aligner import TimedSegment
from dubbing.backends.base import TTSBackend
from dubbing.pipeline import DubbingPipeline


def batch_dub(
    inputs: list[str | Path],
    backend: TTSBackend,
    output_dir: str | Path,
) -> dict[Path, list[TimedSegment]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = DubbingPipeline(backend)
    results: dict[Path, list[TimedSegment]] = {}
    for item in inputs:
        path = Path(item)
        results[path] = pipeline.run(path)
    return results
