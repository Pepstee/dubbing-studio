from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tts_studio.aligner import AlignedEntry, align_timeline
from tts_studio.models import TTSBackend
from tts_studio.pipeline import DubbingPipeline
from tts_studio.srt_parser import parse_srt


@dataclass
class BatchResult:
    srt_path: str
    entries: list[AlignedEntry]
    warnings: list[str] = field(default_factory=list)


def batch_process(
    srt_paths: list[str | Path],
    backend: TTSBackend,
    out_dir: str | Path,
    language: str = "en",
) -> list[BatchResult]:
    pipeline = DubbingPipeline(backend=backend, default_language=language)
    batch_results: list[BatchResult] = []

    for path in srt_paths:
        path = Path(path)
        warnings: list[str] = []
        segments = parse_srt(path, default_language=language)
        if not segments:
            warnings.append(f"No segments parsed from {path.name}")

        results = pipeline.run(segments)
        entries = align_timeline(segments, results)

        over_budget = [e for e in entries if e.over_budget]
        for e in over_budget:
            warnings.append(
                f"Segment at {e.start_ms}ms exceeds allotted duration "
                f"(stretch_ratio={e.stretch_ratio:.2f})"
            )

        batch_results.append(
            BatchResult(srt_path=str(path), entries=entries, warnings=warnings)
        )

    return batch_results
