#!/usr/bin/env python3
"""Acceptance demo: runs the dubbing pipeline on the bundled sample.srt using the mock backend."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep the project importable when run directly without installation.
sys.path.insert(0, str(Path(__file__).parent))

from dubbing.backends.mock import MockTTSBackend
from dubbing.pipeline import DubbingPipeline


def _fmt_ms(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms_ = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"


def main() -> None:
    sample = Path(__file__).parent / "sample.srt"
    pipeline = DubbingPipeline(backend=MockTTSBackend())
    segments = pipeline.run(sample)

    print(f"Timed segment plan — {len(segments)} segment(s)\n")
    header = f"{'#':<4}  {'Start':<15} {'End':<15} {'Tags':<34} Text"
    print(header)
    print("-" * len(header))
    for ts in segments:
        seg = ts.segment
        tags = ", ".join(f"{t.name}:{t.value}" for t in seg.tags) or "—"
        print(
            f"{seg.entry.index:<4}  "
            f"{_fmt_ms(ts.start_ms):<15} "
            f"{_fmt_ms(ts.end_ms):<15} "
            f"{tags:<34} "
            f"{seg.entry.text}"
        )


if __name__ == "__main__":
    main()
