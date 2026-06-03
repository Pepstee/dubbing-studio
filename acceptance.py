#!/usr/bin/env python3
"""Acceptance demo: prints a segment plan table using the mock TTS backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tts_studio.mock_backend import MockTTSBackend
from tts_studio.pipeline import DubbingPipeline
from tts_studio.srt_parser import parse_srt


def main() -> None:
    parser = argparse.ArgumentParser(description="Acceptance demo for TTS Studio dubbing pipeline.")
    parser.add_argument("srt", help="Path to the .srt file")
    parser.add_argument("--mock", action="store_true", help="Use mock TTS backend (no network calls)")
    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.exists():
        print(f"Error: SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)

    segments = parse_srt(srt_path)
    pipeline = DubbingPipeline(MockTTSBackend())
    results = pipeline.run(segments)

    col = (5, 20, 42, 13)
    header = (
        f"{'#':<{col[0]}}  "
        f"{'Start–End (ms)':<{col[1]}}  "
        f"{'Text excerpt':<{col[2]}}  "
        f"{'stretch_ratio':<{col[3]}}"
    )
    print(f"Segment plan — {len(segments)} segment(s)\n")
    print(header)
    print("-" * len(header))

    for seg, result in zip(segments, results):
        window_ms = seg.end_ms - seg.start_ms
        ratio = result.duration_ms / window_ms if window_ms > 0 else float("nan")
        excerpt = (seg.text[:39] + "...") if len(seg.text) > 42 else seg.text
        timing = f"{seg.start_ms}–{seg.end_ms}"
        print(
            f"{seg.index:<{col[0]}}  "
            f"{timing:<{col[1]}}  "
            f"{excerpt:<{col[2]}}  "
            f"{ratio:<{col[3]}.3f}"
        )


if __name__ == "__main__":
    main()
