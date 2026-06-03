from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

from dubbing.backends.base import TTSBackend
from dubbing.backends.mock import MockTTSBackend
from dubbing.batch import batch_dub
from dubbing.pipeline import DubbingPipeline


def _make_backend(name: str) -> TTSBackend:
    if name == "mock":
        return MockTTSBackend()
    raise SystemExit(f"Unknown backend: {name!r}. Available: mock")


def _cmd_dub(args: argparse.Namespace) -> None:
    backend = _make_backend(args.backend)
    pipeline = DubbingPipeline(backend)
    results = pipeline.run(Path(args.srt))
    output = Path(args.output) if args.output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        out_file = output / (Path(args.srt).stem + ".json")
        import json
        out_file.write_text(
            json.dumps(
                [{"start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.segment.entry.text} for s in results],
                indent=2,
            )
        )
        print(f"Wrote {out_file}")
    else:
        for seg in results:
            print(f"[{seg.start_ms}–{seg.end_ms}] {seg.segment.entry.text}")


def _cmd_batch(args: argparse.Namespace) -> None:
    backend = _make_backend(args.backend)
    paths = [Path(p) for p in _glob.glob(args.glob, recursive=True)]
    if not paths:
        print(f"No files matched: {args.glob}", file=sys.stderr)
        sys.exit(1)
    output_dir = args.output or "."
    results = batch_dub(paths, backend, output_dir)
    for path, segs in results.items():
        print(f"{path}: {len(segs)} segment(s)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dubbing",
        description="Dubbing Studio — synthesise dubbed audio from SRT files.",
    )
    parser.add_argument("--backend", default="mock", help="TTS backend to use (default: mock)")
    parser.add_argument("--lang", default=None, help="Target language code")
    parser.add_argument("--output", default=None, help="Output directory")

    sub = parser.add_subparsers(dest="command", required=True)

    dub_p = sub.add_parser("dub", help="Dub a single SRT file")
    dub_p.add_argument("srt", help="Path to the .srt file")
    dub_p.add_argument("--backend", default="mock", help="TTS backend to use (default: mock)")
    dub_p.add_argument("--lang", default=None, help="Target language code")
    dub_p.add_argument("--output", default=None, help="Output directory")

    batch_p = sub.add_parser("batch", help="Dub multiple SRT files matching a glob")
    batch_p.add_argument("glob", help="Glob pattern matching .srt files")
    batch_p.add_argument("--backend", default="mock", help="TTS backend to use (default: mock)")
    batch_p.add_argument("--lang", default=None, help="Target language code")
    batch_p.add_argument("--output", default=None, help="Output directory")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "dub":
        _cmd_dub(args)
    elif args.command == "batch":
        _cmd_batch(args)


if __name__ == "__main__":
    main()
