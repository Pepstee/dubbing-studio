from __future__ import annotations

import argparse
import glob as _glob
import json
import subprocess
import sys
from pathlib import Path

from dubbing.aligner import segment_plan
from dubbing.assembler import assemble_timeline
from dubbing.backends.base import TTSBackend
from dubbing.batch import batch_dub
from dubbing.pipeline import DubbingPipeline


def _make_backend(name: str) -> TTSBackend:
    if name == "say":
        from dubbing.backends.say import SayTTSBackend
        return SayTTSBackend()
    raise SystemExit(f"Unknown backend: {name!r}. Available: say")


def _cmd_dub(args: argparse.Namespace) -> None:
    backend = _make_backend(args.backend)
    pipeline = DubbingPipeline(backend)
    output = Path(args.output) if args.output else None

    timed, tts_results = pipeline.run_full(Path(args.srt), language=args.lang or "")

    if output:
        output.mkdir(parents=True, exist_ok=True)
        stem = Path(args.srt).stem

        if timed:
            wav_path = output / f"{stem}.wav"
            wav_path.write_bytes(assemble_timeline(timed, tts_results))
            print(f"Wrote {wav_path}")

        out_file = output / f"{stem}.json"
        out_file.write_text(json.dumps(segment_plan(timed), indent=2), encoding="utf-8")
        print(f"Wrote {out_file}")
    else:
        for seg in timed:
            print(f"[{seg.start_ms}–{seg.end_ms}] {seg.segment.entry.text}")


def _cmd_batch(args: argparse.Namespace) -> None:
    backend = _make_backend(args.backend)
    paths = [Path(p) for p in _glob.glob(args.glob, recursive=True)]
    if not paths:
        print(f"No files matched: {args.glob}", file=sys.stderr)
        sys.exit(1)
    output_dir = args.output or "."
    results = batch_dub(paths, backend, output_dir, language=args.lang or "")
    for path, segs in results.items():
        print(f"{path}: {len(segs)} segment(s)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dubbing",
        description="Dubbing Studio — synthesise dubbed audio from SRT files.",
    )
    parser.add_argument("--backend", default="say", help="TTS backend to use (default: say)")
    parser.add_argument("--lang", default=None, help="Target language code")
    parser.add_argument("--output", default=None, help="Output directory")

    sub = parser.add_subparsers(dest="command", required=True)

    dub_p = sub.add_parser("dub", help="Dub a single SRT file")
    dub_p.add_argument("srt", help="Path to the .srt file")
    dub_p.add_argument("--backend", default="say", help="TTS backend to use (default: say)")
    dub_p.add_argument("--lang", default=None, help="Target language code")
    dub_p.add_argument("--output", default=None, help="Output directory")

    batch_p = sub.add_parser("batch", help="Dub multiple SRT files matching a glob")
    batch_p.add_argument("glob", help="Glob pattern matching .srt files")
    batch_p.add_argument("--backend", default="say", help="TTS backend to use (default: say)")
    batch_p.add_argument("--lang", default=None, help="Target language code")
    batch_p.add_argument("--output", default=None, help="Output directory")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "dub":
            _cmd_dub(args)
        elif args.command == "batch":
            _cmd_batch(args)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.SubprocessError) as exc:
        # ValueError: hostile SRT rejected by the renderer (e.g. a timestamp
        # beyond the timeline cap); SubprocessError: `say`/`afconvert`
        # failing or timing out. All are clean errors, never tracebacks.
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
