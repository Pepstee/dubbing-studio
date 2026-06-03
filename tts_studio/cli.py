from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tts_studio.mock_backend import MockTTSBackend
from tts_studio.models import TTSBackend
from tts_studio.pipeline import DubbingPipeline
from tts_studio.srt_parser import parse_srt


def _make_backend(name: str) -> TTSBackend:
    if name == "mock":
        return MockTTSBackend()
    print(f"Unknown backend: {name!r}. Available: mock", file=sys.stderr)
    sys.exit(1)


def _cmd_dub(args: argparse.Namespace) -> None:
    srt_path = Path(args.srt)
    if not srt_path.exists():
        print(f"SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)

    backend = _make_backend(args.backend)
    lang = args.lang or "en"
    segments = parse_srt(srt_path, default_language=lang)
    pipeline = DubbingPipeline(backend, default_language=lang)
    results = pipeline.run(segments)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / (srt_path.stem + ".json")
        records = [
            {"start_ms": seg.start_ms, "end_ms": seg.end_ms, "text": seg.text, "duration_ms": result.duration_ms}
            for seg, result in zip(segments, results)
        ]
        out_file.write_text(json.dumps(records, indent=2))
        print(f"Wrote {out_file}")
    else:
        for seg, result in zip(segments, results):
            print(f"[{seg.start_ms}–{seg.end_ms}] {seg.text}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-studio",
        description="TTS Studio — synthesise dubbed audio from SRT files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dub_p = sub.add_parser("dub", help="Dub a single SRT file")
    dub_p.add_argument("srt", help="Path to the .srt file")
    dub_p.add_argument("--lang", default="en", help="Target language code (default: en)")
    dub_p.add_argument("--backend", default="mock", help="TTS backend to use (default: mock)")
    dub_p.add_argument("--out", default=None, help="Output directory")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "dub":
        _cmd_dub(args)


if __name__ == "__main__":
    main()
