from __future__ import annotations

import argparse
import glob as _glob
import io
import json
import sys
import wave
from pathlib import Path

from dubbing.backends.base import TTSBackend
from dubbing.batch import batch_dub
from dubbing.pipeline import DubbingPipeline


def _make_backend(name: str) -> TTSBackend:
    if name == "say":
        from dubbing.backends.say import SayTTSBackend
        return SayTTSBackend()
    raise SystemExit(f"Unknown backend: {name!r}. Available: say")


def _combine_wav(wav_list: list[bytes]) -> bytes:
    pcm_chunks: list[bytes] = []
    params = None
    for wav_data in wav_list:
        try:
            with wave.open(io.BytesIO(wav_data)) as wf:
                if params is None:
                    params = wf.getparams()
                pcm_chunks.append(wf.readframes(wf.getnframes()))
        except Exception:
            pass
    if not pcm_chunks or params is None:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(b"".join(pcm_chunks))
    return buf.getvalue()


def _cmd_dub(args: argparse.Namespace) -> None:
    backend = _make_backend(args.backend)
    pipeline = DubbingPipeline(backend)
    output = Path(args.output) if args.output else None

    timed, tts_results = pipeline.run_full(Path(args.srt))

    if output:
        output.mkdir(parents=True, exist_ok=True)
        stem = Path(args.srt).stem

        wav_chunks = [r.audio_bytes for r in tts_results if r.audio_bytes]
        combined = _combine_wav(wav_chunks)
        if combined:
            wav_path = output / f"{stem}.wav"
            wav_path.write_bytes(combined)
            print(f"Wrote {wav_path}")

        out_file = output / f"{stem}.json"
        out_file.write_text(
            json.dumps(
                [{"start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.segment.entry.text} for s in timed],
                indent=2,
            )
        )
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
    results = batch_dub(paths, backend, output_dir)
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
    if args.command == "dub":
        _cmd_dub(args)
    elif args.command == "batch":
        _cmd_batch(args)


if __name__ == "__main__":
    main()
