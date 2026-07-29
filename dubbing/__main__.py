from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import subprocess
import sys
from pathlib import Path

from dubbing.aligner import segment_plan
from dubbing.assembler import assemble_timeline
from dubbing.backends.base import TTSBackend
from dubbing.batch import batch_dub
from dubbing.diarization import (
    SherpaOnnxDiarizationBackend,
    SpeakerConstraints,
    attribute_window,
    attributed_segment_plan,
)
from dubbing.pipeline import DubbingPipeline
from dubbing.srt_parser import parse_srt
from dubbing.transcription import (
    DEFAULT_MLX_MODEL,
    AudioUnderstandingPipeline,
    MLXWhisperTranscriptionBackend,
    ResumableTranscriptionJob,
    TranscriptionOptions,
    attribute_transcript,
    transcript_to_srt,
    transcript_to_text,
)


def _make_backend(name: str) -> TTSBackend:
    if name in ("auto", ""):
        from dubbing.backends import select_backend
        return select_backend()
    if name == "say":
        from dubbing.backends.say import SayTTSBackend
        return SayTTSBackend()
    if name == "piper":
        from dubbing.backends.piper import PiperTTSBackend
        return PiperTTSBackend()
    if name == "espeak":
        from dubbing.backends.espeak import EspeakTTSBackend
        return EspeakTTSBackend()
    raise SystemExit(f"Unknown backend: {name!r}. Available: auto, say, piper, espeak")


def _make_diarizer(args: argparse.Namespace) -> SherpaOnnxDiarizationBackend:
    segmentation_model = args.segmentation_model or os.environ.get(
        "DUBBING_DIARIZATION_SEGMENTATION_MODEL"
    )
    embedding_model = args.embedding_model or os.environ.get(
        "DUBBING_DIARIZATION_EMBEDDING_MODEL"
    )
    if not segmentation_model or not embedding_model:
        raise ValueError(
            "Sherpa model paths are required. Pass --segmentation-model and "
            "--embedding-model, or set DUBBING_DIARIZATION_SEGMENTATION_MODEL "
            "and DUBBING_DIARIZATION_EMBEDDING_MODEL."
        )
    return SherpaOnnxDiarizationBackend(
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        device=args.diarization_device,
        num_threads=args.diarization_threads,
        cluster_threshold=args.cluster_threshold,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
    )


def _speaker_constraints(args: argparse.Namespace) -> SpeakerConstraints:
    return SpeakerConstraints(
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )


def _make_transcriber(args: argparse.Namespace) -> MLXWhisperTranscriptionBackend:
    if args.asr_backend != "mlx-whisper":
        raise ValueError(
            f"Unknown ASR backend: {args.asr_backend!r}. Available: mlx-whisper"
        )
    return MLXWhisperTranscriptionBackend(
        model=args.asr_model,
        temperature=args.asr_temperature,
    )


def _cmd_dub(args: argparse.Namespace) -> None:
    backend = _make_backend(args.backend)
    pipeline = DubbingPipeline(backend)
    output = Path(args.output) if args.output else None

    diarization = None
    attributions = None
    if args.source_audio:
        diarizer = _make_diarizer(args)
        timed, tts_results, diarization, attributions = pipeline.run_full_with_diarization(
            Path(args.srt),
            Path(args.source_audio),
            diarizer,
            language=args.lang or "",
            constraints=_speaker_constraints(args),
        )
    else:
        timed, tts_results = pipeline.run_full(Path(args.srt), language=args.lang or "")

    if output:
        output.mkdir(parents=True, exist_ok=True)
        stem = Path(args.srt).stem

        if timed:
            wav_path = output / f"{stem}.wav"
            wav_path.write_bytes(assemble_timeline(timed, tts_results))
            print(f"Wrote {wav_path}")

        plan = (
            attributed_segment_plan(timed, attributions)
            if attributions is not None
            else segment_plan(timed)
        )
        out_file = output / f"{stem}.json"
        out_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"Wrote {out_file}")
        if diarization is not None:
            diarization_file = output / f"{stem}.diarization.json"
            diarization_file.write_text(
                json.dumps(diarization.to_dict(), indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {diarization_file}")
    else:
        for index, seg in enumerate(timed):
            speaker = ""
            if attributions is not None:
                attribution = attributions[index]
                speaker = f" [{attribution.speaker or attribution.status.upper()}]"
            print(f"[{seg.start_ms}–{seg.end_ms}]{speaker} {seg.segment.entry.text}")


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


def _cmd_diarize(args: argparse.Namespace) -> None:
    diarizer = _make_diarizer(args)
    result = diarizer.diarize(Path(args.audio), constraints=_speaker_constraints(args))
    document = result.to_dict()
    document["audio"] = Path(args.audio).name

    if args.srt:
        entries = parse_srt(Path(args.srt))
        attributed = []
        for entry in entries:
            attribution = attribute_window(entry.start_ms, entry.end_ms, result.turns)
            item = attribution.to_dict()
            item["index"] = entry.index
            item["text"] = entry.text
            attributed.append(item)
        document["segments"] = attributed

    payload = json.dumps(document, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        print(payload)


def _cmd_transcribe(args: argparse.Namespace) -> None:
    backend = _make_transcriber(args)
    options = TranscriptionOptions(
        language=args.language,
        task=args.task,
        initial_prompt=args.initial_prompt,
        word_timestamps=not args.no_word_timestamps,
    )
    if args.checkpoint_dir:
        result = ResumableTranscriptionJob(
            backend,
            args.checkpoint_dir,
            chunk_seconds=args.chunk_seconds,
        ).run(args.audio, options)
        if args.diarize:
            diarizer = _make_diarizer(args)
            result = attribute_transcript(
                result,
                diarizer.diarize(
                    Path(args.audio),
                    constraints=_speaker_constraints(args),
                ),
            )
    else:
        diarizer = _make_diarizer(args) if args.diarize else None
        result = AudioUnderstandingPipeline(backend).run(
            args.audio,
            options=options,
            diarizer=diarizer,
            speaker_constraints=_speaker_constraints(args) if diarizer else None,
        )

    if args.format == "json":
        payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"
    elif args.format == "srt":
        payload = transcript_to_srt(result)
    else:
        payload = transcript_to_text(result)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        print(payload, end="")


def _add_diarization_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--segmentation-model", help="Path to Sherpa segmentation ONNX model")
    parser.add_argument("--embedding-model", help="Path to Sherpa speaker embedding ONNX model")
    parser.add_argument(
        "--diarization-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Explicit ONNX execution device (default: cpu)",
    )
    parser.add_argument(
        "--diarization-threads",
        type=int,
        default=2,
        help="Inference threads per model (default: 2)",
    )
    parser.add_argument("--num-speakers", type=int, help="Known exact speaker count")
    parser.add_argument("--min-speakers", type=int, help="Minimum speakers, if supported")
    parser.add_argument("--max-speakers", type=int, help="Maximum speakers, if supported")
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.5,
        help="Automatic clustering threshold when speaker count is unknown",
    )
    parser.add_argument("--min-duration-on", type=float, default=0.3)
    parser.add_argument("--min-duration-off", type=float, default=0.5)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dubbing",
        description="Dubbing Studio — synthesise dubbed audio from SRT files.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="TTS backend to use: auto, say, piper, espeak (default: auto)",
    )
    parser.add_argument("--lang", default=None, help="Target language code")
    parser.add_argument("--output", default=None, help="Output directory")

    sub = parser.add_subparsers(dest="command", required=True)

    dub_p = sub.add_parser("dub", help="Dub a single SRT file")
    dub_p.add_argument("srt", help="Path to the .srt file")
    dub_p.add_argument(
        "--backend",
        default="auto",
        help="TTS backend to use: auto, say, piper, espeak (default: auto)",
    )
    dub_p.add_argument("--lang", default=None, help="Target language code")
    dub_p.add_argument("--output", default=None, help="Output directory")
    dub_p.add_argument(
        "--source-audio",
        help="Source audio/video to diarize and map onto the subtitle plan",
    )
    _add_diarization_options(dub_p)

    batch_p = sub.add_parser("batch", help="Dub multiple SRT files matching a glob")
    batch_p.add_argument("glob", help="Glob pattern matching .srt files")
    batch_p.add_argument(
        "--backend",
        default="auto",
        help="TTS backend to use: auto, say, piper, espeak (default: auto)",
    )
    batch_p.add_argument("--lang", default=None, help="Target language code")
    batch_p.add_argument("--output", default=None, help="Output directory")

    diarize_p = sub.add_parser(
        "diarize",
        help="Diarize local audio/video and optionally attribute an SRT file",
    )
    diarize_p.add_argument("audio", help="Path to source audio or video")
    diarize_p.add_argument("--srt", help="Optional SRT to attribute without changing timing")
    diarize_p.add_argument("--output", help="Output JSON path (default: stdout)")
    _add_diarization_options(diarize_p)

    transcribe_p = sub.add_parser(
        "transcribe",
        help="Transcribe local audio/video with an injectable ASR backend",
    )
    transcribe_p.add_argument("audio", help="Path to source audio or video")
    transcribe_p.add_argument(
        "--asr-backend",
        default="mlx-whisper",
        choices=("mlx-whisper",),
    )
    transcribe_p.add_argument("--asr-model", default=DEFAULT_MLX_MODEL)
    transcribe_p.add_argument("--asr-temperature", type=float, default=0.0)
    transcribe_p.add_argument("--language", help="Optional source language code")
    transcribe_p.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
    )
    transcribe_p.add_argument("--initial-prompt")
    transcribe_p.add_argument("--no-word-timestamps", action="store_true")
    transcribe_p.add_argument(
        "--format",
        choices=("json", "srt", "text"),
        default="json",
    )
    transcribe_p.add_argument("--output", help="Output path (default: stdout)")
    transcribe_p.add_argument(
        "--checkpoint-dir",
        help="Resume-safe checkpoint directory for long recordings",
    )
    transcribe_p.add_argument(
        "--chunk-seconds",
        type=int,
        default=1800,
        help="Checkpoint chunk length, 30-3600 seconds (default: 1800)",
    )
    transcribe_p.add_argument(
        "--diarize",
        action="store_true",
        help="Also run speaker diarisation and attach labels",
    )
    _add_diarization_options(transcribe_p)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "dub":
            _cmd_dub(args)
        elif args.command == "batch":
            _cmd_batch(args)
        elif args.command == "diarize":
            _cmd_diarize(args)
        elif args.command == "transcribe":
            _cmd_transcribe(args)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.SubprocessError) as exc:
        # ValueError: hostile SRT rejected by the renderer (e.g. a timestamp
        # beyond the timeline cap); SubprocessError: `say`/`afconvert`
        # failing or timing out. All are clean errors, never tracebacks.
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
