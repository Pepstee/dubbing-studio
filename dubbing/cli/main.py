from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import subprocess
import sys
from pathlib import Path

from dubbing.aligner import segment_plan
from dubbing.assembler import assemble_timeline
from dubbing.batch import batch_dub
from dubbing.apps.personal_capture import CaptureService
from dubbing.apps.personal_capture.config import load_config
from dubbing.apps.personal_capture.runtime import build_control_service
from dubbing.diarization import (
    ResumableDiarizationJob,
    attribute_window,
    attributed_segment_plan,
)
from dubbing.cli.factories import (
    make_diarizer,
    make_transcriber,
    make_tts_backend,
    speaker_aliases,
    speaker_constraints,
)
from dubbing.pipeline import DubbingPipeline
from dubbing.models import (
    DEFAULT_MAX_SEGMENTS,
    DEFAULT_MAX_SYNTHESIS_SECONDS,
    JobConfig,
    SegmentLimitExceeded,
    SynthesisTimeBudgetExceeded,
)
from dubbing.srt_parser import parse_srt
from dubbing.transcription import (
    AudioUnderstandingPipeline,
    ResumableTranscriptionJob,
    TranscriptionOptions,
    attribute_transcript,
    transcript_to_srt,
    transcript_to_text,
)
from dubbing.translation import LinguaLanguageDetector, NLLBTranslationBackend


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer value: {value!r}") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {parsed}")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from None
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative finite number, got {parsed}")
    return parsed


def _make_job_config(args: argparse.Namespace) -> JobConfig:
    return JobConfig(
        max_segments=(args.max_segments if args.max_segments is not None else DEFAULT_MAX_SEGMENTS),
        max_synthesis_seconds=(
            args.max_synthesis_seconds
            if args.max_synthesis_seconds is not None
            else DEFAULT_MAX_SYNTHESIS_SECONDS
        ),
    )


def _add_job_limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-segments",
        type=_non_negative_int,
        default=None,
        help=f"Maximum allowed segment count (default: {DEFAULT_MAX_SEGMENTS})",
    )
    parser.add_argument(
        "--max-synthesis-seconds",
        type=_non_negative_float,
        default=None,
        help=(
            "Maximum estimated and wall-clock synthesis time in seconds "
            f"(default: {DEFAULT_MAX_SYNTHESIS_SECONDS})"
        ),
    )


def _cmd_dub(args: argparse.Namespace) -> None:
    backend = make_tts_backend(args.backend)
    pipeline = DubbingPipeline(backend, _make_job_config(args))
    output = Path(args.output) if args.output else None

    diarization = None
    attributions = None
    if args.source_audio:
        diarizer = make_diarizer(args)
        timed, tts_results, diarization, attributions = pipeline.run_full_with_diarization(
            Path(args.srt),
            Path(args.source_audio),
            diarizer,
            language=args.lang or "",
            constraints=speaker_constraints(args),
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
    backend = make_tts_backend(args.backend)
    paths = [Path(p) for p in _glob.glob(args.glob, recursive=True)]
    if not paths:
        print(f"No files matched: {args.glob}", file=sys.stderr)
        sys.exit(1)
    output_dir = args.output or "."
    results = batch_dub(
        paths,
        backend,
        output_dir,
        language=args.lang or "",
        job_config=_make_job_config(args),
    )
    for path, segs in results.items():
        print(f"{path}: {len(segs)} segment(s)")


def _cmd_diarize(args: argparse.Namespace) -> None:
    diarizer = make_diarizer(args)
    if args.checkpoint_dir:
        result = ResumableDiarizationJob(
            diarizer,
            args.checkpoint_dir,
            chunk_seconds=args.diarization_chunk_seconds,
        ).run(Path(args.audio), constraints=speaker_constraints(args))
    else:
        result = diarizer.diarize(
            Path(args.audio),
            constraints=speaker_constraints(args),
        )
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
    backend = make_transcriber(args)
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
            overlap_seconds=args.overlap_seconds,
        ).run(args.audio, options)
        if args.diarize:
            diarizer = make_diarizer(args)
            result = attribute_transcript(
                result,
                ResumableDiarizationJob(
                    diarizer,
                    Path(args.checkpoint_dir) / "diarization",
                    chunk_seconds=args.diarization_chunk_seconds,
                ).run(
                    Path(args.audio),
                    constraints=speaker_constraints(args),
                ),
            )
    else:
        diarizer = make_diarizer(args) if args.diarize else None
        result = AudioUnderstandingPipeline(backend).run(
            args.audio,
            options=options,
            diarizer=diarizer,
            speaker_constraints=speaker_constraints(args) if diarizer else None,
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


def _cmd_capture(args: argparse.Namespace) -> None:
    if args.capture_command == "approve":
        service = (
            build_control_service(load_config(args.config))
            if args.config
            else CaptureService(args.workspace)
        )
        event = service.approve(
            args.capture_id,
            speaker_aliases=speaker_aliases(args.speaker),
            notes=args.notes,
            diarization_review_acknowledged=args.diarization_review_acknowledged,
        )
        print(f"Approved {args.capture_id}; wrote {event}")
        return
    if args.capture_command == "list":
        service = (
            build_control_service(load_config(args.config))
            if args.config
            else CaptureService(args.workspace)
        )
        for record in service.records():
            print(
                f"{record.capture_id} {record.state} attempts={record.attempt_count} "
                f"{record.source_name}"
            )
        return

    backend = make_transcriber(args)
    diarizer = make_diarizer(args) if args.diarize else None
    detector = LinguaLanguageDetector() if args.translate_to else None
    translator = (
        NLLBTranslationBackend(
            model=args.translation_model,
            device=args.translation_device,
        )
        if args.translate_to
        else None
    )
    service = CaptureService(
        args.workspace,
        backend,
        diarizer=diarizer,
        speaker_constraints=speaker_constraints(args) if diarizer else None,
        transcription_options=TranscriptionOptions(language=args.language),
        language_detector=detector,
        translation_backend=translator,
        translation_target=args.translate_to or "en",
        processing_dir=args.processing_dir,
        transcription_chunk_seconds=args.transcription_chunk_seconds,
        transcription_overlap_seconds=args.transcription_overlap_seconds,
        diarization_chunk_seconds=args.diarization_chunk_seconds,
        minimum_free_bytes=args.minimum_free_bytes,
        maximum_audio_seconds=args.maximum_audio_seconds,
    )
    outcomes = service.scan(args.inbox, min_age_seconds=args.min_age_seconds)
    for outcome in outcomes:
        suffix = f" error={outcome.error}" if outcome.error else ""
        replayed = " replayed" if outcome.replayed else ""
        print(f"{outcome.capture_id} {outcome.state}{replayed} {outcome.source_name}{suffix}")


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
    _add_job_limits(dub_p)

    batch_p = sub.add_parser("batch", help="Dub multiple SRT files matching a glob")
    batch_p.add_argument("glob", help="Glob pattern matching .srt files")
    batch_p.add_argument(
        "--backend",
        default="auto",
        help="TTS backend to use: auto, say, piper, espeak (default: auto)",
    )
    batch_p.add_argument("--lang", default=None, help="Target language code")
    batch_p.add_argument("--output", default=None, help="Output directory")
    _add_job_limits(batch_p)

    diarize_p = sub.add_parser(
        "diarize",
        help="Diarize local audio/video and optionally attribute an SRT file",
    )
    diarize_p.add_argument("audio", help="Path to source audio or video")
    diarize_p.add_argument("--srt", help="Optional SRT to attribute without changing timing")
    diarize_p.add_argument("--output", help="Output JSON path (default: stdout)")
    diarize_p.add_argument(
        "--checkpoint-dir",
        help="Resume-safe checkpoint directory for long recordings",
    )
    diarize_p.add_argument(
        "--diarization-chunk-seconds",
        type=int,
        default=7200,
        help="Long-audio diarization chunk length (default: 7200)",
    )
    _add_diarization_options(diarize_p)

    transcribe_p = sub.add_parser(
        "transcribe",
        help="Transcribe local audio/video with an injectable ASR backend",
    )
    transcribe_p.add_argument("audio", help="Path to source audio or video")
    transcribe_p.add_argument(
        "--asr-backend",
        default="mlx-whisper",
        choices=("mlx-whisper", "faster-whisper"),
    )
    transcribe_p.add_argument(
        "--asr-model",
        help="Model name/path (backend default when omitted)",
    )
    transcribe_p.add_argument(
        "--asr-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Faster-Whisper device (default: auto)",
    )
    transcribe_p.add_argument(
        "--asr-compute-type",
        default="default",
        help="CTranslate2 compute type, e.g. float16 or int8_float16",
    )
    transcribe_p.add_argument("--asr-cpu-threads", type=int, default=0)
    transcribe_p.add_argument("--asr-workers", type=int, default=1)
    transcribe_p.add_argument(
        "--asr-temperature",
        type=float,
        default=None,
        help=(
            "Force one decoding temperature. By default MLX Whisper uses its "
            "anti-repetition fallback sequence and Faster-Whisper uses 0.0"
        ),
    )
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
        "--overlap-seconds",
        type=int,
        default=5,
        help="Audio overlap around transcription chunks (default: 5)",
    )
    transcribe_p.add_argument(
        "--diarize",
        action="store_true",
        help="Also run speaker diarisation and attach labels",
    )
    transcribe_p.add_argument(
        "--diarization-chunk-seconds",
        type=int,
        default=7200,
        help="Long-audio diarization chunk length (default: 7200)",
    )
    _add_diarization_options(transcribe_p)

    capture_p = sub.add_parser(
        "capture",
        help="Process a personal audio inbox into reviewable evidence packages",
    )
    capture_sub = capture_p.add_subparsers(dest="capture_command", required=True)
    capture_scan = capture_sub.add_parser("scan", help="Process stable media in an inbox")
    capture_scan.add_argument("inbox")
    capture_scan.add_argument("--workspace", required=True)
    capture_scan.add_argument("--min-age-seconds", type=float, default=30)
    capture_scan.add_argument("--processing-dir", default="processing")
    capture_scan.add_argument(
        "--transcription-chunk-seconds",
        type=int,
        default=1800,
    )
    capture_scan.add_argument(
        "--transcription-overlap-seconds",
        type=int,
        default=5,
    )
    capture_scan.add_argument(
        "--diarization-chunk-seconds",
        type=int,
        default=7200,
    )
    capture_scan.add_argument(
        "--minimum-free-bytes",
        type=int,
        default=5 * 1024 * 1024 * 1024,
    )
    capture_scan.add_argument(
        "--maximum-audio-seconds",
        type=int,
        default=86400,
    )
    capture_scan.add_argument(
        "--asr-backend",
        default="faster-whisper",
        choices=("mlx-whisper", "faster-whisper"),
    )
    capture_scan.add_argument("--asr-model")
    capture_scan.add_argument("--asr-device", choices=("auto", "cpu", "cuda"), default="auto")
    capture_scan.add_argument("--asr-compute-type", default="default")
    capture_scan.add_argument("--asr-cpu-threads", type=int, default=0)
    capture_scan.add_argument("--asr-workers", type=int, default=1)
    capture_scan.add_argument("--asr-temperature", type=float, default=0.0)
    capture_scan.add_argument("--language")
    capture_scan.add_argument("--diarize", action="store_true")
    capture_scan.add_argument(
        "--translate-to",
        choices=("en", "ko", "ro", "ru"),
        default="en",
    )
    capture_scan.add_argument(
        "--translation-model",
        default="facebook/nllb-200-distilled-600M",
    )
    capture_scan.add_argument(
        "--translation-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    _add_diarization_options(capture_scan)

    capture_approve = capture_sub.add_parser(
        "approve",
        help="Approve one reviewed capture and emit its GIGA event",
    )
    capture_approve.add_argument("capture_id")
    capture_approve_location = capture_approve.add_mutually_exclusive_group(required=True)
    capture_approve_location.add_argument("--workspace")
    capture_approve_location.add_argument(
        "--config",
        help="Deployment config; also publishes to its verified GIGA outbox",
    )
    capture_approve.add_argument("--speaker", action="append", default=[])
    capture_approve.add_argument("--notes", default="")
    capture_approve.add_argument(
        "--diarization-review-acknowledged",
        action="store_true",
        help=("Confirm manual review/correction of HUMAN_REVIEW_REQUIRED speaker attribution"),
    )

    capture_list = capture_sub.add_parser("list", help="List capture-ledger state")
    capture_list_location = capture_list.add_mutually_exclusive_group(required=True)
    capture_list_location.add_argument("--workspace")
    capture_list_location.add_argument("--config")

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
        elif args.command == "capture":
            _cmd_capture(args)
    except (SegmentLimitExceeded, SynthesisTimeBudgetExceeded) as exc:
        raise SystemExit(
            f"error: {exc} "
            f"(error_code={exc.error_code}, limit={exc.limit}, requested={exc.requested})"
        ) from exc
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.SubprocessError) as exc:
        # ValueError: hostile SRT rejected by the renderer (e.g. a timestamp
        # beyond the timeline cap); SubprocessError: `say`/`afconvert`
        # failing or timing out. All are clean errors, never tracebacks.
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
