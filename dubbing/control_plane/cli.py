from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from dubbing.transcription import (
    FasterWhisperTranscriptionBackend,
    MLXWhisperTranscriptionBackend,
    TranscriptionOptions,
)
from dubbing.transcription.adaptive import (
    AdaptiveChunkPlanner,
    AdaptiveLongFormCoordinator,
    detect_silence_intervals,
    probe_media,
)
from dubbing.transcription.whisperkit import (
    WhisperKitServerProcess,
    WhisperKitTranscriptionBackend,
)


def _cached_hugging_face_snapshot(model: str) -> Path | None:
    """Resolve a local Hugging Face model without contacting the network."""
    supplied = Path(model).expanduser()
    if supplied.is_dir():
        return supplied.resolve()
    if "/" not in model:
        return None
    repository = Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--"))
    snapshots = repository / "snapshots"
    main_ref = repository / "refs/main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if revision and candidate.is_dir():
            return candidate.resolve()
    candidates = tuple(path for path in snapshots.glob("*") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0].resolve()
    return None


def discover_whisperkit_models() -> tuple[Path, ...]:
    roots = (
        Path.home() / "Library/Application Support/DubbingStudio/models/whisperkit",
        Path.home() / "Library/Application Support/MacWhisper/models/whisperkit/models",
        Path.home() / ".cache/huggingface/hub",
    )
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        for config in root.rglob("config.json"):
            parent = config.parent
            if "whisper" in str(parent).casefold() and any(parent.rglob("*.mlmodelc")):
                candidates.append(parent.resolve())
    return tuple(sorted(set(candidates)))


def build_backend(args: argparse.Namespace):
    if args.backend == "mlx":
        options = {"model": args.model or "mlx-community/whisper-large-v3-turbo"}
        if args.mlx_temperature is not None:
            options["temperature"] = tuple(args.mlx_temperature)
        return MLXWhisperTranscriptionBackend(**options)
    if args.backend == "faster-whisper":
        return FasterWhisperTranscriptionBackend(
            model=args.model or "large-v3-turbo",
            device=args.device,
            compute_type=args.compute_type,
            local_files_only=args.local_files_only,
        )
    models = discover_whisperkit_models()
    model_path = (
        Path(args.model_path).resolve() if args.model_path else (models[0] if models else None)
    )
    if model_path is None:
        raise SystemExit(
            "No existing WhisperKit Core ML asset was found. A model download is an explicit operator gate."
        )
    model = args.model or model_path.name.removeprefix("openai_whisper-")
    server = None
    if args.start_server:
        executable = (
            args.whisperkit_cli or shutil.which("argmax-cli") or shutil.which("whisperkit-cli")
        )
        if not executable:
            size = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
            raise SystemExit(
                "Official WhisperKit CLI is not installed; no download was performed. "
                f"Reusable model asset: {model_path} ({size / 1024 / 1024:.1f} MiB)."
            )
        server = WhisperKitServerProcess(
            executable=executable,
            model_path=model_path,
            port=args.port,
        )
    return WhisperKitTranscriptionBackend(
        model=model,
        endpoint=args.server_url or f"http://127.0.0.1:{args.port}",
        server=server,
    )


def build_retry_backend(args: argparse.Namespace):
    """Build one independent, offline retry backend for coordinator reuse."""
    if args.retry_backend is None:
        return None
    default_model = (
        "mlx-community/whisper-large-v3-turbo" if args.retry_backend == "mlx" else "large-v3-turbo"
    )
    requested_model = args.retry_model or default_model
    local_model = _cached_hugging_face_snapshot(requested_model)
    if local_model is None:
        raise SystemExit(
            f"Independent {args.retry_backend} retry model is not available locally: "
            f"{requested_model!r}. No download was performed; provide --retry-model "
            "with an existing model directory or populate the Hugging Face cache under "
            "a separate, explicitly authorized download gate."
        )
    if args.retry_backend == "mlx":
        options = {"model": str(local_model)}
        if args.retry_mlx_temperature is not None:
            options["temperature"] = tuple(args.retry_mlx_temperature)
        return MLXWhisperTranscriptionBackend(**options)
    return FasterWhisperTranscriptionBackend(
        model=str(local_model),
        device=args.retry_device,
        compute_type=args.retry_compute_type,
        local_files_only=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed adaptive long-video transcription")
    parser.add_argument("media")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backend",
        choices=("whisperkit", "mlx", "faster-whisper"),
        default="whisperkit",
    )
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--language")
    parser.add_argument("--target-seconds", type=int, default=240)
    parser.add_argument("--minimum-seconds", type=int, default=60)
    parser.add_argument("--maximum-seconds", type=int, default=480)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-silence-seconds", type=float, default=0.7)
    parser.add_argument(
        "--language-retry-policy",
        action="append",
        default=[],
        metavar="LANG=MODE",
        help="Reference-independent detected-language retry; MODE is always or confidence.",
    )
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-url")
    parser.add_argument("--port", type=int, default=50060)
    parser.add_argument("--whisperkit-cli")
    parser.add_argument(
        "--mlx-temperature",
        action="append",
        type=float,
        help=(
            "MLX decode temperature; repeat to configure fallbacks. "
            "Use once with 0 for deterministic fail-fast adjudication."
        ),
    )
    parser.add_argument(
        "--retry-backend",
        choices=("mlx", "faster-whisper"),
        help=(
            "Independent local backend for rejected-span adjudication. The model must "
            "already exist locally; this command never downloads it."
        ),
    )
    parser.add_argument(
        "--retry-model",
        help=(
            "Existing local retry model directory or an already-cached Hugging Face "
            "model identifier."
        ),
    )
    parser.add_argument(
        "--retry-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for an independent Faster-Whisper retry backend.",
    )
    parser.add_argument(
        "--retry-compute-type",
        default="default",
        help="Compute type for an independent Faster-Whisper retry backend.",
    )
    parser.add_argument(
        "--retry-mlx-temperature",
        action="append",
        type=float,
        help=(
            "Independent MLX retry temperature; repeat to configure fallbacks. "
            "Use once with 0 for deterministic fail-fast adjudication."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    language_retry_policy = {}
    for value in args.language_retry_policy:
        if "=" not in value:
            raise SystemExit("--language-retry-policy must use LANG=MODE")
        language, mode = value.split("=", 1)
        if language in language_retry_policy:
            raise SystemExit(f"duplicate language retry policy: {language}")
        language_retry_policy[language] = mode

    source = Path(args.media).resolve()
    output = Path(args.output).resolve()
    planner = AdaptiveChunkPlanner(
        target_seconds=args.target_seconds,
        minimum_seconds=args.minimum_seconds,
        maximum_seconds=args.maximum_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    if args.dry_run:
        probe = probe_media(source)
        silence_intervals = detect_silence_intervals(
            source, minimum_silence_seconds=args.minimum_silence_seconds
        )
        silence_centres = tuple(
            round((start_ms + end_ms) / 2) for start_ms, end_ms in silence_intervals
        )
        document = {
            "schema_version": "dubbing.adaptive-plan-preview.v1",
            "source": source.name,
            "probe": probe.to_dict(),
            "planner": planner.to_dict(),
            "chunks": [
                item.to_dict()
                for item in planner.plan(
                    probe.duration_ms,
                    silence_centres,
                    silence_intervals=silence_intervals,
                )
            ],
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "plan-preview.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(document, indent=2, sort_keys=True))
        return

    backend = build_backend(args)
    retry_backend = build_retry_backend(args)
    if retry_backend is not None and retry_backend.identity == backend.identity:
        raise SystemExit(
            "--retry-backend resolved to the primary backend identity; targeted retry "
            "requires an independent implementation or model configuration"
        )
    coordinator = AdaptiveLongFormCoordinator(
        backend,
        output,
        planner=planner,
        retry_backend=retry_backend,
        minimum_silence_seconds=args.minimum_silence_seconds,
        language_retry_policy=language_retry_policy,
    )
    started = time.monotonic()
    result, quality = coordinator.run(
        source,
        TranscriptionOptions(language=args.language, task="transcribe", word_timestamps=True),
    )
    runtime_seconds = round(time.monotonic() - started, 3)
    run_receipt = {
        "schema_version": "dubbing.long-transcription-run-receipt.v1",
        "source_sha256": result.source_sha256,
        "backend_identity": backend.identity,
        "retry_backend_identity": (retry_backend.identity if retry_backend is not None else None),
        "runtime_seconds": runtime_seconds,
        "quality_status": quality["status"],
        "segment_count": len(result.segments),
        "cloud_allowed": False,
    }
    (output / "run-receipt.json").write_text(
        json.dumps(run_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": str(output / "result.json"),
                "quality_report": str(output / "quality-report.json"),
                "quality_status": quality["status"],
                "segments": len(result.segments),
                "runtime_seconds": runtime_seconds,
            },
            indent=2,
        )
    )
    if quality["status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
