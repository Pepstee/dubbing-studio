from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from dubbing.apps.personal_capture.config import load_config
from dubbing.apps.personal_capture.model_manifest import verify_model_manifest
from dubbing.diarization import SherpaOnnxDiarizationBackend
from dubbing.media import ffmpeg_executable
from dubbing.transcription import (
    FasterWhisperSileroSpeechRegionDetector,
    FasterWhisperTranscriptionBackend,
)
from dubbing.transcription.job import media_duration_ms
from dubbing.translation import NLLBTranslationBackend


def _check(report: list[dict], name: str, status: str, detail: str) -> None:
    report.append({"name": name, "status": status, "detail": detail})


def _check_model_directory(
    checks: list[dict],
    name: str,
    path: Path,
    *,
    required: tuple[str, ...],
    alternatives: tuple[tuple[str, ...], ...] = (),
) -> None:
    missing = [item for item in required if not (path / item).is_file()]
    for group in alternatives:
        if not any((path / item).is_file() for item in group):
            missing.append("one of " + ", ".join(group))
    valid = path.is_dir() and not missing
    detail = str(path) if valid else f"{path}; missing: {', '.join(missing) or 'directory'}"
    _check(checks, f"model:{name}", "pass" if valid else "fail", detail)


def _probe_diarization(defaults: dict) -> None:
    backend = SherpaOnnxDiarizationBackend(
        defaults["segmentation_model"],
        defaults["embedding_model"],
        device=defaults["diarization_device"],
    )
    with tempfile.TemporaryDirectory(prefix="dubbing-diarization-probe-") as directory:
        sample = Path(directory) / "silence.wav"
        with wave.open(str(sample), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(b"\0\0" * 16_000)
        backend.diarize(sample)


def _probe_vad(defaults: dict) -> None:
    detector = FasterWhisperSileroSpeechRegionDetector(
        strict_threshold=defaults["vad_strict_threshold"],
        sensitive_threshold=defaults["vad_sensitive_threshold"],
        minimum_speech_ms=defaults["vad_minimum_speech_ms"],
        minimum_silence_ms=defaults["vad_minimum_silence_ms"],
        speech_pad_ms=defaults["vad_speech_pad_ms"],
        maximum_region_seconds=defaults["vad_maximum_region_seconds"],
    )
    with tempfile.TemporaryDirectory(prefix="dubbing-vad-probe-") as directory:
        sample = Path(directory) / "silence.wav"
        with wave.open(str(sample), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(b"\0\0" * 16_000)
        detector.detect(sample)


def inspect_environment(
    config_path: str | Path,
    *,
    audio: str | Path | None = None,
    prepare: bool = False,
    load_models: bool = False,
) -> dict:
    config = load_config(config_path)
    workspace = Path(config["workspace"]["wsl_path"])
    inbox = Path(config["landing_inbox"]["wsl_path"])
    paths = config["paths"]
    runtime_dirs = [
        workspace,
        inbox,
        workspace / paths["processing"],
        workspace / paths["packages"],
        workspace / paths["state"],
        Path(config["giga_outbox"]["path"]),
    ]
    if prepare:
        for path in runtime_dirs:
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, 0o700)

    checks: list[dict] = []
    for path in runtime_dirs:
        if path.is_dir() and os.access(path, os.R_OK | os.W_OK | os.X_OK):
            _check(checks, f"path:{path}", "pass", "readable and writable")
        else:
            _check(checks, f"path:{path}", "fail", "missing or not writable")

    ffmpeg = ffmpeg_executable()
    _check(
        checks,
        "ffmpeg",
        "pass" if ffmpeg else "fail",
        str(ffmpeg) if ffmpeg else "not available",
    )
    ffprobe = shutil.which("ffprobe")
    _check(
        checks,
        "ffprobe",
        "pass" if ffprobe else "fail",
        ffprobe or "not available",
    )

    dependencies = {
        "faster_whisper": "Faster-Whisper ASR",
        "lingua": "language detection",
        "numpy": "audio arrays",
        "sherpa_onnx": "speaker diarization",
        "torch": "translation runtime",
        "transformers": "NLLB translation",
        "waitress": "production review server",
    }
    for module, purpose in dependencies.items():
        available = importlib.util.find_spec(module) is not None
        _check(
            checks,
            f"dependency:{module}",
            "pass" if available else "fail",
            purpose if available else f"missing dependency for {purpose}",
        )

    cuda_requested = any(
        config["defaults"].get(key) == "cuda"
        for key in (
            "asr_device",
            "asr_retry_device",
            "translation_device",
            "diarization_device",
        )
    )
    if cuda_requested:
        nvidia_smi = shutil.which("nvidia-smi")
        gpu_detail = "nvidia-smi not available"
        gpu_ok = False
        if nvidia_smi:
            process = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            gpu_ok = process.returncode == 0 and bool(process.stdout.strip())
            gpu_detail = process.stdout.strip() or process.stderr.strip() or gpu_detail
        _check(
            checks,
            "cuda-device",
            "pass" if gpu_ok else "fail",
            gpu_detail,
        )

    for key in ("segmentation_model", "embedding_model"):
        model = Path(config["defaults"][key])
        _check(
            checks,
            f"model:{key}",
            "pass" if model.is_file() else "fail",
            str(model),
        )
    _check_model_directory(
        checks,
        "asr_model",
        Path(config["defaults"]["asr_model"]),
        required=("config.json", "model.bin"),
        alternatives=(("tokenizer.json", "vocabulary.txt"),),
    )
    _check_model_directory(
        checks,
        "asr_retry_model",
        Path(config["defaults"]["asr_retry_model"]),
        required=("config.json", "model.bin"),
        alternatives=(("tokenizer.json", "vocabulary.txt"),),
    )
    _check_model_directory(
        checks,
        "translation_model",
        Path(config["defaults"]["translation_model"]),
        required=("config.json",),
        alternatives=(
            ("model.safetensors", "pytorch_model.bin"),
            ("sentencepiece.bpe.model", "tokenizer.json"),
        ),
    )
    try:
        model_manifest = verify_model_manifest(config)
    except ValueError as exc:
        model_manifest = None
        _check(checks, "model-manifest", "fail", str(exc))
    else:
        revisions = {
            name: entry["revision"]
            for name, entry in model_manifest["models"].items()
        }
        _check(
            checks,
            "model-manifest",
            "pass",
            json.dumps(revisions, sort_keys=True),
        )

    if load_models and model_manifest is not None:
        defaults = config["defaults"]
        probes = (
            (
                "asr-load",
                lambda: FasterWhisperTranscriptionBackend(
                    defaults["asr_model"],
                    device=defaults["asr_device"],
                    compute_type=defaults["asr_compute_type"],
                    local_files_only=True,
                )._load_model(),
            ),
            (
                "asr-retry-load",
                lambda: FasterWhisperTranscriptionBackend(
                    defaults["asr_retry_model"],
                    device=defaults["asr_retry_device"],
                    compute_type=defaults["asr_retry_compute_type"],
                    local_files_only=True,
                    model_revision=model_manifest["models"]["asr_retry"]["revision"],
                )._load_model(),
            ),
            (
                "vad-load",
                lambda: _probe_vad(defaults),
            ),
            (
                "translation-load",
                lambda: NLLBTranslationBackend(
                    defaults["translation_model"],
                    device=defaults["translation_device"],
                    local_files_only=True,
                )._load(),
            ),
            (
                "diarization-load",
                lambda: _probe_diarization(defaults),
            ),
        )
        for name, probe in probes:
            try:
                probe()
            except Exception as exc:
                _check(checks, f"model-probe:{name}", "fail", f"{type(exc).__name__}: {exc}")
            else:
                _check(checks, f"model-probe:{name}", "pass", "loaded offline")
    elif not load_models:
        _check(
            checks,
            "model-load-certification",
            "fail",
            "not run; repeat preflight with --load-models before admitting audio",
        )

    reserve = int(config["defaults"]["minimum_free_bytes"])
    if workspace.exists():
        free = shutil.disk_usage(workspace).free
        _check(
            checks,
            "workspace-free-space",
            "pass" if free >= reserve else "fail",
            f"{free} bytes free; {reserve} required reserve",
        )

    token = Path(config["network"]["token_file"])
    try:
        token_ok = (
            token.is_file()
            and len(token.read_text(encoding="utf-8").strip()) >= 32
            and token.stat().st_mode & 0o077 == 0
        )
    except OSError:
        token_ok = False
    _check(
        checks,
        "review-token",
        "pass" if token_ok else "fail",
        str(token),
    )

    audio_document = None
    if audio is not None:
        source = Path(audio).resolve()
        if not source.is_file() or source.is_symlink():
            _check(checks, "audio-source", "fail", "not a regular file")
        else:
            duration_ms = media_duration_ms(source)
            maximum_ms = int(config["defaults"]["maximum_audio_seconds"]) * 1000
            status = "pass" if duration_ms <= maximum_ms else "fail"
            _check(
                checks,
                "audio-duration",
                status,
                f"{duration_ms} ms; maximum {maximum_ms} ms",
            )
            web_limit = int(config["network"]["max_upload_bytes"])
            if source.stat().st_size > web_limit:
                _check(
                    checks,
                    "web-upload-size",
                    "warn",
                    "use atomic filesystem transfer; file exceeds browser upload limit",
                )
            else:
                _check(
                    checks,
                    "web-upload-size",
                    "pass",
                    "file fits configured browser upload limit",
                )
            audio_document = {
                "path": str(source),
                "size": source.stat().st_size,
                "duration_ms": duration_ms,
            }

    ready = not any(item["status"] == "fail" for item in checks)
    return {
        "schema_version": "dubbing.personal-capture-preflight.v1",
        "ready": ready,
        "model_load_certified": load_models and ready,
        "checks": checks,
        "audio": audio_document,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify a Personal Capture host before admitting audio"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="load every configured model offline; required for a ready report",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inspect_environment(
        args.config,
        audio=args.audio,
        prepare=args.prepare,
        load_models=args.load_models,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
