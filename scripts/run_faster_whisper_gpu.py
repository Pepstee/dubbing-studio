from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from dubbing.transcription.windows_cuda import configure_nvidia_dlls

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot() -> dict | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        row = subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip().split(",")
    except (OSError, subprocess.SubprocessError):
        return None
    if len(row) != 5:
        return None
    return {
        "name": row[0].strip(),
        "memory_total_mib": int(row[1]),
        "memory_used_mib": int(row[2]),
        "memory_free_mib": int(row[3]),
        "utilization_percent": int(row[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-bound faster-whisper GPU runner")
    parser.add_argument("audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--bound-source-sha256")
    parser.add_argument(
        "--temperatures",
        default="0",
        help="Comma-separated decode temperatures; default 0 disables escalation.",
    )
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    model_path = Path(args.model).resolve()
    output = Path(args.output).resolve()
    if not audio.is_file():
        raise SystemExit(f"audio not found: {audio}")
    if not model_path.is_dir():
        raise SystemExit(f"model not found: {model_path}")
    if args.bound_source_sha256 and not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.bound_source_sha256
    ):
        raise SystemExit("--bound-source-sha256 must be a SHA-256 hex digest")
    temperatures = tuple(float(item) for item in args.temperatures.split(","))
    if not temperatures or any(value < 0 for value in temperatures):
        raise SystemExit("--temperatures must contain non-negative numbers")
    temperature = temperatures[0] if len(temperatures) == 1 else temperatures

    dll_directories = configure_nvidia_dlls()
    from faster_whisper import WhisperModel

    started = time.monotonic()
    gpu_before = gpu_snapshot()
    model_started = time.monotonic()
    model = WhisperModel(
        str(model_path),
        device="cuda",
        compute_type=args.compute_type,
        local_files_only=True,
    )
    model_load_seconds = time.monotonic() - model_started
    decode_started = time.monotonic()
    segments_iterator, info = model.transcribe(
        str(audio),
        beam_size=args.beam_size,
        word_timestamps=True,
        vad_filter=not args.no_vad,
        condition_on_previous_text=False,
        multilingual=True,
        temperature=temperature,
    )
    segments = []
    texts = []
    for segment in segments_iterator:
        text = segment.text.strip()
        if not text:
            continue
        words = [
            {
                "start_ms": round(word.start * 1000),
                "end_ms": round(word.end * 1000),
                "text": word.word,
                "confidence": word.probability,
                "speaker": None,
            }
            for word in (segment.words or ())
            if word.end > word.start and word.word
        ]
        segments.append(
            {
                "start_ms": round(segment.start * 1000),
                "end_ms": round(segment.end * 1000),
                "text": text,
                "confidence": None,
                "speaker": None,
                "speakers": [],
                "speaker_status": "not_requested",
                "language": None,
                "language_confidence": None,
                "uncertain": False,
                "diagnostics": {
                    "compression_ratio": segment.compression_ratio,
                    "avg_log_probability": segment.avg_logprob,
                    "no_speech_probability": segment.no_speech_prob,
                    "temperature": segment.temperature,
                    "fallback_history": [],
                    "language_probabilities": None,
                    "fallback_exhausted": False,
                    "backend_metadata": {"seek": segment.seek},
                },
                "words": words,
            }
        )
        texts.append(text)
    decode_seconds = time.monotonic() - decode_started
    input_audio_digest = sha256(audio)
    source_digest = (args.bound_source_sha256 or input_audio_digest).lower()
    document = {
        "schema_version": "dubbing.transcription.v1",
        "backend": "faster-whisper",
        "model": model_path.name,
        "device": "cuda",
        "language": info.language,
        "duration_ms": round(info.duration * 1000),
        "confidence_available": True,
        "source_sha256": source_digest,
        "text": " ".join(texts),
        "segments": segments,
        "diagnostics": {
            "language_probability": info.language_probability,
            "duration_after_vad_ms": round(info.duration_after_vad * 1000),
            "vad_filter": not args.no_vad,
            "multilingual": True,
            "beam_size": args.beam_size,
            "configured_temperatures": list(temperatures),
        },
        "provenance": {
            "provider": "faster-whisper",
            "compute_type": args.compute_type,
            "local_files_only": True,
            "cloud_allowed": False,
        },
    }
    receipt = {
        "schema_version": "dubbing.gpu-asr-run-receipt.v1",
        "source_sha256": source_digest,
        "input_audio_sha256": input_audio_digest,
        "model_path": str(model_path),
        "model_bin_sha256": sha256(model_path / "model.bin"),
        "compute_type": args.compute_type,
        "model_load_seconds": round(model_load_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "audio_duration_seconds": info.duration,
        "realtime_factor": round(decode_seconds / info.duration, 6),
        "segment_count": len(segments),
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "cloud_allowed": False,
        "giga_admission_emitted": False,
        "nvidia_dll_directories": dll_directories,
        "configured_temperatures": list(temperatures),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    receipt_path = output.with_name(f"{output.stem}-receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
