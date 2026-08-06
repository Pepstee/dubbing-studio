from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import wave
from array import array
from pathlib import Path


LANGUAGE_ORDER = ("en", "ru", "ro", "ko")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required to normalize fixture audio")
    return executable


def _ffmpeg_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0]


def _read_pcm(path: Path) -> tuple[int, array]:
    result = subprocess.run(
        [
            _ffmpeg_executable(),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    samples = array("h")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    return 16000, samples


def _mix_noise(samples: array, *, snr_db: float, seed: int) -> array:
    if not samples:
        return array("h")
    speech_rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    if speech_rms == 0:
        return array("h", samples)
    noise_rms = speech_rms / (10 ** (snr_db / 20))
    uniform_peak = noise_rms * math.sqrt(3)
    generator = random.Random(seed)
    return array(
        "h",
        (
            max(-32768, min(32767, round(value + generator.uniform(-uniform_peak, uniform_peak))))
            for value in samples
        ),
    )


def _normalize_rms(samples: array, *, target_dbfs: float) -> array:
    if not samples:
        return array("h")
    current_rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    peak = max((abs(value) for value in samples), default=0)
    if current_rms == 0 or peak == 0:
        return array("h", samples)
    target_rms = 32767 * 10 ** (target_dbfs / 20)
    scale = min(target_rms / current_rms, (32767 * 0.98) / peak)
    return array("h", (round(value * scale) for value in samples))


def build_noisy_codeswitch_fixture(
    base_manifest_path: str | Path,
    clips_dir: str | Path,
    output_audio_path: str | Path,
    output_manifest_path: str | Path,
    *,
    clips_per_language: int = 4,
    snr_db: float = 10.0,
    gap_ms: int = 750,
    edge_silence_ms: int = 1000,
    target_rms_dbfs: float = -20.0,
) -> dict:
    if clips_per_language < 1:
        raise ValueError("clips_per_language must be positive")
    if snr_db <= 0:
        raise ValueError("snr_db must be positive")
    if gap_ms < 0 or edge_silence_ms < 0:
        raise ValueError("silence durations cannot be negative")
    if not -40 <= target_rms_dbfs <= -10:
        raise ValueError("target_rms_dbfs must be between -40 and -10")

    base_manifest_file = Path(base_manifest_path).resolve()
    clip_root = Path(clips_dir).resolve()
    audio_file = Path(output_audio_path).resolve()
    manifest_file = Path(output_manifest_path).resolve()
    base = json.loads(base_manifest_file.read_text(encoding="utf-8"))
    buckets = {
        language: [span for span in base["spans"] if span["language"] == language]
        for language in LANGUAGE_ORDER
    }
    missing = [
        language
        for language, spans in buckets.items()
        if len(spans) < clips_per_language
    ]
    if missing:
        raise ValueError(f"insufficient clips for languages: {','.join(missing)}")

    selected = [
        buckets[language][index]
        for index in range(clips_per_language)
        for language in LANGUAGE_ORDER
    ]
    sample_rate: int | None = None
    output_samples = array("h")
    spans = []
    gap_samples: int | None = None
    edge_samples: int | None = None
    for sequence, source_span in enumerate(selected):
        clip = clip_root / source_span["clip_relative_path"]
        if not clip.is_file() or _sha256(clip) != source_span["clip_sha256"]:
            raise ValueError(f"clip hash mismatch: {clip}")
        clip_rate, samples = _read_pcm(clip)
        if sample_rate is None:
            sample_rate = clip_rate
            gap_samples = round(sample_rate * gap_ms / 1000)
            edge_samples = round(sample_rate * edge_silence_ms / 1000)
            output_samples.extend([0] * edge_samples)
        elif clip_rate != sample_rate:
            raise ValueError("fixture clips must share one sample rate")
        seed = int(source_span["clip_sha256"][:16], 16)
        normalized = _normalize_rms(samples, target_dbfs=target_rms_dbfs)
        mixed = _mix_noise(normalized, snr_db=snr_db, seed=seed)
        start_sample = len(output_samples)
        output_samples.extend(mixed)
        end_sample = len(output_samples)
        spans.append(
            {
                "id": f"codeswitch-{sequence:03d}-{source_span['id']}",
                "start_ms": round(start_sample * 1000 / sample_rate),
                "end_ms": round(end_sample * 1000 / sample_rate),
                "segment_start_ms": round(start_sample * 1000 / sample_rate),
                "segment_end_ms": round(end_sample * 1000 / sample_rate),
                "text": source_span["text"],
                "raw_text": source_span.get("raw_text"),
                "language": source_span["language"],
                "speaker": "UNKNOWN",
                "no_speech": False,
                "reference_text_human_verified": True,
                "timestamp_basis": "deterministic complete-clip sample concatenation",
                "source_clip_sha256": source_span["clip_sha256"],
                "source_span_id": source_span["id"],
                "sequence": sequence,
            }
        )
        output_samples.extend([0] * (gap_samples or 0))
    output_samples.extend([0] * (edge_samples or 0))
    if sys.byteorder != "little":
        output_samples.byteswap()
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_audio = audio_file.with_name(f".{audio_file.name}.{os.getpid()}.tmp")
    with wave.open(str(temporary_audio), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate or 16000)
        destination.writeframes(output_samples.tobytes())
    os.replace(temporary_audio, audio_file)

    source_sha256 = _sha256(audio_file)
    fixture = {
        "schema_version": "dubbing.derived-noisy-codeswitch-ground-truth.v1",
        "fixture_id": f"fleurs-validation-noisy-codeswitch-{clips_per_language}x4",
        "source": {
            "path": str(audio_file),
            "sha256": source_sha256,
            "duration_ms": round(len(output_samples) * 1000 / (sample_rate or 16000)),
            "sample_rate_hz": sample_rate,
            "channels": 1,
        },
        "base_fixture": {
            "path": str(base_manifest_file),
            "sha256": _sha256(base_manifest_file),
            "source_sha256": base["source"]["sha256"],
        },
        "derivation": {
            "builder": "scripts/build_noisy_codeswitch_fixture.py",
            "policy_version": "dubbing.noisy-codeswitch-derivation.v1",
            "language_order": list(LANGUAGE_ORDER),
            "clips_per_language": clips_per_language,
            "noise": "deterministic uniform additive noise per source clip",
            "snr_db": snr_db,
            "speech_target_rms_dbfs": target_rms_dbfs,
            "gap_ms": gap_ms,
            "edge_silence_ms": edge_silence_ms,
            "python_version": sys.version.split()[0],
            "ffmpeg_version": _ffmpeg_version(_ffmpeg_executable()),
        },
        "selection_policy": {
            "reference_text_is_human_ground_truth": True,
            "reference_timestamps_are_exact_derived_boundaries": True,
            "accuracy_certification_eligible": False,
            "scope_limitation": (
                "Deterministic noisy utterance-level code switching is development evidence, "
                "not natural conversation, overlap, diarization, or production certification."
            ),
        },
        "language_counts": {
            language: sum(span["language"] == language for span in spans)
            for language in LANGUAGE_ORDER
        },
        "coverage_gaps": [],
        "included_count": len(spans),
        "spans": spans,
        "giga_admission_emitted": False,
    }
    _write_atomic(
        manifest_file,
        (json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Build derived noisy code-switch fixture")
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--output-audio", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--clips-per-language", type=int, default=4)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--gap-ms", type=int, default=750)
    parser.add_argument("--target-rms-dbfs", type=float, default=-20.0)
    args = parser.parse_args()
    result = build_noisy_codeswitch_fixture(
        args.base_manifest,
        args.clips_dir,
        args.output_audio,
        args.output_manifest,
        clips_per_language=args.clips_per_language,
        snr_db=args.snr_db,
        gap_ms=args.gap_ms,
        target_rms_dbfs=args.target_rms_dbfs,
    )
    print(json.dumps({"source": result["source"], "spans": len(result["spans"])}, indent=2))


if __name__ == "__main__":
    main()
