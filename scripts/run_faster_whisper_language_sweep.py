from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from run_faster_whisper_gpu import configure_nvidia_dlls


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent-model multilingual clip sweep")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beam-sizes", default="1,5")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--initial-prompt")
    args = parser.parse_args()

    fixture_path = Path(args.fixture).resolve()
    clips_dir = Path(args.clips_dir).resolve()
    model_path = Path(args.model).resolve()
    output = Path(args.output).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    beam_sizes = tuple(int(value) for value in args.beam_sizes.split(","))
    if not beam_sizes or any(value < 1 for value in beam_sizes):
        raise SystemExit("beam sizes must be positive integers")

    dll_directories = configure_nvidia_dlls()
    from faster_whisper import WhisperModel

    started = time.monotonic()
    model_started = time.monotonic()
    model = WhisperModel(
        str(model_path),
        device="cuda",
        compute_type=args.compute_type,
        local_files_only=True,
    )
    model_load_seconds = time.monotonic() - model_started
    rows = []
    languages = (None, "en", "ru", "ro", "ko")
    for span in fixture["spans"]:
        clip = clips_dir / Path(span["clip_path"]).name
        if not clip.is_file():
            raise SystemExit(f"missing clip: {clip}")
        if sha256(clip) != span["clip_sha256"]:
            raise SystemExit(f"clip SHA-256 mismatch: {clip.name}")
        candidates = []
        for beam_size in beam_sizes:
            for language in languages:
                decode_started = time.monotonic()
                iterator, info = model.transcribe(
                    str(clip),
                    language=language,
                    multilingual=True,
                    beam_size=beam_size,
                    temperature=0.0,
                    word_timestamps=True,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    initial_prompt=args.initial_prompt,
                )
                segments = list(iterator)
                text = " ".join(
                    segment.text.strip() for segment in segments if segment.text.strip()
                )
                logprobs = [segment.avg_logprob for segment in segments]
                candidates.append(
                    {
                        "forced_language": language,
                        "detected_language": info.language,
                        "detected_language_probability": info.language_probability,
                        "beam_size": beam_size,
                        "text": text,
                        "avg_log_probability": (
                            sum(logprobs) / len(logprobs) if logprobs else None
                        ),
                        "decode_seconds": round(time.monotonic() - decode_started, 6),
                        "segments": [
                            {
                                "start_ms": round(segment.start * 1000),
                                "end_ms": round(segment.end * 1000),
                                "text": segment.text.strip(),
                                "avg_log_probability": segment.avg_logprob,
                                "compression_ratio": segment.compression_ratio,
                                "no_speech_probability": segment.no_speech_prob,
                                "words": [
                                    {
                                        "start_ms": round(word.start * 1000),
                                        "end_ms": round(word.end * 1000),
                                        "text": word.word,
                                        "probability": word.probability,
                                    }
                                    for word in (segment.words or [])
                                ],
                            }
                            for segment in segments
                            if segment.text.strip()
                        ],
                    }
                )
        rows.append(
            {
                "id": span["id"],
                "clip_sha256": span["clip_sha256"],
                "declared_reference_language": span["language"],
                "clip_start_ms": span["start_ms"],
                "clip_end_ms": span["end_ms"],
                "target_start_ms": span["segment_start_ms"],
                "target_end_ms": span["segment_end_ms"],
                "candidates": candidates,
            }
        )
    document = {
        "schema_version": "dubbing.faster-whisper-language-sweep.v1",
        "source_sha256": fixture["source"]["sha256"],
        "fixture_sha256": sha256(fixture_path),
        "model": model_path.name,
        "model_bin_sha256": sha256(model_path / "model.bin"),
        "compute_type": args.compute_type,
        "initial_prompt": args.initial_prompt,
        "beam_sizes": list(beam_sizes),
        "forced_languages": ["auto", "en", "ru", "ro", "ko"],
        "model_load_seconds": round(model_load_seconds, 6),
        "runtime_seconds": round(time.monotonic() - started, 6),
        "nvidia_dll_directories": dll_directories,
        "cloud_allowed": False,
        "giga_admission_emitted": False,
        "spans": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps({"spans": len(rows), "runtime_seconds": document["runtime_seconds"]}, indent=2)
    )


if __name__ == "__main__":
    main()
