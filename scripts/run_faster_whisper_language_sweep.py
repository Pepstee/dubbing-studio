from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

try:
    from run_faster_whisper_gpu import configure_nvidia_dlls
except ModuleNotFoundError:  # Imported as scripts.run_faster_whisper_language_sweep in tests.
    from scripts.run_faster_whisper_gpu import configure_nvidia_dlls


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preceding_context_prompt(
    transcript: dict, target_start_ms: int, *, context_seconds: int, max_chars: int
) -> str | None:
    lower_bound = target_start_ms - context_seconds * 1000
    texts = [
        str(segment.get("text", "")).strip()
        for segment in transcript.get("segments", [])
        if int(segment.get("end_ms", 0)) <= target_start_ms
        and int(segment.get("end_ms", 0)) > lower_bound
        and str(segment.get("text", "")).strip()
    ]
    if not texts:
        return None
    prompt = " ".join(texts)
    if len(prompt) > max_chars:
        prompt = prompt[-max_chars:]
        first_space = prompt.find(" ")
        if first_space >= 0:
            prompt = prompt[first_space + 1 :]
    return prompt or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent-model multilingual clip sweep")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beam-sizes", default="1,5")
    parser.add_argument("--forced-languages", default="auto,en,ru,ro,ko")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--patience", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--multilingual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--word-timestamps", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--temperatures", default="0")
    parser.add_argument("--initial-prompt")
    parser.add_argument("--context-transcript")
    parser.add_argument("--context-seconds", type=int, default=30)
    parser.add_argument("--context-max-chars", type=int, default=240)
    args = parser.parse_args()

    fixture_path = Path(args.fixture).resolve()
    clips_dir = Path(args.clips_dir).resolve()
    model_path = Path(args.model).resolve()
    output = Path(args.output).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    context_path = Path(args.context_transcript).resolve() if args.context_transcript else None
    context_transcript = (
        json.loads(context_path.read_text(encoding="utf-8")) if context_path else None
    )
    if (
        context_transcript
        and context_transcript.get("source_sha256") != fixture["source"]["sha256"]
    ):
        raise SystemExit("context transcript source SHA-256 mismatch")
    if args.context_seconds < 1 or args.context_max_chars < 1:
        raise SystemExit("context bounds must be positive")
    if args.patience <= 0 or args.length_penalty <= 0:
        raise SystemExit("beam-search controls must be positive")
    temperatures = tuple(float(value) for value in args.temperatures.split(","))
    if not temperatures or any(value < 0 or value > 1 for value in temperatures):
        raise SystemExit("temperatures must be between 0 and 1")
    decode_temperatures: float | tuple[float, ...] = (
        temperatures[0] if len(temperatures) == 1 else temperatures
    )
    beam_sizes = tuple(int(value) for value in args.beam_sizes.split(","))
    if not beam_sizes or any(value < 1 for value in beam_sizes):
        raise SystemExit("beam sizes must be positive integers")
    requested_languages = tuple(value.strip() for value in args.forced_languages.split(","))
    allowed_languages = {"auto", "en", "ru", "ro", "ko"}
    if not requested_languages or any(
        value not in allowed_languages for value in requested_languages
    ):
        raise SystemExit("forced languages must be selected from auto,en,ru,ro,ko")

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
    languages = tuple(None if value == "auto" else value for value in requested_languages)
    for span in fixture["spans"]:
        clip = clips_dir / span.get("clip_relative_path", Path(span["clip_path"]).name)
        if not clip.is_file():
            raise SystemExit(f"missing clip: {clip}")
        if sha256(clip) != span["clip_sha256"]:
            raise SystemExit(f"clip SHA-256 mismatch: {clip.name}")
        candidates = []
        span_prompt = args.initial_prompt
        if context_transcript is not None:
            span_prompt = preceding_context_prompt(
                context_transcript,
                span["segment_start_ms"],
                context_seconds=args.context_seconds,
                max_chars=args.context_max_chars,
            )
        for beam_size in beam_sizes:
            for language in languages:
                decode_started = time.monotonic()
                iterator, info = model.transcribe(
                    str(clip),
                    language=language,
                    multilingual=args.multilingual,
                    beam_size=beam_size,
                    patience=args.patience,
                    length_penalty=args.length_penalty,
                    temperature=decode_temperatures,
                    word_timestamps=args.word_timestamps,
                    vad_filter=False,
                    condition_on_previous_text=args.condition_on_previous_text,
                    initial_prompt=span_prompt,
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
                        "decode_temperatures": sorted(
                            {float(segment.temperature) for segment in segments}
                        ),
                        "segments": [
                            {
                                "start_ms": round(segment.start * 1000),
                                "end_ms": round(segment.end * 1000),
                                "text": segment.text.strip(),
                                "avg_log_probability": segment.avg_logprob,
                                "compression_ratio": segment.compression_ratio,
                                "no_speech_probability": segment.no_speech_prob,
                                "temperature": segment.temperature,
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
                "initial_prompt": span_prompt,
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
        "context_transcript_sha256": sha256(context_path) if context_path else None,
        "context_seconds": args.context_seconds if context_path else None,
        "context_max_chars": args.context_max_chars if context_path else None,
        "beam_sizes": list(beam_sizes),
        "patience": args.patience,
        "length_penalty": args.length_penalty,
        "multilingual": args.multilingual,
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": args.condition_on_previous_text,
        "temperatures": list(temperatures),
        "forced_languages": ["auto" if value is None else value for value in languages],
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
