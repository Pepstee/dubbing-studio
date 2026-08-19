#!/usr/bin/env python3
"""Build a source-bound repeated-speaker fixture and verify global diarization."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

from dubbing.diarization import ResumableDiarizationJob, SherpaOnnxDiarizationBackend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_fixture(
    source: Path,
    destination: Path,
    *,
    clip_start_ms: int,
    clip_end_ms: int,
    placements_ms: tuple[int, ...],
    duration_ms: int,
) -> tuple[tuple[int, int], ...]:
    with wave.open(str(source), "rb") as stream:
        rate = stream.getframerate()
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        frames = stream.readframes(stream.getnframes())
    if (rate, channels, width) != (16_000, 1, 2):
        raise ValueError("source must be 16 kHz mono 16-bit PCM WAV")
    if not 0 <= clip_start_ms < clip_end_ms:
        raise ValueError("clip interval is invalid")
    clip_start = clip_start_ms * rate // 1000 * width
    clip_end = clip_end_ms * rate // 1000 * width
    clip = frames[clip_start:clip_end]
    if not clip:
        raise ValueError("clip interval contains no samples")
    clip_ms = clip_end_ms - clip_start_ms
    references = tuple((start, start + clip_ms) for start in placements_ms)
    if not placements_ms or any(start < 0 or end > duration_ms for start, end in references):
        raise ValueError("every placement must fit within the fixture duration")
    output = bytearray(duration_ms * rate // 1000 * width)
    for start_ms in placements_ms:
        offset = start_ms * rate // 1000 * width
        output[offset : offset + len(clip)] = clip
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(output)
    return references


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _score(reference: tuple[tuple[int, int], ...], result) -> dict:
    labels = result.speakers
    overlap_by_speaker = {
        label: sum(
            _overlap(interval, (turn.start_ms, turn.end_ms))
            for interval in reference
            for turn in result.turns
            if turn.speaker == label
        )
        for label in labels
    }
    mapped = max(overlap_by_speaker, key=overlap_by_speaker.get) if labels else None
    reference_ms: set[int] = set()
    hypothesis_ms: set[int] = set()
    mapped_ms: set[int] = set()
    for start_ms, end_ms in reference:
        reference_ms.update(range(start_ms, end_ms))
    for turn in result.turns:
        span = set(range(turn.start_ms, turn.end_ms))
        hypothesis_ms.update(span)
        if turn.speaker == mapped:
            mapped_ms.update(span)
    miss_ms = len(reference_ms - hypothesis_ms)
    false_alarm_ms = len(hypothesis_ms - reference_ms)
    confusion_ms = len((reference_ms & hypothesis_ms) - mapped_ms)
    denominator = len(reference_ms)
    union = len(reference_ms | mapped_ms)
    return {
        "der": round((miss_ms + false_alarm_ms + confusion_ms) / denominator, 6),
        "jer": round(1 - len(reference_ms & mapped_ms) / union, 6),
        "miss_ms": miss_ms,
        "false_alarm_ms": false_alarm_ms,
        "confusion_ms": confusion_ms,
        "reference_speech_ms": denominator,
        "global_speaker_count": len(labels),
        "mapped_speaker": mapped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--segmentation-model", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-start-ms", type=int, required=True)
    parser.add_argument("--clip-end-ms", type=int, required=True)
    parser.add_argument("--placements-ms", default="5000,25000,65000,85000,125000,145000")
    parser.add_argument("--duration-ms", type=int, default=180_000)
    parser.add_argument("--chunk-seconds", type=int, default=60)
    parser.add_argument("--sherpa-threshold", type=float, default=0.85)
    parser.add_argument("--global-threshold", type=float, default=0.80)
    parser.add_argument("--global-margin", type=float, default=0.05)
    parser.add_argument("--maximum-der", type=float, default=0.25)
    parser.add_argument("--maximum-jer", type=float, default=0.25)
    args = parser.parse_args()

    placements = tuple(int(value) for value in args.placements_ms.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = args.output_dir / "repeated-real-speaker.wav"
    reference = _build_fixture(
        args.source,
        fixture,
        clip_start_ms=args.clip_start_ms,
        clip_end_ms=args.clip_end_ms,
        placements_ms=placements,
        duration_ms=args.duration_ms,
    )
    backend = SherpaOnnxDiarizationBackend(
        args.segmentation_model,
        args.embedding_model,
        cluster_threshold=args.sherpa_threshold,
        num_threads=4,
    )
    job = ResumableDiarizationJob(
        backend,
        args.output_dir / "checkpoints",
        speaker_embedding_backend=backend,
        chunk_seconds=args.chunk_seconds,
        global_speaker_threshold=args.global_threshold,
        global_speaker_margin=args.global_margin,
    )
    started = time.monotonic()
    first = job.run(fixture, duration_ms=args.duration_ms)
    first_seconds = time.monotonic() - started
    started = time.monotonic()
    replay = job.run(fixture, duration_ms=args.duration_ms)
    replay_seconds = time.monotonic() - started

    metrics = _score(reference, first)
    receipt = first.provenance["global_speaker_reconciliation"]
    mapped = metrics["mapped_speaker"]
    represented_chunks = {
        decision["chunk_speaker"].split("_")[1]
        for decision in receipt["decisions"]
        if decision["global_speaker"] == mapped
    }
    metrics.update(
        {
            "represented_chunk_count": len(represented_chunks),
            "checkpoint_replay_equal": first.to_dict() == replay.to_dict(),
        }
    )
    passed = (
        metrics["global_speaker_count"] == 1
        and metrics["represented_chunk_count"]
        == (args.duration_ms + args.chunk_seconds * 1000 - 1)
        // (args.chunk_seconds * 1000)
        and metrics["der"] <= args.maximum_der
        and metrics["jer"] <= args.maximum_jer
        and metrics["checkpoint_replay_equal"]
    )
    report = {
        "schema_version": "dubbing.project5-repeated-speaker-verification.v2",
        "verdict": "PASS" if passed else "FAIL",
        "fixture": {
            "source": str(args.source.resolve()),
            "source_sha256": _sha256(args.source),
            "derived": str(fixture.resolve()),
            "derived_sha256": _sha256(fixture),
            "source_interval_ms": [args.clip_start_ms, args.clip_end_ms],
            "placements_ms": reference,
            "duration_ms": args.duration_ms,
            "reference_speaker_count": 1,
        },
        "configuration": {
            "sherpa_cluster_threshold": args.sherpa_threshold,
            "global_complete_link_threshold": args.global_threshold,
            "incompatible_candidate_margin": args.global_margin,
            "chunk_seconds": args.chunk_seconds,
        },
        "runtime_seconds": {
            "first": round(first_seconds, 3),
            "checkpoint_replay": round(replay_seconds, 3),
        },
        "metrics": metrics,
        "result": first.to_dict(),
        "metric_note": (
            "Single-speaker, zero-collar millisecond scoring on exact derived active "
            "intervals; natural multi-speaker DER/JER still requires human labels."
        ),
    }
    report_path = args.output_dir / "verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
