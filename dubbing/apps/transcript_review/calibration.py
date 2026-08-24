from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dubbing.apps.transcript_review.package import _atomic_json, _extract_clip
from dubbing.transcription.job import source_sha256
from dubbing.transcription.models import transcription_result_from_dict

_MANIFEST_SCHEMA = "dubbing.ground-truth-calibration-package.v1"
_DECISIONS_SCHEMA = "dubbing.ground-truth-calibration-decisions.v1"
_CLIP_DURATION_MS = 50_000


def _json_sha256(document: dict) -> str:
    encoded = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_metrics(segments, start_ms: int, end_ms: int) -> dict:
    rows = []
    language_ms: dict[str, int] = {}
    language_segments: dict[str, int] = {}
    overlap_count = 0
    previous = None
    for index, segment in enumerate(segments):
        midpoint = segment.start_ms + (segment.end_ms - segment.start_ms) // 2
        if not start_ms <= midpoint < end_ms:
            continue
        rows.append((index, segment))
        language = segment.language or "unknown"
        duration = max(0, segment.end_ms - segment.start_ms)
        language_ms[language] = language_ms.get(language, 0) + duration
        language_segments[language] = language_segments.get(language, 0) + 1
        if previous is not None and segment.start_ms < previous.end_ms:
            overlap_count += 1
        previous = segment
    return {
        "rows": rows,
        "language_ms": language_ms,
        "language_segments": language_segments,
        "overlap_count": overlap_count,
        "speech_ms": sum(language_ms.values()),
    }


def _score(metrics: dict, target: str) -> float:
    milliseconds = metrics["language_ms"]
    counts = metrics["language_segments"]
    language_count = sum(value > 0 for value in milliseconds.values())
    if target == "en":
        return milliseconds.get("en", 0) + 2500 * counts.get("en", 0)
    if target == "ru":
        return (
            milliseconds.get("ru", 0)
            + 0.5 * milliseconds.get("mixed", 0)
            + 2500 * counts.get("ru", 0)
        )
    if target == "ko":
        return 3 * milliseconds.get("ko", 0) + 2500 * counts.get("ko", 0)
    if target == "mixed":
        return (
            4 * milliseconds.get("mixed", 0)
            + 8000 * max(0, language_count - 1)
            + 2000 * counts.get("mixed", 0)
        )
    return (
        5000 * metrics["overlap_count"]
        + 1500 * len(metrics["rows"])
        + 8000 * max(0, language_count - 1)
    )


def _selection_specs(duration_ms: int) -> tuple[tuple[str, str, float, float], ...]:
    return (
        ("korean", "ko", 0.00, 1.00),
        ("mixed-a", "mixed", 0.00, 1.00),
        ("russian-early", "ru", 0.10, 0.35),
        ("english-early", "en", 0.00, 0.20),
        ("english-middle", "en", 0.20, 0.45),
        ("russian-middle", "ru", 0.30, 0.60),
        ("mixed-b", "mixed", 0.00, 1.00),
        ("russian-late", "ru", 0.54, 0.74),
        ("english-late-a", "en", 0.64, 0.81),
        ("english-late-b", "en", 0.78, 0.91),
        ("english-end", "en", 0.88, 1.00),
        ("challenging", "challenging", 0.00, 1.00),
    )


def select_calibration_windows(transcript, clip_duration_ms: int = _CLIP_DURATION_MS) -> list[dict]:
    duration_ms = transcript.duration_ms or max(segment.end_ms for segment in transcript.segments)
    starts = set()
    for segment in transcript.segments:
        midpoint = segment.start_ms + (segment.end_ms - segment.start_ms) // 2
        start = max(0, min(duration_ms - clip_duration_ms, midpoint - clip_duration_ms // 2))
        starts.add(start // 1000 * 1000)
    candidates = []
    for start_ms in sorted(starts):
        metrics = _window_metrics(
            transcript.segments, start_ms, min(duration_ms, start_ms + clip_duration_ms)
        )
        if metrics["speech_ms"] >= 8000:
            candidates.append({"start_ms": start_ms, "metrics": metrics})

    observed_languages = {segment.language for segment in transcript.segments}
    selected = []
    for label, target, lower, upper in _selection_specs(duration_ms):
        if target in {"en", "ru", "ko", "mixed"} and target not in observed_languages:
            continue
        viable = [
            candidate
            for candidate in candidates
            if lower * duration_ms <= candidate["start_ms"] < upper * duration_ms
            and all(
                abs(candidate["start_ms"] - existing["start_ms"]) >= clip_duration_ms
                for existing in selected
            )
        ]
        if not viable:
            continue
        winner = max(
            viable,
            key=lambda candidate: (
                _score(candidate["metrics"], target),
                -candidate["start_ms"],
            ),
        )
        selected.append({**winner, "label": label, "target": target})
    selected_targets = {item["target"] for item in selected}
    for target in ("en", "ru", "ko", "mixed"):
        if target not in observed_languages or target in selected_targets:
            continue
        viable = [
            candidate
            for candidate in candidates
            if all(
                abs(candidate["start_ms"] - existing["start_ms"]) >= clip_duration_ms
                for existing in selected
            )
        ]
        if not viable:
            continue
        winner = max(viable, key=lambda candidate: _score(candidate["metrics"], target))
        selected.append({**winner, "label": f"{target}-coverage", "target": target})
        selected_targets.add(target)
    return sorted(selected, key=lambda item: item["start_ms"])


def build_calibration_package(
    source_path: str | Path,
    transcript_path: str | Path,
    output_dir: str | Path,
    *,
    clip_duration_ms: int = _CLIP_DURATION_MS,
) -> dict:
    if not 30_000 <= clip_duration_ms <= 90_000:
        raise ValueError("clip_duration_ms must be between 30000 and 90000")
    source = Path(source_path).resolve()
    transcript_path = Path(transcript_path).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file() or not transcript_path.is_file():
        raise FileNotFoundError("source and reviewed transcript must both exist")
    transcript_document = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = transcription_result_from_dict(transcript_document)
    source_hash = source_sha256(source)
    if transcript.source_sha256 != source_hash:
        raise ValueError("reviewed transcript source hash does not match recording")

    windows = select_calibration_windows(transcript, clip_duration_ms)
    if not windows:
        raise ValueError("no calibration windows contained enough timestamped speech")
    items = []
    for position, window in enumerate(windows, start=1):
        start_ms = window["start_ms"]
        end_ms = min(transcript.duration_ms or start_ms + clip_duration_ms, start_ms + clip_duration_ms)
        window_document = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "target": window["target"],
            "source_sha256": source_hash,
        }
        item_id = f"c-{position:02d}-{_json_sha256(window_document)[:10]}"
        turns = []
        for segment_index, segment in window["metrics"]["rows"]:
            turn_start = max(start_ms, segment.start_ms)
            turn_end = min(end_ms, segment.end_ms)
            if turn_end <= turn_start:
                continue
            turns.append(
                {
                    "id": f"{item_id}-t{len(turns) + 1:03d}",
                    "candidate_segment_index": segment_index,
                    "start_ms": turn_start - start_ms,
                    "end_ms": turn_end - start_ms,
                    "speaker": "UNKNOWN",
                    "language": segment.language or "unknown",
                    "text": segment.text,
                }
            )
        items.append(
            {
                "id": item_id,
                "position": position,
                "label": window["label"],
                "target": window["target"],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "clip_path": f"clips/{item_id}.wav",
                "observed_language_ms": window["metrics"]["language_ms"],
                "overlap_count": window["metrics"]["overlap_count"],
                "candidate_turns": turns,
            }
        )

    observed = sorted({segment.language or "unknown" for segment in transcript.segments})
    required = ["en", "ru", "ro", "ko", "mixed"]
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "source": {"path": str(source), "sha256": source_hash, "name": source.name},
        "candidate": {
            "path": str(transcript_path),
            "sha256": source_sha256(transcript_path),
            "human_ground_truth": False,
        },
        "clip_duration_ms": clip_duration_ms,
        "item_count": len(items),
        "total_audio_duration_ms": sum(item["duration_ms"] for item in items),
        "observed_languages": observed,
        "coverage_gaps": [language for language in required if language not in observed],
        "selection_policy": "dubbing.stratified-calibration-v1",
        "items": items,
        "giga_admission_authorized": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing calibration package does not match source or transcript")
    else:
        _atomic_json(manifest_path, manifest)
    for item in items:
        clip_path = output / item["clip_path"]
        if not clip_path.is_file() or clip_path.stat().st_size == 0:
            _extract_clip(source, clip_path, item["start_ms"], item["end_ms"])

    decisions_path = output / "decisions.json"
    if not decisions_path.exists():
        _atomic_json(
            decisions_path,
            {
                "schema_version": _DECISIONS_SCHEMA,
                "manifest_sha256": source_sha256(manifest_path),
                "items": {
                    item["id"]: {
                        "status": "pending",
                        "turns": item["candidate_turns"],
                        "notes": "",
                        "updated_at": None,
                    }
                    for item in items
                },
            },
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private ground-truth calibration package")
    parser.add_argument("source")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-duration-ms", type=int, default=_CLIP_DURATION_MS)
    args = parser.parse_args()
    manifest = build_calibration_package(
        args.source,
        args.transcript,
        args.output,
        clip_duration_ms=args.clip_duration_ms,
    )
    print(
        json.dumps(
            {
                "package": str(Path(args.output).resolve()),
                "clips": manifest["item_count"],
                "audio_duration_ms": manifest["total_audio_duration_ms"],
                "coverage_gaps": manifest["coverage_gaps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
