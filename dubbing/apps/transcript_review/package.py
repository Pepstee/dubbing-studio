from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from dubbing.media import ffmpeg_executable
from dubbing.transcription.job import source_sha256
from dubbing.transcription.language import script_evidence
from dubbing.transcription.models import transcription_result_from_dict

_MANIFEST_SCHEMA = "dubbing.uncertain-review-package.v1"
_DECISIONS_SCHEMA = "dubbing.uncertain-review-decisions.v1"


def _json_sha256(document: dict) -> str:
    encoded = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_clip(source: Path, destination: Path, start_ms: int, end_ms: int) -> None:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build a transcript review package")
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(source),
            "-t",
            f"{(end_ms - start_ms) / 1000:.3f}",
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=max(120, (end_ms - start_ms) // 1000 + 30),
    )
    if process.returncode:
        raise RuntimeError(f"review clip extraction failed: {process.stderr.strip()}")


def build_review_package(
    source_path: str | Path,
    transcript_path: str | Path,
    output_dir: str | Path,
    *,
    context_ms: int = 1500,
) -> dict:
    if not 0 <= context_ms <= 10_000:
        raise ValueError("context_ms must be between 0 and 10000")
    source = Path(source_path).resolve()
    transcript_path = Path(transcript_path).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file() or not transcript_path.is_file():
        raise FileNotFoundError("source and transcript must both exist")

    transcript_document = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = transcription_result_from_dict(transcript_document)
    source_hash = source_sha256(source)
    transcript_hash = source_sha256(transcript_path)
    if transcript.source_sha256 and transcript.source_sha256 != source_hash:
        raise ValueError("transcript source hash does not match the recording")

    duration_ms = transcript.duration_ms or max(
        (segment.end_ms for segment in transcript.segments), default=0
    )
    items = []
    for segment_index, segment in enumerate(transcript.segments):
        if not segment.uncertain:
            continue
        segment_document = transcript_document["segments"][segment_index]
        segment_hash = _json_sha256(segment_document)
        clip_start_ms = max(0, segment.start_ms - context_ms)
        clip_end_ms = min(duration_ms, segment.end_ms + context_ms)
        item_id = f"u-{segment_index:06d}-{segment_hash[:10]}"
        items.append(
            {
                "id": item_id,
                "segment_index": segment_index,
                "segment_sha256": segment_hash,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "clip_start_ms": clip_start_ms,
                "clip_end_ms": clip_end_ms,
                "clip_path": f"clips/{item_id}.wav",
                "proposed_text": segment.text,
                "proposed_language": segment.language or "unknown",
                "speaker": segment.speaker,
                "script_evidence": script_evidence(segment.text),
            }
        )
    if not items:
        raise ValueError("transcript has no uncertain segments to review")

    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "source": {
            "path": str(source),
            "sha256": source_hash,
            "name": source.name,
        },
        "transcript": {
            "path": str(transcript_path),
            "sha256": transcript_hash,
            "source_sha256": transcript.source_sha256,
        },
        "context_ms": context_ms,
        "item_count": len(items),
        "uncertain_audio_duration_ms": sum(
            item["end_ms"] - item["start_ms"] for item in items
        ),
        "items": items,
        "giga_admission_authorized": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing review package does not match source or transcript")
    else:
        _atomic_json(manifest_path, manifest)

    for item in items:
        clip = output / item["clip_path"]
        if not clip.is_file() or clip.stat().st_size == 0:
            _extract_clip(
                source,
                clip,
                item["clip_start_ms"],
                item["clip_end_ms"],
            )

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
                        "text": item["proposed_text"],
                        "language": item["proposed_language"],
                        "notes": "",
                        "updated_at": None,
                    }
                    for item in items
                },
            },
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a private uncertain-span transcript review package"
    )
    parser.add_argument("source")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-ms", type=int, default=1500)
    args = parser.parse_args()
    manifest = build_review_package(
        args.source,
        args.transcript,
        args.output,
        context_ms=args.context_ms,
    )
    print(
        json.dumps(
            {
                "package": str(Path(args.output).resolve()),
                "items": manifest["item_count"],
                "uncertain_audio_duration_ms": manifest["uncertain_audio_duration_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
