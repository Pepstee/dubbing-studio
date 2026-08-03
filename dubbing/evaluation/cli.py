from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dubbing.evaluation.metrics import TimedText, evaluate_documents
from dubbing.transcription.models import transcription_result_from_dict
from dubbing.transcription.quality import evaluate_transcript_quality


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference(path: Path) -> tuple[str, tuple[TimedText, ...]]:
    if path.suffix.lower() != ".json":
        return path.read_text(encoding="utf-8"), ()
    document = json.loads(path.read_text(encoding="utf-8"))
    segments = tuple(
        TimedText(
            start_ms=item["start_ms"],
            end_ms=item["end_ms"],
            text=item["text"],
            language=item.get("language"),
            speaker=item.get("speaker"),
        )
        for item in document.get("segments", [])
    )
    return document.get("text", " ".join(item.text for item in segments)), segments


def evaluate_fixture(manifest_path: str | Path, output_path: str | Path | None = None) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent

    def resolve(item: dict) -> Path:
        path = Path(item["path"])
        return path if path.is_absolute() else (root / path).resolve()

    for key in ("source", "reference", "candidate"):
        path = resolve(manifest[key])
        actual = _sha256(path)
        if actual != manifest[key]["sha256"]:
            raise ValueError(f"{key} SHA-256 mismatch: expected {manifest[key]['sha256']}")

    reference_text, reference_segments = _reference(resolve(manifest["reference"]))
    candidate_document = json.loads(resolve(manifest["candidate"]).read_text(encoding="utf-8"))
    candidate = transcription_result_from_dict(candidate_document)
    candidate_segments = tuple(
        TimedText(
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            text=item.text,
            language=item.language,
            speaker=item.speaker,
        )
        for item in candidate.segments
    )
    metrics = evaluate_documents(
        reference_text,
        candidate.text,
        reference_segments=reference_segments,
        candidate_segments=candidate_segments,
        duration_ms=manifest["source"].get("duration_ms"),
        window_ms=manifest.get("evaluation", {}).get("window_ms", 300_000),
    )
    quality = evaluate_transcript_quality(
        candidate, expected_duration_ms=manifest["source"].get("duration_ms")
    )
    report = {
        "schema_version": "dubbing.regression-fixture-report.v1",
        "fixture_id": manifest["fixture_id"],
        "source_sha256": manifest["source"]["sha256"],
        "reference_kind": manifest["reference"]["kind"],
        "reference_is_human_ground_truth": manifest["reference"].get(
            "human_ground_truth", False
        ),
        "backend": manifest["backend"],
        "metrics": metrics,
        "quality": quality.to_dict(),
    }
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a hash-bound transcript fixture")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = evaluate_fixture(args.manifest, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["quality"]["status"] in {"FAILED", "REPROCESS_REQUIRED"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
