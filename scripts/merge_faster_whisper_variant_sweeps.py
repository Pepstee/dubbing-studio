from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


COMMON_FIELDS = (
    "fixture_sha256",
    "source_sha256",
    "model",
    "model_bin_sha256",
    "compute_type",
    "cloud_allowed",
    "giga_admission_emitted",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_variant_sweeps(
    variants: dict[str, str | Path],
    output_path: str | Path,
    *,
    language_retry_policy: dict[str, str] | None = None,
) -> dict:
    if len(variants) < 2:
        raise ValueError("at least two uniquely named variants are required")
    retry_policy = language_retry_policy or {}
    if any(
        mode not in {"always", "confidence", "global_confidence"}
        for mode in retry_policy.values()
    ):
        raise ValueError(
            "language retry policy must use always, confidence, or global_confidence"
        )
    loaded: list[tuple[str, Path, dict]] = []
    for variant_id, source in variants.items():
        if not variant_id or any(character.isspace() for character in variant_id):
            raise ValueError("variant identifiers must be non-empty and contain no whitespace")
        path = Path(source).resolve()
        loaded.append((variant_id, path, json.loads(path.read_text(encoding="utf-8"))))

    baseline = loaded[0][2]
    baseline_ids = [row["id"] for row in baseline["spans"]]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("baseline sweep contains duplicate span identifiers")
    for variant_id, _, document in loaded:
        for field in COMMON_FIELDS:
            if document.get(field) != baseline.get(field):
                raise ValueError(f"variant {variant_id} disagrees on {field}")
        if [row["id"] for row in document["spans"]] != baseline_ids:
            raise ValueError(f"variant {variant_id} span order or identifiers differ")

    merged_rows = []
    for index, baseline_row in enumerate(baseline["spans"]):
        candidates = []
        for variant_id, _, document in loaded:
            row = document["spans"][index]
            if row["declared_reference_language"] != baseline_row["declared_reference_language"]:
                raise ValueError(f"variant {variant_id} declared language differs")
            candidates.extend(
                {**candidate, "variant_id": variant_id}
                for candidate in row["candidates"]
            )
        merged_rows.append(
            {
                "id": baseline_row["id"],
                "declared_reference_language": baseline_row["declared_reference_language"],
                "candidates": candidates,
            }
        )

    merged = {
        "schema_version": "dubbing.faster-whisper-variant-sweep.v1",
        **{field: baseline.get(field) for field in COMMON_FIELDS},
        "beam_sizes": sorted(
            {value for _, _, document in loaded for value in document["beam_sizes"]}
        ),
        "forced_languages": sorted(
            {value for _, _, document in loaded for value in document["forced_languages"]}
        ),
        "temperatures": sorted(
            {value for _, _, document in loaded for value in document.get("temperatures", [0.0])}
        ),
        "runtime_seconds": sum(document["runtime_seconds"] for _, _, document in loaded),
        "word_timestamps": all(
            document.get("word_timestamps", True) for _, _, document in loaded
        ),
        "language_retry_policy": dict(sorted(retry_policy.items())),
        "variants": [
            {
                "variant_id": variant_id,
                "sweep_sha256": _sha256(path),
                "runtime_seconds": document["runtime_seconds"],
                "decode_configuration": {
                    key: document.get(key)
                    for key in (
                        "beam_sizes",
                        "forced_languages",
                        "temperatures",
                        "patience",
                        "length_penalty",
                        "multilingual",
                        "word_timestamps",
                        "condition_on_previous_text",
                        "vad_filter",
                        "vad_threshold",
                        "vad_speech_pad_ms",
                        "vad_min_silence_duration_ms",
                    )
                },
                "postprocessing": document.get("postprocessing"),
            }
            for variant_id, path, document in loaded
        ],
        "spans": merged_rows,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge hash-bound ASR variant sweeps")
    parser.add_argument("--variant", action="append", required=True, metavar="ID=PATH")
    parser.add_argument(
        "--retry-policy", action="append", default=[], metavar="LANG=MODE"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    variants = {}
    for value in args.variant:
        if "=" not in value:
            raise SystemExit("variants must use ID=PATH")
        variant_id, path = value.split("=", 1)
        if variant_id in variants:
            raise SystemExit(f"duplicate variant: {variant_id}")
        variants[variant_id] = path
    retry_policy = {}
    for value in args.retry_policy:
        if "=" not in value:
            raise SystemExit("retry policies must use LANG=MODE")
        language, mode = value.split("=", 1)
        if language in retry_policy:
            raise SystemExit(f"duplicate retry policy: {language}")
        retry_policy[language] = mode
    merged = merge_variant_sweeps(
        variants, args.output, language_retry_policy=retry_policy
    )
    print(
        json.dumps(
            {
                "variants": len(merged["variants"]),
                "spans": len(merged["spans"]),
                "runtime_seconds": merged["runtime_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
