from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dubbing.transcription.korean_spacing import (
    kiwi_runtime_receipt,
    kiwi_space,
    normalize_candidate_spacing,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_korean_spacing(source_path: str | Path, output_path: str | Path) -> dict:
    source = Path(source_path).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("postprocessing") is not None:
        raise RuntimeError("source sweep already contains postprocessing provenance")
    runtime_receipt = kiwi_runtime_receipt()
    provider = (
        f"kiwipiepy=={runtime_receipt['runtime']['version']}:"
        "Kiwi.space(reset_whitespace=True)"
    )
    applied = 0
    unchanged = 0
    rejected = 0
    for row in document["spans"]:
        candidates = []
        for candidate in row["candidates"]:
            if candidate.get("forced_language") != "ko":
                candidates.append(candidate)
                continue
            processed = normalize_candidate_spacing(
                candidate, kiwi_space, provider=provider
            )
            status = processed["spacing_normalization"]["status"]
            applied += status == "APPLIED"
            unchanged += status == "UNCHANGED"
            rejected += status.startswith("REJECTED_")
            candidates.append(processed)
        row["candidates"] = candidates
    document["postprocessing"] = {
        "kind": "korean_spacing",
        "provider": provider,
        "runtime_receipt": runtime_receipt,
        "source_sweep_sha256": _sha256(source),
        "character_changes_allowed": False,
        "applied_candidates": applied,
        "unchanged_candidates": unchanged,
        "rejected_candidates": rejected,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return document["postprocessing"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply guarded local Korean spacing")
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            apply_korean_spacing(args.sweep, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
