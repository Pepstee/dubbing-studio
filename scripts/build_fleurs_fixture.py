from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

DATASET = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SPLIT = "validation"
LANGUAGES = {"en_us": "en", "ru_ru": "ru", "ro_ro": "ro", "ko_kr": "ko"}
DATASET_API = "https://huggingface.co/api/datasets/google/fleurs"
ROWS_API = "https://datasets-server.huggingface.co/rows"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "dubbing-studio/1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def build_fleurs_fixture(
    output_dir: str | Path,
    fixture_path: str | Path,
    *,
    samples_per_language: int = 10,
    http_get: Callable[[str], bytes] = _http_get,
) -> dict:
    if samples_per_language < 1 or samples_per_language > 100:
        raise ValueError("samples_per_language must be between 1 and 100")
    metadata_bytes = http_get(DATASET_API)
    metadata = json.loads(metadata_bytes)
    if metadata.get("sha") != REVISION:
        raise RuntimeError(
            f"FLEURS revision changed: expected {REVISION}, got {metadata.get('sha')}"
        )
    licenses = metadata.get("cardData", {}).get("license", [])
    if "cc-by-4.0" not in licenses:
        raise RuntimeError("FLEURS CC-BY-4.0 license metadata is missing")

    clips_dir = Path(output_dir).resolve()
    destination = Path(fixture_path).resolve()
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate FLEURS audio")
    spans = []
    total_audio_bytes = 0
    for config, language in LANGUAGES.items():
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": config,
                "split": SPLIT,
                "offset": 0,
                "length": samples_per_language,
            }
        )
        rows_bytes = http_get(f"{ROWS_API}?{query}")
        rows_document = json.loads(rows_bytes)
        rows = rows_document.get("rows", [])
        if len(rows) != samples_per_language:
            raise RuntimeError(
                f"expected {samples_per_language} {config} rows, received {len(rows)}"
            )
        for wrapped in rows:
            row_index = int(wrapped["row_idx"])
            row = wrapped["row"]
            audio_items = row.get("audio", [])
            if len(audio_items) != 1 or audio_items[0].get("type") != "audio/wav":
                raise RuntimeError(f"unexpected audio representation for {config}:{row_index}")
            audio_url = audio_items[0]["src"]
            if f"/{REVISION}/" not in audio_url:
                raise RuntimeError(f"audio asset is not pinned to {REVISION}")
            audio_bytes = http_get(audio_url)
            clip = clips_dir / config / f"{row_index:06d}-{int(row['id'])}.wav"
            _write_atomic(clip, audio_bytes)
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate,channels,duration_ts,time_base",
                    "-of",
                    "json",
                    str(clip),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if probe.returncode:
                raise RuntimeError(f"ffprobe failed for {config}:{row_index}")
            stream = json.loads(probe.stdout)["streams"][0]
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
            if stream["time_base"] != f"1/{sample_rate}":
                raise RuntimeError(f"unexpected WAV time base for {config}:{row_index}")
            frame_count = int(stream["duration_ts"])
            if sample_rate != 16000 or channels != 1:
                raise RuntimeError(f"unexpected WAV format for {config}:{row_index}")
            if frame_count != int(row["num_samples"]):
                raise RuntimeError(f"sample count mismatch for {config}:{row_index}")
            duration_ms = round(frame_count * 1000 / sample_rate)
            total_audio_bytes += len(audio_bytes)
            spans.append(
                {
                    "id": f"fleurs-{config}-{SPLIT}-{row_index:06d}-{int(row['id'])}",
                    "dataset_config": config,
                    "dataset_row_index": row_index,
                    "dataset_row_id": int(row["id"]),
                    "start_ms": 0,
                    "end_ms": duration_ms,
                    "segment_start_ms": 0,
                    "segment_end_ms": duration_ms,
                    "text": row["transcription"],
                    "raw_text": row["raw_transcription"],
                    "language": language,
                    "no_speech": False,
                    "clip_path": str(clip),
                    "clip_relative_path": str(Path(config) / clip.name),
                    "clip_sha256": _sha256(clip),
                    "sample_rate_hz": sample_rate,
                    "sample_count": frame_count,
                    "reference_text_human_verified": True,
                    "timestamp_human_verified": True,
                }
            )

    source_binding = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": SPLIT,
        "spans": [
            {
                "id": span["id"],
                "clip_sha256": span["clip_sha256"],
                "text": span["text"],
            }
            for span in spans
        ],
    }
    source_sha256 = _sha256_bytes(
        json.dumps(
            source_binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    fixture = {
        "schema_version": "dubbing.fleurs-ground-truth.v1",
        "fixture_id": f"fleurs-{SPLIT}-{samples_per_language}x4-{REVISION[:12]}",
        "source": {
            "kind": "dataset_slice",
            "dataset": DATASET,
            "revision": REVISION,
            "split": SPLIT,
            "sha256": source_sha256,
        },
        "dataset_evidence": {
            "metadata_url": DATASET_API,
            "metadata_sha256": _sha256_bytes(metadata_bytes),
            "repository_url": "https://huggingface.co/datasets/google/fleurs",
            "paper_url": "https://arxiv.org/abs/2205.12446",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": (
                "FLEURS: Few-shot Learning Evaluation of Universal Representations of "
                "Speech (Conneau et al., 2022)"
            ),
        },
        "selection_policy": {
            "policy_version": "dubbing.fleurs-validation-prefix.v1",
            "strategy": "first N validation rows from each pinned language configuration",
            "samples_per_language": samples_per_language,
            "reference_text_is_human_ground_truth": True,
            "reference_timestamps_are_human_ground_truth": True,
            "timing_basis": "complete dataset utterance WAV",
            "accuracy_certification_eligible": True,
            "scope_limitation": (
                "Clean read speech validates language coverage; it does not certify noisy "
                "long-form or code-switched production audio."
            ),
        },
        "language_counts": {language: samples_per_language for language in LANGUAGES.values()},
        "coverage_gaps": [],
        "included_count": len(spans),
        "excluded_count": 0,
        "total_audio_bytes": total_audio_bytes,
        "spans": spans,
        "giga_admission_emitted": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        destination,
        (json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pinned four-language FLEURS fixture")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--samples-per-language", type=int, default=10)
    args = parser.parse_args()
    fixture = build_fleurs_fixture(
        args.output_dir,
        args.fixture,
        samples_per_language=args.samples_per_language,
    )
    print(
        json.dumps(
            {
                "spans": fixture["included_count"],
                "languages": fixture["language_counts"],
                "audio_bytes": fixture["total_audio_bytes"],
                "source_sha256": fixture["source"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
