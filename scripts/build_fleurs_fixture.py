from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

DATASET = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
DEFAULT_SPLIT = "validation"
ALLOWED_SPLITS = {"validation", "test"}
LANGUAGES = {"en_us": "en", "ru_ru": "ru", "ro_ro": "ro", "ko_kr": "ko"}
DATASET_API = "https://huggingface.co/api/datasets/google/fleurs"
ROWS_API = "https://datasets-server.huggingface.co/rows"
REPOSITORY_RESOLVE = "https://huggingface.co/datasets/google/fleurs/resolve"
ALLOWED_SOURCE_MODES = {"rows_api", "tar_stream"}


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


def _http_open(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "dubbing-studio/1"})
    return urllib.request.urlopen(request, timeout=60)  # noqa: S310


def _stream_archive_rows(
    config: str,
    split: str,
    count: int,
    *,
    http_get: Callable[[str], bytes],
    http_open: Callable,
) -> list[dict]:
    tsv_url = f"{REPOSITORY_RESOLVE}/{REVISION}/data/{config}/{split}.tsv"
    archive_url = (
        f"{REPOSITORY_RESOLVE}/{REVISION}/data/{config}/audio/{split}.tar.gz"
    )
    lines = http_get(tsv_url).decode("utf-8").splitlines()
    metadata = {}
    for row_index, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) != 7:
            raise RuntimeError(f"unexpected TSV row for {config}:{row_index}")
        row_id, filename, raw_text, text, _, sample_count, _ = fields
        metadata[filename] = {
            "row_idx": row_index,
            "row": {
                "id": int(row_id),
                "num_samples": int(sample_count),
                "transcription": text,
                "raw_transcription": raw_text,
            },
        }

    selected = []
    with http_open(archive_url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".wav"):
                    continue
                filename = Path(member.name).name
                if filename not in metadata:
                    raise RuntimeError(f"archive member missing from TSV: {config}:{filename}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot extract archive member: {config}:{filename}")
                wrapped = metadata[filename]
                selected.append(
                    {
                        **wrapped,
                        "row": {
                            **wrapped["row"],
                            "audio": [
                                {
                                    "src": f"{archive_url}#{member.name}",
                                    "type": "audio/wav",
                                    "content": extracted.read(),
                                }
                            ],
                        },
                    }
                )
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f"expected {count} {config} archive rows, received {len(selected)}")
    return selected


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
    split: str = DEFAULT_SPLIT,
    source_mode: str = "rows_api",
    http_get: Callable[[str], bytes] = _http_get,
    http_open: Callable = _http_open,
) -> dict:
    if samples_per_language < 1 or samples_per_language > 100:
        raise ValueError("samples_per_language must be between 1 and 100")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
    if source_mode not in ALLOWED_SOURCE_MODES:
        raise ValueError(f"source_mode must be one of {sorted(ALLOWED_SOURCE_MODES)}")
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
        if source_mode == "rows_api":
            query = urllib.parse.urlencode(
                {
                    "dataset": DATASET,
                    "config": config,
                    "split": split,
                    "offset": 0,
                    "length": samples_per_language,
                }
            )
            rows_bytes = http_get(f"{ROWS_API}?{query}")
            rows = json.loads(rows_bytes).get("rows", [])
        else:
            rows = _stream_archive_rows(
                config,
                split,
                samples_per_language,
                http_get=http_get,
                http_open=http_open,
            )
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
            audio_bytes = audio_items[0].get("content")
            if audio_bytes is None:
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
                    "id": f"fleurs-{config}-{split}-{row_index:06d}-{int(row['id'])}",
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
        "split": split,
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
        "fixture_id": f"fleurs-{split}-{samples_per_language}x4-{REVISION[:12]}",
        "source": {
            "kind": "dataset_slice",
            "dataset": DATASET,
            "revision": REVISION,
            "split": split,
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
            "policy_version": "dubbing.fleurs-split-prefix.v1",
            "strategy": (
                f"first N {split} rows from each pinned language configuration"
                if source_mode == "rows_api"
                else f"first N WAV members from each pinned {split} archive"
            ),
            "source_mode": source_mode,
            "source_order": (
                "dataset row order"
                if source_mode == "rows_api"
                else "first N WAV members in pinned tar archive order, joined to pinned TSV"
            ),
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
    parser.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default=DEFAULT_SPLIT)
    parser.add_argument("--source-mode", choices=sorted(ALLOWED_SOURCE_MODES), default="rows_api")
    args = parser.parse_args()
    fixture = build_fleurs_fixture(
        args.output_dir,
        args.fixture,
        samples_per_language=args.samples_per_language,
        split=args.split,
        source_mode=args.source_mode,
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
