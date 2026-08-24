from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from dubbing.media import ffmpeg_executable
from dubbing.transcription.job import media_duration_ms, source_sha256
from dubbing.transcription.models import (
    TranscriptionError,
    transcription_result_from_dict,
)


_SPAN_SCHEMA = "dubbing.cloud-teacher-canary-spans.v1"
_OUTPUT_SCHEMA = "dubbing.cloud-teacher-canary-preparation.v1"
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ALLOWED_TOP_LEVEL = {
    "schema_version",
    "canary_id",
    "parent_source_sha256",
    "parent_local_result_sha256",
    "network_allowed",
    "giga_admission_allowed",
    "spans",
}
_ALLOWED_SPAN_FIELDS = {
    "id",
    "role",
    "start_ms",
    "end_ms",
    "expected_clip_sha256",
    "expected_local_result_sha256",
}


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_existing_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        raise ValueError(f"{label} must not be a symlink")
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return path.resolve()


def _safe_new_directory(value: str | Path) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or path.exists():
        raise ValueError(f"output directory already exists or is a symlink: {path}")
    parent = path.parent
    if any(candidate.is_symlink() for candidate in parent.parents) or parent.is_symlink():
        raise ValueError(f"output parent must not be a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"output parent is not a safe directory: {parent}")
    return path


def _load_span_manifest(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load canary span manifest: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("canary span manifest must be a JSON object")
    unknown = set(document) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise ValueError(f"canary span manifest has unsupported fields: {sorted(unknown)}")
    if document.get("schema_version") != _SPAN_SCHEMA:
        raise ValueError(f"canary span manifest must use schema {_SPAN_SCHEMA}")
    canary_id = document.get("canary_id")
    if not isinstance(canary_id, str) or not _SLUG.fullmatch(canary_id):
        raise ValueError("canary_id must be a lowercase kebab-case slug")
    for field in ("parent_source_sha256", "parent_local_result_sha256"):
        if not isinstance(document.get(field), str) or not _DIGEST.fullmatch(document[field]):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if document.get("network_allowed") is not False:
        raise ValueError("offline canary preparation requires network_allowed=false")
    if document.get("giga_admission_allowed") is not False:
        raise ValueError("offline canary preparation requires giga_admission_allowed=false")
    raw_spans = document.get("spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("canary span manifest requires at least one span")

    spans = []
    identifiers = set()
    for raw in raw_spans:
        if not isinstance(raw, dict):
            raise ValueError("each canary span must be a JSON object")
        unknown = set(raw) - _ALLOWED_SPAN_FIELDS
        if unknown:
            raise ValueError(f"canary span has unsupported fields: {sorted(unknown)}")
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not _SLUG.fullmatch(identifier):
            raise ValueError("canary span id must be a lowercase kebab-case slug")
        if identifier in identifiers:
            raise ValueError(f"duplicate canary span id: {identifier}")
        identifiers.add(identifier)
        role = raw.get("role")
        if role not in {"control", "hard"}:
            raise ValueError("canary span role must be control or hard")
        if type(raw.get("start_ms")) is not int or type(raw.get("end_ms")) is not int:
            raise ValueError("canary span boundaries must be integer milliseconds")
        start_ms = raw["start_ms"]
        end_ms = raw["end_ms"]
        if start_ms < 0 or not 20_000 <= end_ms - start_ms <= 60_000:
            raise ValueError("canary spans must be 20-60 seconds with non-negative starts")
        for field in ("expected_clip_sha256", "expected_local_result_sha256"):
            value = raw.get(field)
            if value is not None and (not isinstance(value, str) or not _DIGEST.fullmatch(value)):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        spans.append({**raw, "start_ms": start_ms, "end_ms": end_ms})

    ordered = sorted(spans, key=lambda item: (item["start_ms"], item["end_ms"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start_ms"] < left["end_ms"]:
            raise ValueError("canary spans must not overlap")
    return {**document, "spans": spans}


def _extract_lossless_clip(source: Path, start_ms: int, end_ms: int, output: Path) -> None:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise TranscriptionError("ffmpeg is required for offline canary preparation")
    command = [
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
        "-map_metadata",
        "-1",
        "-c:a",
        "flac",
        "-compression_level",
        "8",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, timeout=600)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TranscriptionError("could not extract lossless canary clip") from exc


def _derive_local_result(parent, span: dict, clip_sha256: str, parent_result_sha256: str):
    start_ms = span["start_ms"]
    end_ms = span["end_ms"]
    crossing = [
        segment
        for segment in parent.segments
        if segment.start_ms < end_ms
        and segment.end_ms > start_ms
        and not (start_ms <= segment.start_ms and segment.end_ms <= end_ms)
    ]
    if crossing:
        raise ValueError(
            f"span {span['id']} crosses {len(crossing)} parent transcript segment boundary"
        )
    selected = tuple(
        segment
        for segment in parent.segments
        if start_ms <= segment.start_ms and segment.end_ms <= end_ms
    )
    if not selected:
        raise ValueError(f"span {span['id']} contains no complete transcript segment")
    shifted = []
    for segment in selected:
        words = tuple(
            replace(
                word,
                start_ms=word.start_ms - start_ms,
                end_ms=word.end_ms - start_ms,
            )
            for word in segment.words
        )
        shifted.append(
            replace(
                segment,
                start_ms=segment.start_ms - start_ms,
                end_ms=segment.end_ms - start_ms,
                words=words,
            )
        )
    provenance = {
        "derivation": "offline-cloud-teacher-canary-span-v1",
        "parent_source_sha256": parent.source_sha256,
        "parent_local_result_sha256": parent_result_sha256,
        "parent_start_ms": start_ms,
        "parent_end_ms": end_ms,
        "derived_clip_sha256": clip_sha256,
        "parent_backend_provenance": parent.provenance,
    }
    return replace(
        parent,
        segments=tuple(shifted),
        text=" ".join(segment.text for segment in shifted),
        duration_ms=end_ms - start_ms,
        source_sha256=clip_sha256,
        diarization=None,
        diagnostics=None,
        provenance=provenance,
    )


def prepare_cloud_teacher_canary(
    source: str | Path,
    parent_local_result: str | Path,
    span_manifest: str | Path,
    output: str | Path,
) -> dict:
    """Prepare hash-bound local canary inputs without credentials or network access."""

    source_path = _safe_existing_file(source, "source")
    parent_path = _safe_existing_file(parent_local_result, "parent local result")
    manifest_path = _safe_existing_file(span_manifest, "span manifest")
    span_document = _load_span_manifest(manifest_path)
    output_path = _safe_new_directory(output)

    source_digest = source_sha256(source_path)
    parent_digest = source_sha256(parent_path)
    if source_digest != span_document["parent_source_sha256"]:
        raise ValueError("source SHA-256 does not match canary span manifest")
    if parent_digest != span_document["parent_local_result_sha256"]:
        raise ValueError("parent local result SHA-256 does not match canary span manifest")
    try:
        parent_document = json.loads(parent_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("could not load parent local transcription result") from exc
    parent = transcription_result_from_dict(parent_document)
    if parent.source_sha256 != source_digest:
        raise ValueError("parent local result is not bound to the supplied source")
    source_duration = media_duration_ms(source_path)
    if parent.duration_ms is None:
        raise ValueError("parent local result must preserve source processing duration")
    for span in span_document["spans"]:
        if span["end_ms"] > source_duration or span["end_ms"] > parent.duration_ms:
            raise ValueError(f"span {span['id']} exceeds source or parent processing duration")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=str(output_path.parent))
    )
    try:
        clips = temporary / "clips"
        local_results = temporary / "local-results"
        clips.mkdir()
        local_results.mkdir()
        rows = []
        for span in span_document["spans"]:
            clip = clips / f"{span['id']}.flac"
            local_result_path = local_results / f"{span['id']}.json"
            _extract_lossless_clip(source_path, span["start_ms"], span["end_ms"], clip)
            expected_duration = span["end_ms"] - span["start_ms"]
            observed_duration = media_duration_ms(clip)
            if observed_duration != expected_duration:
                raise ValueError(
                    f"span {span['id']} duration drift: expected {expected_duration} ms, "
                    f"observed {observed_duration} ms"
                )
            clip_digest = source_sha256(clip)
            if span.get("expected_clip_sha256") not in {None, clip_digest}:
                raise ValueError(f"span {span['id']} derived clip SHA-256 drift")
            derived = _derive_local_result(parent, span, clip_digest, parent_digest)
            _atomic_json(local_result_path, derived.to_dict())
            reparsed = transcription_result_from_dict(
                json.loads(local_result_path.read_text(encoding="utf-8"))
            )
            if reparsed.source_sha256 != clip_digest or reparsed.duration_ms != expected_duration:
                raise ValueError(f"span {span['id']} derived local result binding drift")
            local_digest = source_sha256(local_result_path)
            if span.get("expected_local_result_sha256") not in {None, local_digest}:
                raise ValueError(f"span {span['id']} derived local result SHA-256 drift")
            rows.append(
                {
                    "id": span["id"],
                    "role": span["role"],
                    "parent_start_ms": span["start_ms"],
                    "parent_end_ms": span["end_ms"],
                    "duration_ms": expected_duration,
                    "clip": f"clips/{clip.name}",
                    "clip_sha256": clip_digest,
                    "local_result": f"local-results/{local_result_path.name}",
                    "local_result_sha256": local_digest,
                    "segment_count": len(reparsed.segments),
                    "uncertain_segment_count": sum(
                        segment.uncertain for segment in reparsed.segments
                    ),
                }
            )
        result = {
            "schema_version": _OUTPUT_SCHEMA,
            "canary_id": span_document["canary_id"],
            "source": str(source_path),
            "source_sha256": source_digest,
            "source_duration_ms": source_duration,
            "parent_local_result": str(parent_path),
            "parent_local_result_sha256": parent_digest,
            "span_manifest_sha256": source_sha256(manifest_path),
            "clip_encoding": "FLAC lossless over decoded source PCM",
            "network_used": False,
            "credentials_accessed": False,
            "cloud_allowed": False,
            "giga_admission_allowed": False,
            "spans": rows,
        }
        _atomic_json(temporary / "preparation-manifest.json", result)
        if output_path.exists() or output_path.is_symlink():
            raise ValueError(f"output directory appeared during preparation: {output_path}")
        os.rename(temporary, output_path)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare lossless, source-bound spans for an offline cloud-teacher canary; "
            "this command cannot access credentials or the network"
        )
    )
    parser.add_argument("source")
    parser.add_argument("--parent-local-result", required=True)
    parser.add_argument("--span-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = prepare_cloud_teacher_canary(
            args.source,
            args.parent_local_result,
            args.span_manifest,
            args.output,
        )
    except (OSError, TranscriptionError, ValueError) as exc:
        raise SystemExit(f"could not prepare cloud-teacher canary: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
