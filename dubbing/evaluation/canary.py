from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import sys
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

from dubbing.control_plane.cli import build_backend, build_retry_backend
from dubbing.evaluation.metrics import TimedText, evaluate_documents
from dubbing.media import ffmpeg_executable
from dubbing.transcription.adaptive import AdaptiveChunkPlanner, AdaptiveLongFormCoordinator
from dubbing.transcription.faster_whisper import FasterWhisperTranscriptionBackend
from dubbing.transcription.job import SourceBinding, create_source_binding
from dubbing.transcription.mlx_whisper import MLXWhisperTranscriptionBackend
from dubbing.transcription.models import (
    TranscriptionError,
    TranscriptionOptions,
    transcription_result_from_dict,
)
from dubbing.transcription.quality import evaluate_transcript_quality
from dubbing.transcription.speech_regions import FasterWhisperSileroSpeechRegionDetector
from dubbing.transcription.whisperkit import WhisperKitTranscriptionBackend


SCHEMA_VERSION = "dubbing.historical-canary.v1"
REPORT_SCHEMA_VERSION = "dubbing.historical-canary-report.v1"
SUMMARY_SCHEMA_VERSION = "dubbing.historical-canary-summary.v1"
RECEIPT_SCHEMA_VERSION = "dubbing.historical-canary-run-receipt.v2"
ATTEMPT_SCHEMA_VERSION = "dubbing.historical-canary-attempt.v1"

_ENTRY_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLES = {"development", "holdout", "stress"}
_QUALITY_STATUSES = {
    "PASS",
    "PASS_WITH_UNCERTAIN_SPANS",
    "REPROCESS_REQUIRED",
    "HUMAN_REVIEW_REQUIRED",
    "FAILED",
}
_LOCAL_BACKEND_PREFIXES = (
    "faster-whisper:",
    "mlx-whisper:",
    "whisperkit-local-server:",
)
_POLICY_KEYS = {
    "allowed_quality_statuses",
    "maximum_realtime_factor",
    "maximum_provisional_reference_wer",
    "window_ms",
    "accuracy_claim",
    "diarization_claim",
}
_EXECUTION_KEYS = {
    "candidate_languages",
    "target_seconds",
    "minimum_seconds",
    "maximum_seconds",
    "overlap_seconds",
    "minimum_silence_seconds",
    "language_retry_policy",
}


class _UnsetSpeechRegionDetector:
    pass


_UNSET_SPEECH_REGION_DETECTOR = _UnsetSpeechRegionDetector()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_sha256(document: dict) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_evidence(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"runtime executable is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _model_tree_evidence(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"local model directory is missing: {resolved}")
    digest = hashlib.sha256()
    total = 0
    count = 0
    for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(block)
                digest.update(block)
        count += 1
    if count == 0:
        raise ValueError(f"local model directory contains no files: {resolved}")
    return {
        "path": str(resolved),
        "algorithm": "sha256-relative-path-content-v1",
        "tree_sha256": digest.hexdigest(),
        "size_bytes": total,
        "file_count": count,
    }


def _installed_distribution_version(package: str) -> str:
    try:
        return distribution_version(package)
    except PackageNotFoundError as exc:
        raise ValueError(f"local backend runtime is not installed: {package}") from exc


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_json(path: Path, document: dict) -> None:
    """Create immutable evidence without overwriting an existing event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f"attempt evidence already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return converted


def _load_json_object(path: Path, label: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is malformed: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return document


def _validate_manifest_contract(document: dict) -> None:
    canary_id = document.get("canary_id")
    if not isinstance(canary_id, str) or not _ENTRY_ID.fullmatch(canary_id):
        raise ValueError("canary_id must be a lowercase kebab-case slug")
    required_languages = document.get("required_languages", [])
    if not isinstance(required_languages, list) or any(
        not isinstance(item, str) or not re.fullmatch(r"[a-z]{2,3}", item)
        for item in required_languages
    ):
        raise ValueError("required_languages must contain ISO-style lowercase language codes")
    if len(set(required_languages)) != len(required_languages):
        raise ValueError("required_languages must not contain duplicates")
    policy = document.get("policy", {})
    if not isinstance(policy, dict) or set(policy) - _POLICY_KEYS:
        unknown = sorted(set(policy) - _POLICY_KEYS) if isinstance(policy, dict) else []
        raise ValueError(f"canary policy contains unsupported fields: {unknown}")
    allowed = policy.get("allowed_quality_statuses", ["PASS"])
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(item not in _QUALITY_STATUSES for item in allowed)
    ):
        raise ValueError("allowed_quality_statuses contains an invalid status")
    if "maximum_realtime_factor" in policy:
        value = _finite_number(policy["maximum_realtime_factor"], "maximum_realtime_factor")
        if value == 0:
            raise ValueError("maximum_realtime_factor must be greater than zero")
    if "maximum_provisional_reference_wer" in policy:
        value = _finite_number(
            policy["maximum_provisional_reference_wer"],
            "maximum_provisional_reference_wer",
        )
        if value > 1:
            raise ValueError("maximum_provisional_reference_wer must not exceed 1")
    if "window_ms" in policy and (
        isinstance(policy["window_ms"], bool)
        or not isinstance(policy["window_ms"], int)
        or policy["window_ms"] <= 0
    ):
        raise ValueError("window_ms must be a positive integer")
    for claim in ("accuracy_claim", "diarization_claim"):
        if policy.get(claim, False) is not False:
            raise ValueError(f"historical canary must keep {claim} disabled")
    execution = document.get("execution", {})
    if not isinstance(execution, dict) or set(execution) - _EXECUTION_KEYS:
        unknown = sorted(set(execution) - _EXECUTION_KEYS) if isinstance(execution, dict) else []
        raise ValueError(f"canary execution contains unsupported fields: {unknown}")
    for key in ("target_seconds", "minimum_seconds", "maximum_seconds"):
        if key in execution and (
            isinstance(execution[key], bool)
            or not isinstance(execution[key], int)
            or execution[key] <= 0
        ):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("overlap_seconds", "minimum_silence_seconds"):
        if key in execution:
            _finite_number(execution[key], key)
    if "language_retry_policy" in execution and not isinstance(
        execution["language_retry_policy"], dict
    ):
        raise ValueError("language_retry_policy must be an object")
    candidate_languages = execution.get("candidate_languages")
    if candidate_languages is not None and (
        not isinstance(candidate_languages, list)
        or not candidate_languages
        or any(
            not isinstance(item, str) or not re.fullmatch(r"[a-z]{2,3}", item)
            for item in candidate_languages
        )
        or len(set(candidate_languages)) != len(candidate_languages)
    ):
        raise ValueError(
            "candidate_languages must be a nonempty unique list of ISO-style "
            "lowercase language codes"
        )
    minimum = execution.get("minimum_seconds", 60)
    target = execution.get("target_seconds", 240)
    maximum = execution.get("maximum_seconds", 480)
    if not minimum <= target <= maximum:
        raise ValueError(
            "execution durations must satisfy minimum_seconds <= target_seconds <= maximum_seconds"
        )


def _validate_entry_contract(entry: dict) -> None:
    entry_id = entry.get("id", "<unknown>")
    if entry.get("role") not in _ROLES:
        raise ValueError(f"{entry_id} has an unsupported role")
    for key in ("source", "reference"):
        item = entry.get(key)
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"{entry_id} {key} binding is incomplete")
        if not isinstance(item["path"], str):
            raise ValueError(f"{entry_id} {key} path must be a string")
        if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
            raise ValueError(f"{entry_id} {key} SHA-256 is malformed")
    duration_ms = entry["source"].get("duration_ms")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ValueError(f"{entry_id} source duration_ms must be a positive integer")
    evaluation_end_ms = entry["source"].get("evaluation_end_ms")
    if evaluation_end_ms is not None and (
        isinstance(evaluation_end_ms, bool)
        or not isinstance(evaluation_end_ms, int)
        or evaluation_end_ms <= 0
        or evaluation_end_ms > duration_ms
    ):
        raise ValueError(
            f"{entry_id} source evaluation_end_ms must satisfy 0 < evaluation_end_ms <= duration_ms"
        )
    processing_end_ms = entry["source"].get("processing_end_ms")
    if processing_end_ms is not None and (
        isinstance(processing_end_ms, bool)
        or not isinstance(processing_end_ms, int)
        or processing_end_ms <= 0
        or processing_end_ms > duration_ms
    ):
        raise ValueError(
            f"{entry_id} source processing_end_ms must satisfy 0 < processing_end_ms <= duration_ms"
        )
    if (
        evaluation_end_ms is not None
        and processing_end_ms is not None
        and evaluation_end_ms > processing_end_ms
    ):
        raise ValueError(f"{entry_id} source evaluation_end_ms must not exceed processing_end_ms")
    reference_kind = entry["reference"].get("kind")
    if not isinstance(reference_kind, str) or not reference_kind.strip():
        raise ValueError(f"{entry_id} reference kind is required")
    human_ground_truth = entry["reference"].get("human_ground_truth", False)
    if not isinstance(human_ground_truth, bool):
        raise ValueError(f"{entry_id} reference human_ground_truth must be boolean")


def load_canary_manifest(path: str | Path) -> tuple[Path, dict]:
    manifest_path = Path(path).resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load canary manifest: {manifest_path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"canary manifest must use schema {SCHEMA_VERSION}")
    _validate_manifest_contract(document)
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("canary manifest requires at least one entry")
    ids = [str(item.get("id", "")) for item in entries if isinstance(item, dict)]
    if len(ids) != len(entries) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("canary entry IDs must be unique non-empty strings")
    if any(not _ENTRY_ID.fullmatch(value) for value in ids):
        raise ValueError("canary entry IDs must be lowercase kebab-case slugs")
    for entry in entries:
        _validate_entry_contract(entry)
    roles = {str(item.get("role", "")) for item in entries}
    if not roles.issubset(_ROLES):
        raise ValueError("canary entries contain an unsupported role")
    if not _ROLES.issubset(roles):
        raise ValueError("canary manifest requires development, holdout and stress roles")
    if document.get("giga_admission_allowed") is not False:
        raise ValueError("historical canary must keep GIGA admission disabled")
    return manifest_path, document


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_canary_bindings(
    manifest_path: Path,
    manifest: dict,
    *,
    bindings_by_entry_id: dict[str, SourceBinding] | None = None,
) -> tuple[dict, ...]:
    root = manifest_path.parent
    bound = []
    for entry in manifest["entries"]:
        _validate_entry_contract(entry)
        resolved = dict(entry)
        for key in ("source", "reference"):
            item = entry.get(key)
            path = _resolve(root, item["path"])
            if not path.is_file():
                raise ValueError(f"{entry['id']} {key} is missing: {path}")
            if key == "source":
                source_binding = create_source_binding(
                    path,
                    expected_sha256=item["sha256"],
                )
                if bindings_by_entry_id is not None:
                    bindings_by_entry_id[entry["id"]] = source_binding
            else:
                actual = _sha256(path)
                if actual != item["sha256"]:
                    raise ValueError(
                        f"{entry['id']} {key} SHA-256 mismatch: expected {item['sha256']}"
                    )
            resolved[key] = {**item, "path": str(path)}
        bound.append(resolved)
    return tuple(bound)


def _corpus_binding(manifest: dict) -> tuple[dict, str]:
    document = {
        "canary_id": manifest["canary_id"],
        "entries": [
            {
                "id": entry["id"],
                "role": entry["role"],
                "source_sha256": entry["source"]["sha256"],
                "source_duration_ms": entry["source"]["duration_ms"],
                "source_evaluation_end_ms": entry["source"].get("evaluation_end_ms"),
                "source_processing_end_ms": entry["source"].get("processing_end_ms"),
                "reference_sha256": entry["reference"]["sha256"],
                "reference_kind": entry["reference"]["kind"],
            }
            for entry in manifest["entries"]
        ],
    }
    return document, _document_sha256(document)


def _implementation_binding(*backends) -> tuple[dict, str]:
    project_root = Path(__file__).resolve().parents[2]
    candidates = {
        Path(__file__).resolve(),
        (project_root / "dubbing/control_plane/cli.py").resolve(),
        (project_root / "dubbing/evaluation/metrics.py").resolve(),
        (project_root / "dubbing/media.py").resolve(),
    }
    candidates.update((project_root / "dubbing/transcription").glob("*.py"))
    for backend in backends:
        if backend is None:
            continue
        backend_source = inspect.getsourcefile(type(backend))
        if backend_source:
            candidates.add(Path(backend_source).resolve())
    files = []
    for path in sorted((item for item in candidates if item.is_file()), key=str):
        try:
            label = str(path.relative_to(project_root))
        except ValueError:
            label = str(path)
        files.append({"path": label, "sha256": _sha256(path)})
    document = {"algorithm": "sha256-file-set-v1", "files": files}
    return document, _document_sha256(document)


def _candidate_languages(manifest: dict) -> tuple[str, ...]:
    configured = manifest.get("execution", {}).get("candidate_languages")
    return tuple(configured or manifest.get("required_languages") or ("en", "ru", "ro", "ko"))


def _backend_evidence(backend, *, label: str) -> dict:
    identity = getattr(backend, "identity", None)
    if not isinstance(identity, str) or not identity.startswith(_LOCAL_BACKEND_PREFIXES):
        raise ValueError(f"historical canary requires a recognised local {label} backend")
    source = inspect.getsourcefile(type(backend))
    source_path = Path(source).resolve() if source else None
    if source_path is not None and not source_path.is_file():
        source_path = None
    backend_evidence = {
        "identity": identity,
        "model": getattr(backend, "model", None),
        "implementation": {
            "class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "source_path": str(source_path) if source_path else None,
            "source_sha256": _sha256(source_path) if source_path else None,
        },
        "cloud_allowed": False,
    }
    if isinstance(
        backend,
        (MLXWhisperTranscriptionBackend, FasterWhisperTranscriptionBackend),
    ):
        package = (
            "mlx-whisper"
            if isinstance(backend, MLXWhisperTranscriptionBackend)
            else "faster-whisper"
        )
        backend_evidence["runtime_version"] = _installed_distribution_version(package)
        model_path = Path(backend.model).expanduser().resolve()
        model_must_be_local = (
            isinstance(backend, MLXWhisperTranscriptionBackend) or label == "retry"
        )
        if model_must_be_local and not model_path.is_dir():
            raise ValueError(
                f"historical canary requires an existing local {label} model directory"
            )
        if model_path.is_dir():
            backend_evidence["local_model"] = _model_tree_evidence(model_path)
    server = getattr(backend, "server", None)
    if isinstance(backend, WhisperKitTranscriptionBackend) and server is None:
        raise ValueError(
            "historical canary requires an owned local WhisperKit server "
            "to bind its runtime executable"
        )
    if server is not None:
        backend_evidence.update(
            {
                "model_tree_sha256": server.model_sha256,
                "model_size_bytes": server.model_size_bytes,
                "runtime_version": server.version,
                "runtime_executable": _file_evidence(Path(server.executable)),
            }
        )
    return backend_evidence


def _detector_evidence(detector) -> dict | None:
    if detector is None:
        return None
    identity = getattr(detector, "identity", None)
    if not isinstance(identity, str) or not identity:
        raise ValueError("historical canary speech detector requires a stable identity")
    source = inspect.getsourcefile(type(detector))
    source_path = Path(source).resolve() if source else None
    if source_path is not None and not source_path.is_file():
        source_path = None
    return {
        "identity": identity,
        "implementation": {
            "class": f"{type(detector).__module__}.{type(detector).__qualname__}",
            "source_path": str(source_path) if source_path else None,
            "source_sha256": _sha256(source_path) if source_path else None,
        },
        "local_only": True,
    }


def _execution_fingerprint(
    manifest: dict,
    backend,
    retry_backend=None,
    silence_verification_detector=None,
    targeted_retry_region_detector=None,
) -> tuple[dict, str]:
    backend_evidence = _backend_evidence(backend, label="transcription")
    retry_evidence = (
        _backend_evidence(retry_backend, label="retry") if retry_backend is not None else None
    )
    if retry_evidence is not None and retry_evidence["identity"] == backend_evidence["identity"]:
        raise ValueError("independent retry backend resolved to the primary backend identity")
    corpus, corpus_sha256 = _corpus_binding(manifest)
    implementation, implementation_sha256 = _implementation_binding(
        backend,
        retry_backend,
        silence_verification_detector,
        targeted_retry_region_detector,
    )
    execution_config = dict(manifest.get("execution", {}))
    execution_config["candidate_languages"] = list(_candidate_languages(manifest))
    ffmpeg = ffmpeg_executable()
    execution = {
        "schema_version": "dubbing.historical-canary-execution.v2",
        "backend": backend_evidence,
        "retry_backend": retry_evidence,
        "silence_verification_detector": _detector_evidence(silence_verification_detector),
        "targeted_retry_region_detector": _detector_evidence(targeted_retry_region_detector),
        "required_languages": manifest.get("required_languages", []),
        "execution": execution_config,
        "policy": manifest.get("policy", {}),
        "corpus": corpus,
        "corpus_sha256": corpus_sha256,
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "media_runtime": {
            "ffmpeg": _file_evidence(Path(ffmpeg)) if ffmpeg is not None else None,
        },
        "giga_admission_allowed": False,
    }
    return execution, _document_sha256(execution)


def _admit_frozen_execution(
    path: Path, execution: dict, fingerprint: str, *, roles: set[str]
) -> None:
    document = {**execution, "fingerprint_sha256": fingerprint}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("frozen canary execution is malformed") from exc
        if existing != document:
            raise ValueError("canary execution changed after it was frozen")
        return
    if "development" not in roles:
        raise ValueError("run the development canary before any holdout or stress entry")
    _atomic_json(path, document)


def _attempt_events(entry_output: Path) -> list[dict]:
    events = []
    for path in sorted((entry_output / "attempts").glob("*.json")):
        event = _load_json_object(path, "canary attempt event")
        if event.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
            raise ValueError(f"attempt event has an unsupported schema: {path}")
        if not isinstance(event.get("attempt_index"), int) or event["attempt_index"] < 1:
            raise ValueError(f"attempt event has an invalid index: {path}")
        if event.get("event") not in {"STARTED", "COMPLETED", "FAILED"}:
            raise ValueError(f"attempt event has an invalid event type: {path}")
        events.append(event)
    return events


def _start_attempt(
    entry_output: Path,
    entry: dict,
    fingerprint: str,
    *,
    evaluate_only: bool,
) -> tuple[int, int]:
    events = _attempt_events(entry_output)
    attempt_index = max((item["attempt_index"] for item in events), default=0) + 1
    started_ns = time.time_ns()
    event = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_index": attempt_index,
        "event": "STARTED",
        "entry_id": entry["id"],
        "source_sha256": entry["source"]["sha256"],
        "execution_fingerprint_sha256": fingerprint,
        "evaluation_only": evaluate_only,
        "started_at_unix_ns": started_ns,
    }
    _atomic_create_json(entry_output / "attempts" / f"{attempt_index:06d}-started.json", event)
    return attempt_index, started_ns


def _finish_attempt(
    entry_output: Path,
    entry: dict,
    fingerprint: str,
    *,
    attempt_index: int,
    started_ns: int,
    evaluate_only: bool,
    elapsed_seconds: float,
    status: str,
    result_sha256: str | None = None,
    quality_report_sha256: str | None = None,
    previous_result_sha256: str | None = None,
    result_byte_identical_to_previous: bool | None = None,
    error: BaseException | None = None,
) -> None:
    if status not in {"COMPLETED", "FAILED"}:
        raise ValueError("attempt terminal status is invalid")
    event = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_index": attempt_index,
        "event": status,
        "entry_id": entry["id"],
        "source_sha256": entry["source"]["sha256"],
        "execution_fingerprint_sha256": fingerprint,
        "evaluation_only": evaluate_only,
        "started_at_unix_ns": started_ns,
        "finished_at_unix_ns": time.time_ns(),
        "elapsed_seconds": round(_finite_number(elapsed_seconds, "attempt elapsed_seconds"), 3),
        "result_sha256": result_sha256,
        "quality_report_sha256": quality_report_sha256,
        "previous_result_sha256": previous_result_sha256,
        "result_byte_identical_to_previous": result_byte_identical_to_previous,
    }
    if error is not None:
        event["error_type"] = type(error).__name__
        event["error_message"] = str(error)
    _atomic_create_json(
        entry_output / "attempts" / f"{attempt_index:06d}-{status.lower()}.json",
        event,
    )


def _attempt_accounting(
    entry_output: Path,
    *,
    entry: dict | None = None,
    fingerprint: str | None = None,
) -> dict:
    events = _attempt_events(entry_output)
    started: dict[int, dict] = {}
    terminal: dict[int, dict] = {}
    for event in events:
        index = event["attempt_index"]
        if entry is not None and (
            event.get("entry_id") != entry["id"]
            or event.get("source_sha256") != entry["source"]["sha256"]
        ):
            raise ValueError(f"attempt {index} entry binding mismatch")
        if fingerprint is not None and event.get("execution_fingerprint_sha256") != fingerprint:
            raise ValueError(f"attempt {index} execution binding mismatch")
        if not isinstance(event.get("evaluation_only"), bool):
            raise ValueError(f"attempt {index} evaluation provenance is malformed")
        started_ns = event.get("started_at_unix_ns")
        if isinstance(started_ns, bool) or not isinstance(started_ns, int) or started_ns <= 0:
            raise ValueError(f"attempt {index} start timestamp is malformed")
        if event["event"] == "STARTED":
            if index in started:
                raise ValueError(f"attempt {index} has duplicate STARTED evidence")
            started[index] = event
        else:
            finished_ns = event.get("finished_at_unix_ns")
            if (
                isinstance(finished_ns, bool)
                or not isinstance(finished_ns, int)
                or finished_ns < started_ns
            ):
                raise ValueError(f"attempt {index} finish timestamp is malformed")
            _finite_number(event.get("elapsed_seconds"), "attempt elapsed_seconds")
            if index in terminal:
                raise ValueError(f"attempt {index} has duplicate terminal evidence")
            terminal[index] = event
    if set(terminal) - set(started):
        raise ValueError("attempt ledger contains terminal evidence without STARTED evidence")
    for index, event in terminal.items():
        if event["started_at_unix_ns"] != started[index]["started_at_unix_ns"]:
            raise ValueError(f"attempt {index} terminal evidence does not bind to STARTED")
    completed = [
        terminal[index] for index in sorted(terminal) if terminal[index]["event"] == "COMPLETED"
    ]
    for event in completed:
        _finite_number(event.get("elapsed_seconds"), "attempt elapsed_seconds")
    transcription = [item for item in completed if item.get("evaluation_only") is False]
    evaluations = [item for item in completed if item.get("evaluation_only") is True]
    replay_mismatches = [
        item
        for item in transcription
        if item.get("previous_result_sha256") is not None
        and item.get("result_byte_identical_to_previous") is not True
    ]
    latest_attempt_index = max(started, default=None)
    latest_terminal = terminal.get(latest_attempt_index)
    return {
        "attempt_count": len(started),
        "completed_attempt_count": len(completed),
        "failed_attempt_count": sum(item["event"] == "FAILED" for item in terminal.values()),
        "unterminated_attempt_indices": sorted(set(started) - set(terminal)),
        "runtime_accounting_complete": set(started) == set(terminal),
        "initial_runtime_seconds": (
            float(transcription[0]["elapsed_seconds"]) if transcription else None
        ),
        "cumulative_runtime_seconds": round(
            sum(float(item["elapsed_seconds"]) for item in transcription), 3
        ),
        "transcription_attempt_count": len(transcription),
        "evaluation_attempt_count": len(evaluations),
        "replay_mismatch_count": len(replay_mismatches),
        "latest_attempt_index": latest_attempt_index,
        "latest_terminal_status": (
            latest_terminal["event"] if latest_terminal is not None else None
        ),
    }


def _validate_receipt(
    receipt_path: Path,
    entry: dict,
    fingerprint: str,
    *,
    require_report: bool = True,
) -> dict:
    receipt = _load_json_object(receipt_path, "canary run receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError(f"canary receipt must use schema {RECEIPT_SCHEMA_VERSION}")
    expected = {
        "entry_id": entry["id"],
        "source_sha256": entry["source"]["sha256"],
        "execution_fingerprint_sha256": fingerprint,
        "cloud_allowed": False,
        "giga_admission_allowed": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{entry['id']} receipt {key} binding mismatch")
    for key in (
        "initial_runtime_seconds",
        "cumulative_runtime_seconds",
        "last_run_runtime_seconds",
    ):
        _finite_number(receipt.get(key), f"{entry['id']} receipt {key}")
    if receipt.get("runtime_accounting_complete") is not True:
        raise ValueError(f"{entry['id']} receipt has incomplete runtime accounting")
    accounting = _attempt_accounting(receipt_path.parent, entry=entry, fingerprint=fingerprint)
    if accounting["unterminated_attempt_indices"]:
        raise ValueError(f"{entry['id']} receipt has newer unterminated attempt evidence")
    if accounting["latest_terminal_status"] != "COMPLETED":
        raise ValueError(f"{entry['id']} receipt latest attempt did not complete")
    if receipt.get("last_attempt_index") != accounting["latest_attempt_index"]:
        raise ValueError(f"{entry['id']} receipt does not bind to latest attempt")
    for key in (
        "initial_runtime_seconds",
        "cumulative_runtime_seconds",
        "runtime_accounting_complete",
        "unterminated_attempt_indices",
        "replay_count",
        "evaluation_count",
        "replay_mismatch_count",
    ):
        accounting_key = {
            "replay_count": "transcription_attempt_count",
            "evaluation_count": "evaluation_attempt_count",
        }.get(key, key)
        expected = accounting.get(accounting_key)
        if key == "replay_count":
            expected = max(0, expected - 1)
        if receipt.get(key) != expected:
            raise ValueError(f"{entry['id']} receipt {key} disagrees with attempt ledger")
    result_path = receipt_path.parent / "result.json"
    quality_path = receipt_path.parent / "quality-report.json"
    if not result_path.is_file() or receipt.get("result_sha256") != _sha256(result_path):
        raise ValueError(f"{entry['id']} receipt result binding mismatch")
    if not quality_path.is_file() or receipt.get("quality_report_sha256") != _sha256(quality_path):
        raise ValueError(f"{entry['id']} receipt quality binding mismatch")
    if require_report:
        report_path = receipt_path.parent / "canary-report.json"
        if not report_path.is_file() or receipt.get("canary_report_sha256") != _sha256(report_path):
            raise ValueError(f"{entry['id']} receipt report binding mismatch")
    return receipt


def _validate_report(path: Path, entry: dict, fingerprint: str) -> dict:
    report = _load_json_object(path, "canary report")
    expected = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "entry_id": entry["id"],
        "role": entry["role"],
        "source_sha256": entry["source"]["sha256"],
        "reference_sha256": entry["reference"]["sha256"],
        "execution_fingerprint_sha256": fingerprint,
        "giga_admission_allowed": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{entry['id']} report {key} binding mismatch")
    scope = report.get("evaluation_scope")
    if not isinstance(scope, dict) or scope.get("evaluation_end_ms") != entry["source"].get(
        "evaluation_end_ms"
    ):
        raise ValueError(f"{entry['id']} report evaluation scope binding mismatch")
    processing_scope = report.get("processing_scope")
    if not isinstance(processing_scope, dict) or processing_scope.get("processing_end_ms") != entry[
        "source"
    ].get("processing_end_ms"):
        raise ValueError(f"{entry['id']} report processing scope binding mismatch")
    gate = report.get("operational_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not (gate.get("status") == "PASS"):
        raise ValueError(f"{entry['id']} report operational gate is malformed")
    _finite_number(report.get("runtime_seconds"), f"{entry['id']} report runtime")
    _finite_number(report.get("realtime_factor"), f"{entry['id']} report realtime factor")
    return report


def _require_development_pass(output: Path, entries: tuple[dict, ...], fingerprint: str) -> None:
    development = [entry for entry in entries if entry["role"] == "development"]
    if not development:
        raise ValueError("frozen corpus has no development entry")
    for entry in development:
        entry_output = output / entry["id"]
        receipt = _validate_receipt(entry_output / "canary-run-receipt.json", entry, fingerprint)
        report = _validate_report(entry_output / "canary-report.json", entry, fingerprint)
        if not report["operational_gate"]["passed"]:
            raise ValueError(f"development canary {entry['id']} did not PASS")
        if receipt.get("origin") != "LOCAL_TRANSCRIPTION":
            raise ValueError(f"development canary {entry['id']} lacks a transcription receipt")


def _result_scoped_to_end(result, evaluation_end_ms: int):
    scoped_segments = []
    crossing_segment_count = 0
    for segment in result.segments:
        if segment.start_ms >= evaluation_end_ms:
            continue
        if segment.end_ms <= evaluation_end_ms:
            scoped_segments.append(segment)
            continue
        crossing_segment_count += 1
        scoped_words = []
        for word in segment.words:
            if word.start_ms >= evaluation_end_ms:
                continue
            scoped_words.append(replace(word, end_ms=min(word.end_ms, evaluation_end_ms)))
        scoped_segments.append(
            replace(
                segment,
                end_ms=evaluation_end_ms,
                words=tuple(scoped_words),
            )
        )
    scoped = replace(
        result,
        segments=tuple(scoped_segments),
        text=" ".join(segment.text for segment in scoped_segments).strip(),
        duration_ms=evaluation_end_ms,
        provenance={
            **(result.provenance or {}),
            "evaluation_scope": {
                "kind": "SOURCE_TIME_BOUNDARY",
                "evaluation_end_ms": evaluation_end_ms,
                "derived_for_canary_only": True,
            },
        },
    )
    out_of_scope = tuple(
        segment for segment in result.segments if segment.end_ms > evaluation_end_ms
    )
    out_text = " ".join(segment.text for segment in out_of_scope).strip()
    observation = {
        "used_for_operational_gate": False,
        "start_ms": evaluation_end_ms,
        "end_ms": result.duration_ms,
        "segment_count": len(out_of_scope),
        "wholly_out_of_scope_segment_count": sum(
            segment.start_ms >= evaluation_end_ms for segment in result.segments
        ),
        "boundary_crossing_segment_count": crossing_segment_count,
        "text_character_count": len(out_text),
        "text_sha256": hashlib.sha256(out_text.encode("utf-8")).hexdigest(),
    }
    return scoped, observation


def evaluate_canary_result(
    entry: dict,
    result_document: dict,
    quality: dict,
    *,
    runtime_seconds: float,
    policy: dict,
    execution_fingerprint: str,
    replay_result_byte_identical: bool | None = None,
    runtime_accounting_complete: bool = True,
    evaluation_provenance: dict | None = None,
) -> dict:
    runtime_seconds = _finite_number(runtime_seconds, "runtime_seconds")
    full_result = transcription_result_from_dict(result_document)
    source_duration_ms = int(entry["source"]["duration_ms"])
    processing_end_ms = entry["source"].get("processing_end_ms")
    processed_duration_ms = processing_end_ms or source_duration_ms
    evaluation_end_ms = entry["source"].get("evaluation_end_ms")
    if evaluation_end_ms is not None:
        result, out_of_scope_observation = _result_scoped_to_end(full_result, evaluation_end_ms)
        operational_quality = evaluate_transcript_quality(
            result, expected_duration_ms=evaluation_end_ms
        ).to_dict()
    else:
        result = full_result
        operational_quality = quality
        out_of_scope_observation = {
            "used_for_operational_gate": False,
            "status": "NOT_APPLICABLE",
            "reason": "No source evaluation boundary is configured.",
            "segment_count": 0,
        }
    reference_path = Path(entry["reference"]["path"])
    reference_text = reference_path.read_text(encoding="utf-8")
    candidate_segments = tuple(
        TimedText(
            item.start_ms,
            item.end_ms,
            item.text,
            item.language,
            item.speaker,
        )
        for item in result.segments
    )
    metrics = evaluate_documents(
        reference_text,
        result.text,
        candidate_segments=candidate_segments,
        duration_ms=result.duration_ms,
        window_ms=int(policy.get("window_ms", 60_000)),
    )
    expected_source = entry["source"]["sha256"]
    reasons = []
    if result.source_sha256 != expected_source:
        reasons.append("RESULT_SOURCE_BINDING_MISMATCH")
    result_escapes_processing_scope = any(
        segment.end_ms > processed_duration_ms
        or any(word.end_ms > processed_duration_ms for word in segment.words)
        for segment in full_result.segments
    )
    if full_result.duration_ms != processed_duration_ms or result_escapes_processing_scope:
        reasons.append("RESULT_PROCESSING_SCOPE_MISMATCH")
    if replay_result_byte_identical is False:
        reasons.append("CHECKPOINT_REPLAY_RESULT_MISMATCH")
    if not runtime_accounting_complete:
        reasons.append("ATTEMPT_LEDGER_INCOMPLETE")
    quality_status = str(operational_quality.get("status", "FAILED"))
    allowed_quality_states = set(policy.get("allowed_quality_statuses", ["PASS"]))
    if quality_status not in allowed_quality_states:
        reasons.append(f"QUALITY_{quality_status}")
    quality_metrics = operational_quality.get("metrics", {})
    if int(quality_metrics.get("repetition_finding_count", 0)):
        reasons.append("PATHOLOGICAL_REPETITION")
    fallback_exhausted_count = sum(
        bool(item.diagnostics and item.diagnostics.fallback_exhausted) for item in result.segments
    )
    if fallback_exhausted_count:
        reasons.append("DECODER_FALLBACK_EXHAUSTED")
    processed_duration_seconds = max(0.001, processed_duration_ms / 1000)
    full_source_duration_seconds = max(0.001, source_duration_ms / 1000)
    realtime_factor = runtime_seconds / processed_duration_seconds
    full_source_realtime_factor_observation = runtime_seconds / full_source_duration_seconds
    if realtime_factor > float(policy.get("maximum_realtime_factor", 0.25)):
        reasons.append("RUNTIME_FACTOR_EXCEEDED")
    provisional_wer = float(metrics["wer"]["rate"])
    agreement_warnings = []
    if provisional_wer > float(policy.get("maximum_provisional_reference_wer", 0.60)):
        agreement_warnings.append("PROVISIONAL_REFERENCE_DISAGREEMENT_HIGH")
    if not metrics["time_window_evaluation_available"]:
        agreement_warnings.append("TIMESTAMPED_REFERENCE_UNAVAILABLE")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "entry_id": entry["id"],
        "role": entry["role"],
        "source_sha256": expected_source,
        "reference_sha256": entry["reference"]["sha256"],
        "reference_kind": entry["reference"]["kind"],
        "reference_is_human_ground_truth": False,
        "execution_fingerprint_sha256": execution_fingerprint,
        "evaluation_provenance": evaluation_provenance
        or {
            "mode": "TRANSCRIPTION_RUN",
        },
        "runtime_seconds": runtime_seconds,
        "realtime_factor": realtime_factor,
        "processing_scope": {
            "kind": "SOURCE_TIME_CAP" if processing_end_ms is not None else "FULL_SOURCE",
            "processing_end_ms": processing_end_ms,
            "processed_duration_ms": processed_duration_ms,
            "full_source_duration_ms": source_duration_ms,
            "unprocessed_tail_duration_ms": source_duration_ms - processed_duration_ms,
            "full_source_sha256_bound": True,
            "realtime_factor_gate_denominator": "processed_duration_ms",
            "full_source_realtime_factor_observation": (full_source_realtime_factor_observation),
        },
        "decoder_fallback_exhausted_segment_count": fallback_exhausted_count,
        "evaluation_scope": {
            "kind": ("SOURCE_TIME_BOUNDARY" if evaluation_end_ms is not None else "FULL_SOURCE"),
            "evaluation_end_ms": evaluation_end_ms,
            "source_duration_ms": entry["source"]["duration_ms"],
            "operational_quality_recomputed": evaluation_end_ms is not None,
            "full_result_preserved": True,
            "full_quality_report_preserved": True,
        },
        "quality": operational_quality,
        "full_source_quality_observation": {
            "used_for_operational_gate": evaluation_end_ms is None,
            "quality": quality,
        },
        "full_source_evidence": {
            "result_document_sha256": _document_sha256(result_document),
            "quality_document_sha256": _document_sha256(quality),
            "artifacts_modified_for_scope": False,
        },
        "out_of_scope_observation": out_of_scope_observation,
        "provisional_reference_metrics": metrics,
        "provisional_reference_comparison_scope": {
            "reference_scope": "WHOLE_UNTIMED_REFERENCE",
            "candidate_scope": (
                "SOURCE_TIME_BOUNDARY" if evaluation_end_ms is not None else "FULL_SOURCE"
            ),
            "observation_only": True,
            "reference_text_was_trimmed": False,
        },
        "operational_gate": {
            "status": "PASS" if not reasons else "FAIL",
            "passed": not reasons,
            "failure_reasons": reasons,
        },
        "agreement_observation": {
            "status": "WARNING" if agreement_warnings else "WITHIN_CANARY_BOUND",
            "warnings": agreement_warnings,
            "accuracy_certified": False,
            "limitation": (
                "MacWhisper text is provisional, untimed and has no speaker labels; "
                "it cannot be safely trimmed to a source-time boundary. Whole-reference "
                "WER is observation-only drift evidence, not ground-truth accuracy."
            ),
        },
        "diarization_claim": {
            "status": "NOT_MEASURED",
            "reason": "The historical references contain no trusted speaker labels.",
        },
        "giga_admission_allowed": False,
    }


def _summary(
    manifest: dict,
    reports: list[dict],
    execution_fingerprint: str,
    *,
    invalid_evidence: dict[str, str] | None = None,
) -> dict:
    invalid_evidence = invalid_evidence or {}
    failures = [
        report["entry_id"] for report in reports if not report["operational_gate"]["passed"]
    ]
    roles_run = sorted({report["role"] for report in reports})
    expected = {entry["id"] for entry in manifest["entries"]}
    completed = {report["entry_id"] for report in reports}
    missing = sorted(expected - completed)
    status = "FAIL" if failures or invalid_evidence else ("PARTIAL" if missing else "PASS")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "canary_id": manifest["canary_id"],
        "execution_fingerprint_sha256": execution_fingerprint,
        "status": status,
        "operational_failures": failures,
        "invalid_evidence": invalid_evidence,
        "completed_entries": sorted(completed),
        "missing_entries": missing,
        "roles_run": roles_run,
        "accuracy_certified": False,
        "diarization_certified": False,
        "giga_admission_allowed": False,
    }


def _aggregate_summary(
    output: Path,
    manifest: dict,
    entries: tuple[dict, ...],
    fingerprint: str,
) -> dict:
    reports = []
    invalid_evidence = {}
    for entry in entries:
        report_path = output / entry["id"] / "canary-report.json"
        receipt_path = output / entry["id"] / "canary-run-receipt.json"
        if not report_path.exists() and not receipt_path.exists():
            continue
        try:
            _validate_receipt(receipt_path, entry, fingerprint)
            reports.append(_validate_report(report_path, entry, fingerprint))
        except ValueError as exc:
            invalid_evidence[entry["id"]] = str(exc)
    summary = _summary(
        manifest,
        reports,
        fingerprint,
        invalid_evidence=invalid_evidence,
    )
    _atomic_json(output / "summary.json", summary)
    return summary


def run_canary(
    manifest_path: str | Path,
    output_dir: str | Path,
    backend,
    *,
    retry_backend=None,
    speech_region_detector=None,
    silence_verification_detector=_UNSET_SPEECH_REGION_DETECTOR,
    targeted_retry_region_detector=_UNSET_SPEECH_REGION_DETECTOR,
    selected_ids: set[str] | None = None,
    evaluate_only: bool = False,
) -> dict:
    resolved_silence_verification_detector = (
        speech_region_detector
        if isinstance(silence_verification_detector, _UnsetSpeechRegionDetector)
        else silence_verification_detector
    )
    resolved_targeted_retry_region_detector = (
        speech_region_detector
        if isinstance(targeted_retry_region_detector, _UnsetSpeechRegionDetector)
        else targeted_retry_region_detector
    )
    manifest_path, manifest = load_canary_manifest(manifest_path)
    all_entries = tuple(dict(entry) for entry in manifest["entries"])
    bindings_by_entry_id: dict[str, SourceBinding] = {}
    if selected_ids:
        unknown = selected_ids - {entry["id"] for entry in manifest["entries"]}
        if unknown:
            raise ValueError(f"unknown canary entries: {', '.join(sorted(unknown))}")
        selected_manifest = {
            **manifest,
            "entries": [entry for entry in manifest["entries"] if entry["id"] in selected_ids],
        }
        selected_entries = validate_canary_bindings(
            manifest_path,
            selected_manifest,
            bindings_by_entry_id=bindings_by_entry_id,
        )
    else:
        selected_entries = validate_canary_bindings(
            manifest_path,
            manifest,
            bindings_by_entry_id=bindings_by_entry_id,
        )
    selected_by_id = {entry["id"]: entry for entry in selected_entries}
    entries = tuple(
        selected_by_id[entry["id"]]
        for role in ("development", "holdout", "stress")
        for entry in manifest["entries"]
        if entry["role"] == role and entry["id"] in selected_by_id
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    execution, fingerprint = _execution_fingerprint(
        manifest,
        backend,
        retry_backend,
        resolved_silence_verification_detector,
        resolved_targeted_retry_region_detector,
    )
    _admit_frozen_execution(
        output / "frozen-execution.json",
        execution,
        fingerprint,
        roles={entry["role"] for entry in entries},
    )
    # Revalidate all prior evidence before admitting another sequential step. This
    # also replaces a stale PASS summary if a receipt or report was corrupted.
    _aggregate_summary(output, manifest, all_entries, fingerprint)
    config = manifest.get("execution", {})
    candidate_languages = _candidate_languages(manifest)
    planner = AdaptiveChunkPlanner(
        target_seconds=int(config.get("target_seconds", 240)),
        minimum_seconds=int(config.get("minimum_seconds", 60)),
        maximum_seconds=int(config.get("maximum_seconds", 480)),
        overlap_seconds=float(config.get("overlap_seconds", 2.0)),
    )
    for entry in entries:
        if entry["role"] != "development":
            _require_development_pass(output, all_entries, fingerprint)
        entry_output = output / entry["id"]
        result_path = entry_output / "result.json"
        quality_path = entry_output / "quality-report.json"
        receipt_path = entry_output / "canary-run-receipt.json"
        previous_result_sha256 = _sha256(result_path) if result_path.is_file() else None
        previous_receipt = None
        if receipt_path.exists():
            previous_receipt = _validate_receipt(
                receipt_path, entry, fingerprint, require_report=True
            )
        elif previous_result_sha256 is not None:
            raise ValueError(f"{entry['id']} has a result without a validated receipt")
        if evaluate_only and previous_receipt is None:
            raise ValueError(
                f"{entry['id']} evaluate-only requires a validated transcription receipt"
            )
        origin_receipt_sha256 = _sha256(receipt_path) if previous_receipt else None
        attempt_index, started_ns = _start_attempt(
            entry_output,
            entry,
            fingerprint,
            evaluate_only=evaluate_only,
        )
        started = time.monotonic()
        terminal_written = False
        try:
            if not evaluate_only:
                coordinator = AdaptiveLongFormCoordinator(
                    backend,
                    entry_output,
                    planner=planner,
                    retry_backend=retry_backend,
                    silence_verification_detector=(resolved_silence_verification_detector),
                    targeted_retry_region_detector=(resolved_targeted_retry_region_detector),
                    candidate_languages=candidate_languages,
                    minimum_silence_seconds=float(config.get("minimum_silence_seconds", 0.7)),
                    language_retry_policy=dict(config.get("language_retry_policy", {})),
                )
                _, quality = coordinator.run(
                    Path(entry["source"]["path"]),
                    TranscriptionOptions(task="transcribe", word_timestamps=True),
                    source_binding=bindings_by_entry_id[entry["id"]],
                    processing_end_ms=entry["source"].get("processing_end_ms"),
                )
                current_execution, current_fingerprint = _execution_fingerprint(
                    manifest,
                    backend,
                    retry_backend,
                    resolved_silence_verification_detector,
                    resolved_targeted_retry_region_detector,
                )
                _admit_frozen_execution(
                    output / "frozen-execution.json",
                    current_execution,
                    current_fingerprint,
                    roles={entry["role"] for entry in entries},
                )
            else:
                if not result_path.is_file() or not quality_path.is_file():
                    raise ValueError(f"{entry['id']} has no completed result to evaluate")
                quality = _load_json_object(quality_path, "quality report")
            current_runtime_seconds = round(time.monotonic() - started, 3)
            result_document = _load_json_object(result_path, "transcription result")
            result_sha256 = _sha256(result_path)
            quality_sha256 = _sha256(quality_path)
            replay_match = (
                previous_result_sha256 == result_sha256
                if previous_result_sha256 is not None and not evaluate_only
                else None
            )
            _finish_attempt(
                entry_output,
                entry,
                fingerprint,
                attempt_index=attempt_index,
                started_ns=started_ns,
                evaluate_only=evaluate_only,
                elapsed_seconds=current_runtime_seconds,
                status="COMPLETED",
                result_sha256=result_sha256,
                quality_report_sha256=quality_sha256,
                previous_result_sha256=previous_result_sha256,
                result_byte_identical_to_previous=replay_match,
            )
            terminal_written = True
            accounting = _attempt_accounting(entry_output, entry=entry, fingerprint=fingerprint)
            initial_runtime_seconds = accounting["initial_runtime_seconds"]
            if initial_runtime_seconds is None:
                raise ValueError(f"{entry['id']} has no completed transcription attempt")
            provenance = {
                "mode": "EVALUATE_ONLY" if evaluate_only else "TRANSCRIPTION_RUN",
                "attempt_index": attempt_index,
                "origin_receipt_sha256": origin_receipt_sha256,
                "result_sha256": result_sha256,
                "quality_report_sha256": quality_sha256,
            }
            report = evaluate_canary_result(
                entry,
                result_document,
                quality,
                runtime_seconds=initial_runtime_seconds,
                policy=manifest.get("policy", {}),
                execution_fingerprint=fingerprint,
                replay_result_byte_identical=(
                    False if accounting["replay_mismatch_count"] else replay_match
                ),
                runtime_accounting_complete=accounting["runtime_accounting_complete"],
                evaluation_provenance=provenance,
            )
            report_path = entry_output / "canary-report.json"
            _atomic_json(report_path, report)
            receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "entry_id": entry["id"],
                "source_sha256": entry["source"]["sha256"],
                "execution_fingerprint_sha256": fingerprint,
                "result_sha256": result_sha256,
                "quality_report_sha256": quality_sha256,
                "canary_report_sha256": _sha256(report_path),
                "origin": "LOCAL_TRANSCRIPTION",
                "initial_runtime_seconds": initial_runtime_seconds,
                "cumulative_runtime_seconds": accounting["cumulative_runtime_seconds"],
                "runtime_measurement": "append-only-monotonic-attempt-ledger-v1",
                "runtime_accounting_complete": accounting["runtime_accounting_complete"],
                "unterminated_attempt_indices": accounting["unterminated_attempt_indices"],
                "last_run_runtime_seconds": current_runtime_seconds,
                "last_attempt_index": attempt_index,
                "last_attempt_evaluation_only": evaluate_only,
                "checkpoint_replay": (
                    accounting["transcription_attempt_count"] > 1 and not evaluate_only
                ),
                "result_byte_identical_on_replay": replay_match,
                "replay_count": max(0, accounting["transcription_attempt_count"] - 1),
                "evaluation_count": accounting["evaluation_attempt_count"],
                "replay_mismatch_count": accounting["replay_mismatch_count"],
                "cloud_allowed": False,
                "giga_admission_allowed": False,
            }
            _atomic_json(receipt_path, receipt)
            _aggregate_summary(output, manifest, all_entries, fingerprint)
        except Exception as exc:
            if not terminal_written:
                _finish_attempt(
                    entry_output,
                    entry,
                    fingerprint,
                    attempt_index=attempt_index,
                    started_ns=started_ns,
                    evaluate_only=evaluate_only,
                    elapsed_seconds=time.monotonic() - started,
                    status="FAILED",
                    error=exc,
                )
            _aggregate_summary(output, manifest, all_entries, fingerprint)
            raise
    return _aggregate_summary(output, manifest, all_entries, fingerprint)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a hash-bound, fail-closed historical transcription canary"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lesson", action="append", dest="selected")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("whisperkit", "mlx", "faster-whisper"),
        default="whisperkit",
    )
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-url")
    parser.add_argument("--port", type=int, default=50060)
    parser.add_argument("--whisperkit-cli")
    parser.add_argument("--mlx-temperature", action="append", type=float)
    parser.add_argument(
        "--retry-backend",
        choices=("mlx", "faster-whisper"),
        help="Independent already-local backend for rejected-span adjudication.",
    )
    parser.add_argument(
        "--retry-model",
        help="Existing local retry model directory or cached Hugging Face identifier.",
    )
    parser.add_argument(
        "--retry-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--retry-compute-type", default="default")
    parser.add_argument("--retry-mlx-temperature", action="append", type=float)
    parser.add_argument("--language", help=argparse.SUPPRESS)
    args = parser.parse_args()
    backend = build_backend(args)
    retry_backend = build_retry_backend(args)
    silence_verification_detector = FasterWhisperSileroSpeechRegionDetector()
    try:
        summary = run_canary(
            args.manifest,
            args.output,
            backend,
            retry_backend=retry_backend,
            silence_verification_detector=silence_verification_detector,
            targeted_retry_region_detector=None,
            selected_ids=set(args.selected or ()),
            evaluate_only=args.evaluate_only,
        )
    except (OSError, ValueError, RuntimeError, TranscriptionError) as exc:
        raise SystemExit(f"could not complete historical canary: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
