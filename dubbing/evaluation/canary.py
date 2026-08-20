from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dubbing.control_plane.cli import build_backend
from dubbing.evaluation.metrics import TimedText, evaluate_documents
from dubbing.transcription.adaptive import AdaptiveChunkPlanner, AdaptiveLongFormCoordinator
from dubbing.transcription.models import (
    TranscriptionError,
    TranscriptionOptions,
    transcription_result_from_dict,
)


SCHEMA_VERSION = "dubbing.historical-canary.v1"
REPORT_SCHEMA_VERSION = "dubbing.historical-canary-report.v1"
SUMMARY_SCHEMA_VERSION = "dubbing.historical-canary-summary.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_sha256(document: dict) -> str:
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def load_canary_manifest(path: str | Path) -> tuple[Path, dict]:
    manifest_path = Path(path).resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load canary manifest: {manifest_path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"canary manifest must use schema {SCHEMA_VERSION}")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("canary manifest requires at least one entry")
    ids = [str(item.get("id", "")) for item in entries if isinstance(item, dict)]
    if len(ids) != len(entries) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("canary entry IDs must be unique non-empty strings")
    roles = {str(item.get("role", "")) for item in entries}
    if not {"development", "holdout", "stress"}.issubset(roles):
        raise ValueError("canary manifest requires development, holdout and stress roles")
    if document.get("giga_admission_allowed") is not False:
        raise ValueError("historical canary must keep GIGA admission disabled")
    return manifest_path, document


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_canary_bindings(manifest_path: Path, manifest: dict) -> tuple[dict, ...]:
    root = manifest_path.parent
    bound = []
    for entry in manifest["entries"]:
        resolved = dict(entry)
        for key in ("source", "reference"):
            item = entry.get(key)
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ValueError(f"{entry['id']} {key} binding is incomplete")
            path = _resolve(root, item["path"])
            if not path.is_file():
                raise ValueError(f"{entry['id']} {key} is missing: {path}")
            actual = _sha256(path)
            if actual != item["sha256"]:
                raise ValueError(
                    f"{entry['id']} {key} SHA-256 mismatch: expected {item['sha256']}"
                )
            resolved[key] = {**item, "path": str(path)}
        bound.append(resolved)
    return tuple(bound)


def _execution_fingerprint(manifest: dict, backend) -> tuple[dict, str]:
    backend_evidence = {"identity": backend.identity}
    server = getattr(backend, "server", None)
    if server is not None:
        backend_evidence.update(
            {
                "model_tree_sha256": server.model_sha256,
                "model_size_bytes": server.model_size_bytes,
                "runtime_version": server.version,
            }
        )
    execution = {
        "schema_version": "dubbing.historical-canary-execution.v1",
        "backend": backend_evidence,
        "required_languages": manifest.get("required_languages", []),
        "execution": manifest.get("execution", {}),
        "policy": manifest.get("policy", {}),
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


def evaluate_canary_result(
    entry: dict,
    result_document: dict,
    quality: dict,
    *,
    runtime_seconds: float,
    policy: dict,
    execution_fingerprint: str,
) -> dict:
    result = transcription_result_from_dict(result_document)
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
    quality_status = str(quality.get("status", "FAILED"))
    allowed_quality_states = set(policy.get("allowed_quality_statuses", ["PASS"]))
    if quality_status not in allowed_quality_states:
        reasons.append(f"QUALITY_{quality_status}")
    quality_metrics = quality.get("metrics", {})
    if int(quality_metrics.get("repetition_finding_count", 0)):
        reasons.append("PATHOLOGICAL_REPETITION")
    fallback_exhausted_count = sum(
        bool(item.diagnostics and item.diagnostics.fallback_exhausted)
        for item in result.segments
    )
    if fallback_exhausted_count:
        reasons.append("DECODER_FALLBACK_EXHAUSTED")
    duration_seconds = max(0.001, int(entry["source"]["duration_ms"]) / 1000)
    realtime_factor = runtime_seconds / duration_seconds
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
        "runtime_seconds": runtime_seconds,
        "realtime_factor": realtime_factor,
        "decoder_fallback_exhausted_segment_count": fallback_exhausted_count,
        "quality": quality,
        "provisional_reference_metrics": metrics,
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
                "agreement is drift evidence, not ground-truth accuracy."
            ),
        },
        "diarization_claim": {
            "status": "NOT_MEASURED",
            "reason": "The historical references contain no trusted speaker labels.",
        },
        "giga_admission_allowed": False,
    }


def _summary(manifest: dict, reports: list[dict], execution_fingerprint: str) -> dict:
    failures = [
        report["entry_id"]
        for report in reports
        if not report["operational_gate"]["passed"]
    ]
    roles_run = sorted({report["role"] for report in reports})
    expected = {entry["id"] for entry in manifest["entries"]}
    completed = {report["entry_id"] for report in reports}
    missing = sorted(expected - completed)
    status = "FAIL" if failures else ("PARTIAL" if missing else "PASS")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "canary_id": manifest["canary_id"],
        "execution_fingerprint_sha256": execution_fingerprint,
        "status": status,
        "operational_failures": failures,
        "completed_entries": sorted(completed),
        "missing_entries": missing,
        "roles_run": roles_run,
        "accuracy_certified": False,
        "diarization_certified": False,
        "giga_admission_allowed": False,
    }


def run_canary(
    manifest_path: str | Path,
    output_dir: str | Path,
    backend,
    *,
    selected_ids: set[str] | None = None,
    evaluate_only: bool = False,
) -> dict:
    manifest_path, manifest = load_canary_manifest(manifest_path)
    if selected_ids:
        unknown = selected_ids - {entry["id"] for entry in manifest["entries"]}
        if unknown:
            raise ValueError(f"unknown canary entries: {', '.join(sorted(unknown))}")
        selected_manifest = {
            **manifest,
            "entries": [
                entry for entry in manifest["entries"] if entry["id"] in selected_ids
            ],
        }
        entries = validate_canary_bindings(manifest_path, selected_manifest)
    else:
        entries = validate_canary_bindings(manifest_path, manifest)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    execution, fingerprint = _execution_fingerprint(manifest, backend)
    _admit_frozen_execution(
        output / "frozen-execution.json",
        execution,
        fingerprint,
        roles={entry["role"] for entry in entries},
    )
    config = manifest.get("execution", {})
    planner = AdaptiveChunkPlanner(
        target_seconds=int(config.get("target_seconds", 240)),
        minimum_seconds=int(config.get("minimum_seconds", 60)),
        maximum_seconds=int(config.get("maximum_seconds", 480)),
        overlap_seconds=float(config.get("overlap_seconds", 2.0)),
    )
    reports = []
    for entry in entries:
        entry_output = output / entry["id"]
        result_path = entry_output / "result.json"
        quality_path = entry_output / "quality-report.json"
        receipt_path = entry_output / "canary-run-receipt.json"
        previous_result_sha256 = _sha256(result_path) if result_path.is_file() else None
        previous_receipt = (
            json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt_path.is_file()
            else None
        )
        started = time.monotonic()
        if not evaluate_only:
            coordinator = AdaptiveLongFormCoordinator(
                backend,
                entry_output,
                planner=planner,
                minimum_silence_seconds=float(config.get("minimum_silence_seconds", 0.7)),
                language_retry_policy=dict(config.get("language_retry_policy", {})),
            )
            _, quality = coordinator.run(
                Path(entry["source"]["path"]),
                TranscriptionOptions(task="transcribe", word_timestamps=True),
            )
        else:
            if not result_path.is_file() or not quality_path.is_file():
                raise ValueError(f"{entry['id']} has no completed result to evaluate")
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        current_runtime_seconds = round(time.monotonic() - started, 3)
        result_document = json.loads(result_path.read_text(encoding="utf-8"))
        result_sha256 = _sha256(result_path)
        initial_runtime_seconds = (
            float(previous_receipt["initial_runtime_seconds"])
            if previous_receipt is not None
            else current_runtime_seconds
        )
        report = evaluate_canary_result(
            entry,
            result_document,
            quality,
            runtime_seconds=initial_runtime_seconds,
            policy=manifest.get("policy", {}),
            execution_fingerprint=fingerprint,
        )
        receipt = {
            "schema_version": "dubbing.historical-canary-run-receipt.v1",
            "entry_id": entry["id"],
            "source_sha256": entry["source"]["sha256"],
            "result_sha256": result_sha256,
            "quality_report_sha256": _sha256(quality_path),
            "initial_runtime_seconds": initial_runtime_seconds,
            "last_run_runtime_seconds": current_runtime_seconds,
            "checkpoint_replay": previous_result_sha256 is not None and not evaluate_only,
            "result_byte_identical_on_replay": (
                previous_result_sha256 == result_sha256
                if previous_result_sha256 is not None and not evaluate_only
                else None
            ),
            "evaluation_only": evaluate_only,
            "replay_count": (
                int(previous_receipt.get("replay_count", 0)) + 1
                if previous_result_sha256 is not None and not evaluate_only
                else int(previous_receipt.get("replay_count", 0))
                if previous_receipt is not None
                else 0
            ),
            "cloud_allowed": False,
            "giga_admission_allowed": False,
        }
        _atomic_json(receipt_path, receipt)
        _atomic_json(entry_output / "canary-report.json", report)
        reports.append(report)
    summary = _summary(manifest, reports, fingerprint)
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a hash-bound, fail-closed historical transcription canary"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lesson", action="append", dest="selected")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--backend", choices=("whisperkit", "mlx", "faster-whisper"), default="whisperkit")
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
    parser.add_argument("--language", help=argparse.SUPPRESS)
    args = parser.parse_args()
    backend = build_backend(args)
    try:
        summary = run_canary(
            args.manifest,
            args.output,
            backend,
            selected_ids=set(args.selected or ()),
            evaluate_only=args.evaluate_only,
        )
    except (OSError, ValueError, RuntimeError, TranscriptionError) as exc:
        raise SystemExit(f"could not complete historical canary: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["status"] == "FAIL":
        sys.exit(2)


if __name__ == "__main__":
    main()
