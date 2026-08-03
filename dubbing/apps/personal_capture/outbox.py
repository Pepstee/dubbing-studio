from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_EVENT_SCHEMA = "giga.personal-capture-event.v1"
_EVIDENCE_FIELDS = {
    "transcript_file": "transcript_sha256",
    "translation_file": "translation_sha256",
    "approval_file": "approval_sha256",
    "quality_report_file": "quality_report_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_path(root: Path, value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("outbox evidence names must be plain filenames")
    path = (root / value).resolve()
    if path.parent != root.resolve():
        raise ValueError("outbox evidence path escapes its bundle")
    return path


def _validated_event(
    root: Path,
    *,
    expected_capture_id: str | None = None,
) -> tuple[dict, dict[str, Path]]:
    event_path = root / "giga-event.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    if event.get("schema_version") != _EVENT_SCHEMA:
        raise ValueError("unsupported GIGA event schema")
    source = event.get("source")
    payload = event.get("payload")
    if not isinstance(source, dict) or not isinstance(payload, dict):
        raise ValueError("GIGA event source and payload must be objects")
    capture_id = source.get("capture_id")
    if (
        not isinstance(capture_id, str)
        or len(capture_id) != 64
        or capture_id != source.get("audio_sha256")
        or (expected_capture_id or root.name) != capture_id
    ):
        raise ValueError("GIGA event capture identity is inconsistent")
    evidence: dict[str, Path] = {}
    for payload_key, hash_key in _EVIDENCE_FIELDS.items():
        path = _evidence_path(root, payload.get(payload_key))
        expected_hash = source.get(hash_key)
        if path is None:
            if expected_hash is not None:
                raise ValueError(f"{hash_key} is present without {payload_key}")
            continue
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"{payload_key} is missing or does not match {hash_key}")
        evidence[path.name] = path
    if "transcript.json" not in evidence:
        raise ValueError("a transcript is required in every GIGA outbox bundle")
    return event, evidence


def verify_outbox_bundle(bundle: str | Path) -> bool:
    """Return whether an outbox directory is complete and hash-consistent."""
    try:
        _validated_event(Path(bundle).resolve())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _publish_bundle(package: Path, destination: Path, event_hash: str) -> Path:
    event, evidence = _validated_event(package)
    if event["event_id"] != f"personal-capture:{package.name}":
        raise RuntimeError("GIGA event ID does not match the capture")
    target = destination / package.name
    if target.exists():
        if not target.is_dir() or not verify_outbox_bundle(target):
            raise RuntimeError(f"existing outbox bundle is invalid: {target}")
        if _sha256(target / "giga-event.json") != event_hash:
            raise RuntimeError(f"outbox event changed for delivered capture: {package.name}")
        return target

    for abandoned in destination.glob(f".{package.name}.staging.*"):
        if abandoned.is_dir():
            shutil.rmtree(abandoned)
    staging = destination / f".{package.name}.staging.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    try:
        shutil.copyfile(package / "giga-event.json", staging / "giga-event.json")
        for name, source in evidence.items():
            shutil.copyfile(source, staging / name)
        for path in staging.iterdir():
            os.chmod(path, 0o600)
        try:
            _validated_event(staging, expected_capture_id=package.name)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            raise RuntimeError(f"could not construct valid outbox bundle for {package.name}")
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def export_approved(
    workspace: str | Path,
    outbox: str | Path,
    *,
    inbox: str | Path | None = None,
    packages_dir: str | Path = "outputs/packages",
) -> int:
    """Atomically publish self-contained, immutable evidence bundles."""
    workspace = Path(workspace).resolve()
    destination = Path(outbox).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    os.chmod(destination, 0o700)
    packages = (workspace / packages_dir).resolve()
    if packages != workspace and workspace not in packages.parents:
        raise ValueError(f"packages path escapes workspace: {packages_dir}")

    delivered = 0
    with sqlite3.connect(destination / "outbox.sqlite3") as ledger:
        ledger.execute(
            "CREATE TABLE IF NOT EXISTS delivered("
            "event_id TEXT PRIMARY KEY, event_sha256 TEXT NOT NULL, delivered_at TEXT NOT NULL)"
        )
        ledger.execute("BEGIN IMMEDIATE")
        for event_path in sorted(packages.glob("*/giga-event.json")):
            package = event_path.parent.resolve()
            try:
                event, _ = _validated_event(package)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            source = event["source"]
            if inbox is not None:
                root = Path(inbox).resolve()
                audio_name = source.get("audio_name")
                if not isinstance(audio_name, str) or Path(audio_name).name != audio_name:
                    continue
                audio = (root / audio_name).resolve()
                if (
                    audio.parent != root
                    or audio.is_symlink()
                    or not audio.is_file()
                    or _sha256(audio) != source.get("audio_sha256")
                ):
                    continue

            event_id = event.get("event_id")
            if not isinstance(event_id, str):
                continue
            event_hash = _sha256(event_path)
            existing = ledger.execute(
                "SELECT event_sha256 FROM delivered WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if existing is not None and existing[0] != event_hash:
                raise RuntimeError(f"event payload changed after delivery: {event_id}")
            _publish_bundle(package, destination, event_hash)
            if existing is None:
                ledger.execute(
                    "INSERT INTO delivered VALUES(?,?,?)",
                    (event_id, event_hash, datetime.now(timezone.utc).isoformat()),
                )
                delivered += 1
    return delivered
