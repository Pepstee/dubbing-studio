from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from dubbing.apps.personal_capture.config import SUPPORTED_MEDIA_SUFFIXES, load_config
from dubbing.transcription.job import SourceBinding, SourceStatIdentity, create_source_binding
from dubbing.transcription.models import TranscriptionError


_RECEIPT_SCHEMA = "dubbing.capture-ingest-receipt.v1"
_COPY_BLOCK_BYTES = 1024 * 1024


def _stat_identity(stat: os.stat_result) -> SourceStatIdentity:
    return SourceStatIdentity(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
    )


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.partial"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _copy_bound_source(binding: SourceBinding, partial: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_descriptor = os.open(binding.path, flags)
    destination_descriptor = -1
    try:
        if _stat_identity(os.fstat(source_descriptor)) != binding.stat:
            raise TranscriptionError("capture source changed before ingest copy")
        destination_descriptor = os.open(
            partial,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        digest = hashlib.sha256()
        while True:
            block = os.read(source_descriptor, _COPY_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("capture ingest copy made no forward progress")
                view = view[written:]
        os.fsync(destination_descriptor)
        if _stat_identity(os.fstat(source_descriptor)) != binding.stat:
            raise TranscriptionError("capture source changed during ingest copy")
        if digest.hexdigest() != binding.sha256:
            raise TranscriptionError("capture source digest changed during ingest copy")
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def _validate_existing_receipt(receipt: dict, binding: SourceBinding, inbox: Path) -> Path:
    if receipt.get("schema_version") != _RECEIPT_SCHEMA:
        raise TranscriptionError("capture ingest receipt has an unsupported schema")
    if receipt.get("capture_id") != binding.sha256:
        raise TranscriptionError("capture ingest receipt does not match source content")
    landing = receipt.get("landing")
    if not isinstance(landing, dict) or not isinstance(landing.get("name"), str):
        raise TranscriptionError("capture ingest receipt is malformed")
    destination = inbox / landing["name"]
    if destination.parent != inbox or not destination.is_file() or destination.is_symlink():
        raise TranscriptionError("capture ingest receipt destination is missing or unsafe")
    landed = create_source_binding(destination, expected_sha256=binding.sha256)
    if landing.get("sha256") != landed.sha256 or landing.get("size_bytes") != landed.stat.size_bytes:
        raise TranscriptionError("capture ingest receipt destination binding is invalid")
    return destination


def ingest_capture(source: str | Path, config: dict) -> dict:
    original = Path(source)
    if original.name.startswith(".") or original.suffix.casefold() not in SUPPORTED_MEDIA_SUFFIXES:
        raise TranscriptionError("capture source has an unsupported media filename")
    binding = create_source_binding(original)
    workspace = Path(config["workspace"]["wsl_path"])
    inbox = Path(config["landing_inbox"]["wsl_path"])
    if inbox.is_symlink() or not inbox.is_dir():
        raise TranscriptionError("capture inbox must be a prepared non-symlink directory")
    if inbox.resolve() != inbox or workspace.resolve() not in inbox.parents:
        raise TranscriptionError("capture inbox path is not canonical inside the workspace")
    state = workspace / config["paths"]["state"]
    receipt_path = state / "ingest" / f"{binding.sha256}.json"
    if receipt_path.is_symlink():
        raise TranscriptionError("capture ingest receipt must not be a symbolic link")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        destination = _validate_existing_receipt(receipt, binding, inbox)
        return {**receipt, "landing_path": str(destination), "replayed": True}

    destination = inbox / original.name
    recovered_existing = False
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise TranscriptionError("capture inbox destination is unsafe")
        landed = create_source_binding(destination)
        if landed.sha256 != binding.sha256:
            raise TranscriptionError("capture inbox destination already exists with different bytes")
        recovered_existing = True
    else:
        partial = inbox / f".{destination.name}.{secrets.token_hex(8)}.partial"
        try:
            _copy_bound_source(binding, partial)
            create_source_binding(partial, expected_sha256=binding.sha256)
            os.link(partial, destination, follow_symlinks=False)
            partial.unlink()
            directory_descriptor = os.open(inbox, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            partial.unlink(missing_ok=True)

    landed = create_source_binding(destination, expected_sha256=binding.sha256)
    receipt = {
        "schema_version": _RECEIPT_SCHEMA,
        "capture_id": binding.sha256,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": original.name,
            "sha256": binding.sha256,
            "size_bytes": binding.stat.size_bytes,
            "verified_stable": True,
        },
        "landing": {
            "name": destination.name,
            "sha256": landed.sha256,
            "size_bytes": landed.stat.size_bytes,
            "atomic_partial_publish": not recovered_existing,
            "publish_method": (
                "hard-link-no-overwrite" if not recovered_existing else "recovered-existing"
            ),
            "recovered_existing": recovered_existing,
        },
        "giga_admission_allowed": False,
    }
    _atomic_json(receipt_path, receipt)
    return {**receipt, "landing_path": str(destination), "replayed": recovered_existing}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash-verify and atomically land one untouched recorder file"
    )
    parser.add_argument("source")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        report = ingest_capture(args.source, load_config(args.config))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"capture ingest failed: {exc}") from exc
    except TranscriptionError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
