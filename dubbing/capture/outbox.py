from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_approved(
    workspace: str | Path, outbox: str | Path, *, inbox: str | Path | None = None
) -> int:
    workspace = Path(workspace).resolve()
    destination = Path(outbox).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    ledger = sqlite3.connect(destination / "outbox.sqlite3")
    ledger.execute(
        "CREATE TABLE IF NOT EXISTS delivered(event_id TEXT PRIMARY KEY, event_sha256 TEXT NOT NULL,"
        " delivered_at TEXT NOT NULL)"
    )
    delivered = 0
    for event_path in sorted((workspace / "packages").glob("*/giga-event.json")):
        package = event_path.parent
        event = json.loads(event_path.read_text(encoding="utf-8"))
        source = event.get("source", {})
        transcript = package / event.get("payload", {}).get("transcript_file", "")
        if event.get("schema_version") != "giga.personal-capture-event.v1":
            continue
        if not transcript.is_file() or _sha256(transcript) != source.get("transcript_sha256"):
            continue
        if package.name != source.get("capture_id") or package.name != source.get("audio_sha256"):
            continue
        if inbox is not None:
            root = Path(inbox).resolve()
            audio = (root / source.get("audio_name", "")).resolve()
            if (
                audio.parent != root
                or audio.is_symlink()
                or not audio.is_file()
                or _sha256(audio) != source.get("audio_sha256")
            ):
                continue
        event_id = event["event_id"]
        event_hash = hashlib.sha256(event_path.read_bytes()).hexdigest()
        try:
            with ledger:
                ledger.execute(
                    "INSERT INTO delivered VALUES(?,?,?)",
                    (event_id, event_hash, datetime.now(timezone.utc).isoformat()),
                )
                target = destination / f"{package.name}.json"
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(event_path.read_bytes())
                os.replace(temporary, target)
            delivered += 1
        except sqlite3.IntegrityError:
            continue
    ledger.close()
    return delivered
