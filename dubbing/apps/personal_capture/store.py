from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CaptureRecord:
    capture_id: str
    source_name: str
    source_sha256: str
    source_size: int
    source_mtime_ns: int
    state: str
    attempt_count: int
    package_path: str | None
    error: str | None
    created_at: str = ""
    updated_at: str = ""


class CaptureStore:
    """Small durable ledger for idempotent personal-capture processing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS captures (
                    capture_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL UNIQUE,
                    source_size INTEGER NOT NULL CHECK(source_size >= 0),
                    source_mtime_ns INTEGER NOT NULL CHECK(source_mtime_ns >= 0),
                    state TEXT NOT NULL CHECK(
                        state IN (
                            'queued', 'processing', 'review', 'approved', 'failed', 'rejected'
                        )
                    ),
                    attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                    package_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='captures'"
            ).fetchone()[0]
            if "'queued'" not in sql:
                connection.executescript(
                    """
                    ALTER TABLE captures RENAME TO captures_legacy;
                    CREATE TABLE captures (
                        capture_id TEXT PRIMARY KEY,
                        source_name TEXT NOT NULL,
                        source_sha256 TEXT NOT NULL UNIQUE,
                        source_size INTEGER NOT NULL CHECK(source_size >= 0),
                        source_mtime_ns INTEGER NOT NULL CHECK(source_mtime_ns >= 0),
                        state TEXT NOT NULL CHECK(
                            state IN (
                                'queued','processing','review','approved','failed','rejected'
                            )
                        ),
                        attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                        package_path TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    INSERT INTO captures SELECT * FROM captures_legacy;
                    DROP TABLE captures_legacy;
                    """
                )

    @staticmethod
    def _record(row: sqlite3.Row | None) -> CaptureRecord | None:
        if row is None:
            return None
        return CaptureRecord(
            capture_id=row["capture_id"],
            source_name=row["source_name"],
            source_sha256=row["source_sha256"],
            source_size=row["source_size"],
            source_mtime_ns=row["source_mtime_ns"],
            state=row["state"],
            attempt_count=row["attempt_count"],
            package_path=row["package_path"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, capture_id: str) -> CaptureRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM captures WHERE capture_id = ?",
                (capture_id,),
            ).fetchone()
        return self._record(row)

    def find_by_source_snapshot(
        self,
        source_name: str,
        source_size: int,
        source_mtime_ns: int,
    ) -> CaptureRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM captures
                WHERE source_name = ? AND source_size = ? AND source_mtime_ns = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (source_name, source_size, source_mtime_ns),
            ).fetchone()
        return self._record(row)

    def claim(
        self,
        *,
        capture_id: str,
        source_name: str,
        source_sha256: str,
        source_size: int,
        source_mtime_ns: int,
    ) -> tuple[CaptureRecord, bool]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM captures WHERE source_sha256 = ?",
                (source_sha256,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO captures(
                        capture_id, source_name, source_sha256, source_size,
                        source_mtime_ns, state, attempt_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'processing', 1, ?, ?)
                    """,
                    (
                        capture_id,
                        source_name,
                        source_sha256,
                        source_size,
                        source_mtime_ns,
                        now,
                        now,
                    ),
                )
                claimed = True
            elif row["state"] == "queued":
                connection.execute(
                    """
                    UPDATE captures
                    SET state = 'processing', attempt_count = attempt_count + 1,
                        error = NULL, updated_at = ?
                    WHERE capture_id = ?
                    """,
                    (now, row["capture_id"]),
                )
                capture_id = row["capture_id"]
                claimed = True
            else:
                capture_id = row["capture_id"]
                claimed = False
            updated = connection.execute(
                "SELECT * FROM captures WHERE capture_id = ?",
                (capture_id,),
            ).fetchone()
        record = self._record(updated)
        assert record is not None
        return record, claimed

    def transition(
        self,
        capture_id: str,
        state: str,
        *,
        package_path: str | None = None,
        error: str | None = None,
    ) -> CaptureRecord:
        if state not in {"review", "approved", "failed", "rejected"}:
            raise ValueError(f"invalid capture state: {state}")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE captures
                SET state = ?, package_path = COALESCE(?, package_path),
                    error = ?, updated_at = ?
                WHERE capture_id = ?
                """,
                (state, package_path, error, now, capture_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown capture: {capture_id}")
            row = connection.execute(
                "SELECT * FROM captures WHERE capture_id = ?",
                (capture_id,),
            ).fetchone()
        record = self._record(row)
        assert record is not None
        return record

    def retry(self, capture_id: str) -> CaptureRecord:
        record = self.get(capture_id)
        if record is None:
            raise KeyError(f"unknown capture: {capture_id}")
        if record.state not in {"failed", "rejected"}:
            raise ValueError(f"capture cannot be retried from state {record.state}")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "UPDATE captures SET state = 'queued', error = NULL, updated_at = ? "
                "WHERE capture_id = ?",
                (now, capture_id),
            )
        result = self.get(capture_id)
        assert result is not None
        return result

    def requeue_interrupted(self) -> int:
        """Requeue work abandoned by a previous watcher process.

        The watcher calls this only after acquiring its process-wide filesystem
        lock, so no earlier watcher can still own these rows.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE captures
                SET state = 'queued',
                    error = 'previous watcher stopped during processing',
                    updated_at = ?
                WHERE state = 'processing'
                """,
                (now,),
            )
        return cursor.rowcount
