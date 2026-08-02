from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from dubbing.apps.personal_capture.config import load_config
from dubbing.apps.personal_capture.runtime import build_service


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def watch(config_path: str | Path, *, once: bool = False) -> int:
    config = load_config(config_path)
    workspace = Path(config["workspace"]["wsl_path"])
    workspace.mkdir(parents=True, exist_ok=True)
    os.chmod(workspace, 0o700)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(workspace / "capture.log"), logging.StreamHandler()],
    )
    lock = (workspace / "watcher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logging.error("another capture watcher owns the lock")
        return 3
    lock.write(str(os.getpid()))
    lock.flush()
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    service = build_service(config)
    recovered = service.store.requeue_interrupted()
    if recovered:
        logging.warning("requeued %s capture(s) interrupted by the prior watcher", recovered)
    inbox = config["landing_inbox"]["wsl_path"]
    min_age = float(config["landing_inbox"].get("minimum_file_age_seconds", 60))
    poll = min(300.0, max(2.0, float(config.get("service", {}).get("poll_seconds", 15))))
    while not stopping:
        started = datetime.now(timezone.utc).isoformat()
        error = None
        outcomes = []
        try:
            outcomes = list(service.scan(inbox, min_age_seconds=min_age))
            for outcome in outcomes:
                if not outcome.replayed:
                    logging.info(
                        "%s %s %s",
                        outcome.state,
                        outcome.capture_id,
                        outcome.source_name,
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2000]
            logging.exception("capture scan failed")
        _atomic_json(
            workspace / "health.json",
            {
                "schema_version": "dubbing.capture-health.v1",
                "pid": os.getpid(),
                "last_scan_started_at": started,
                "last_scan_finished_at": datetime.now(timezone.utc).isoformat(),
                "outcomes": len(outcomes),
                "error": error,
                "stopping": stopping,
            },
        )
        if once:
            return 1 if error else 0
        deadline = time.monotonic() + poll
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(1, deadline - time.monotonic()))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously process stable capture files")
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    raise SystemExit(watch(args.config, once=args.once))


if __name__ == "__main__":
    main()
