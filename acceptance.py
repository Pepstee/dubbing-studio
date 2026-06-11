#!/usr/bin/env python3
"""Acceptance test: dubbing-cli with say backend produces a non-empty WAV; web server returns HTTP 200."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent


def check_cli() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "-m", "dubbing", "dub",
                str(ROOT / "samples" / "sample.srt"),
                "--backend", "say",
                "--output", tmpdir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            sys.exit(f"CLI failed (exit {result.returncode}): {result.stderr}")

        wav_files = list(Path(tmpdir).glob("*.wav"))
        if not wav_files:
            sys.exit("FAIL: CLI produced no WAV file")

        wav = wav_files[0]
        size = wav.stat().st_size
        if size == 0:
            sys.exit(f"FAIL: {wav.name} is empty")

        print(f"CLI OK: {wav.name} ({size} bytes)")


def check_web() -> None:
    from dubbing.web import app

    thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=7432, use_reloader=False),
        daemon=True,
    )
    thread.start()
    time.sleep(1.5)

    try:
        with urllib.request.urlopen("http://127.0.0.1:7432/") as resp:
            status = resp.status
    except urllib.error.URLError as exc:
        sys.exit(f"FAIL: web server unreachable: {exc}")

    if status != 200:
        sys.exit(f"FAIL: GET / returned HTTP {status}")

    print(f"Web OK: GET / returned HTTP {status}")


if __name__ == "__main__":
    check_cli()
    check_web()
    print("All acceptance checks passed.")
