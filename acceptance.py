#!/usr/bin/env python3
"""Acceptance: CLI dubs the sample SRT into a timeline-true WAV (duration
matches the final subtitle's end time); the web UI serves the form and dubs
an uploaded SRT end-to-end, returning downloadable timeline-true audio."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

ROOT = Path(__file__).parent

# samples/sample.srt: last subtitle ends at 00:00:17,000.
_SAMPLE_TIMELINE_END_MS = 17_000
_TOLERANCE_MS = 100


def _wav_duration_ms(data: bytes) -> int:
    with wave.open(io.BytesIO(data)) as wf:
        return int(wf.getnframes() * 1000 / wf.getframerate())


def verify_wav_header(
    data: bytes,
    expected_end_ms: int,
    tolerance_ms: int = _TOLERANCE_MS,
) -> int:
    """Return WAV duration in ms; raise ValueError if data is empty or not timeline-true."""
    if not data:
        raise ValueError("WAV data is empty")
    duration = _wav_duration_ms(data)
    if abs(duration - expected_end_ms) > tolerance_ms:
        raise ValueError(
            f"WAV is {duration}ms long but timeline ends at {expected_end_ms}ms "
            f"— output is not timeline-true"
        )
    return duration


def verify_segments(plan: list, expected_count: int) -> None:
    """Raise ValueError if plan does not contain exactly expected_count segments."""
    if len(plan) != expected_count:
        raise ValueError(
            f"expected {expected_count} segments in plan, got {len(plan)}"
        )


def check_cli() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "-m", "dubbing", "dub",
                str(ROOT / "samples" / "sample.srt"),
                "--backend", "auto",
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
        data = wav.read_bytes()
        try:
            duration = verify_wav_header(data, _SAMPLE_TIMELINE_END_MS)
        except ValueError as exc:
            sys.exit(f"FAIL: {wav.name}: {exc}")

        json_files = list(Path(tmpdir).glob("*.json"))
        if not json_files:
            sys.exit("FAIL: CLI produced no JSON segment plan")
        plan = json.loads(json_files[0].read_text(encoding="utf-8"))
        try:
            verify_segments(plan, 5)
        except ValueError as exc:
            sys.exit(f"FAIL: {exc}")

        print(f"CLI OK: {wav.name} ({len(data)} bytes, {duration}ms ≈ timeline end)")


def check_web() -> None:
    from dubbing.apps.dubbing_web import app

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

    # Real end-to-end dub through the web UI: upload a one-line SRT,
    # download the audio, verify it spans the subtitle timeline (2s).
    srt_body = "1\n00:00:00,500 --> 00:00:02,000\nWeb acceptance check\n"
    boundary = uuid.uuid4().hex
    payload = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="srt"; filename="check.srt"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
        f"{srt_body}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:7432/dub",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        sys.exit(f"FAIL: POST /dub failed: {exc}")

    match = re.search(r'href="(/download/[^"]+)"', body)
    if match is None:
        sys.exit("FAIL: POST /dub returned no download link")
    download = match.group(1)

    with urllib.request.urlopen(f"http://127.0.0.1:7432{download}", timeout=30) as resp:
        audio = resp.read()
    try:
        duration = verify_wav_header(audio, 2000)
    except ValueError as exc:
        sys.exit(f"FAIL: web dub: {exc}")

    print(f"Web OK: POST /dub → {download} ({len(audio)} bytes, {duration}ms ≈ timeline end)")


if __name__ == "__main__":
    check_cli()
    check_web()
    print("All acceptance checks passed.")
