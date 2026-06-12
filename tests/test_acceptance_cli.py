"""Tests for acceptance.check_cli — subprocess fully mocked, no real synthesis.

check_cli() shells out to `python -m dubbing dub … --backend say` and then
verifies the produced WAV and JSON plan. These tests replace subprocess.run
with a fake that records the exact command and keyword arguments and writes
the expected artefacts into the --output directory, so every detail of the
invocation (sample path, capture_output, text, returncode handling) is
asserted rather than merely executed.
"""
from __future__ import annotations

import io
import json
import subprocess
import wave
from pathlib import Path

import pytest

import acceptance
from acceptance import check_cli


def _wav_bytes(duration_ms: int, rate: int = 1000) -> bytes:
    """Real 16-bit mono WAV bytes of exactly *duration_ms* ms (1 frame = 1 ms)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(rate * duration_ms / 1000))
    return buf.getvalue()


class _FakeRun:
    """Stand-in for subprocess.run: records the call, optionally writes artefacts."""

    def __init__(
        self,
        returncode: int = 0,
        write_wav_ms: int | None = 17_000,
        plan_segments: int = 5,
    ) -> None:
        self.returncode = returncode
        self.write_wav_ms = write_wav_ms
        self.plan_segments = plan_segments
        self.cmd: list | None = None
        self.kwargs: dict | None = None

    def __call__(self, cmd, **kwargs):
        self.cmd = list(cmd)
        self.kwargs = kwargs
        out_dir = Path(self.cmd[self.cmd.index("--output") + 1])
        if self.returncode == 0 and self.write_wav_ms is not None:
            (out_dir / "sample.wav").write_bytes(_wav_bytes(self.write_wav_ms))
            (out_dir / "sample.json").write_text(
                json.dumps([{"index": i} for i in range(self.plan_segments)]),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            cmd, self.returncode, stdout="", stderr="boom" if self.returncode else ""
        )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestCheckCliSuccess:
    def test_completes_without_exit_on_clean_run(self, monkeypatch, capsys):
        fake = _FakeRun()
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        check_cli()  # must not raise SystemExit
        assert "CLI OK" in capsys.readouterr().out

    def test_invokes_dub_on_the_sample_srt(self, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        check_cli()
        sample = str(acceptance.ROOT / "samples" / "sample.srt")
        assert fake.cmd is not None
        assert fake.cmd[1:4] == ["-m", "dubbing", "dub"]
        assert sample in fake.cmd
        assert "--backend" in fake.cmd
        assert fake.cmd[fake.cmd.index("--backend") + 1] == "say"

    def test_captures_output_as_text(self, monkeypatch):
        # stderr must be captured as str so the failure message is readable.
        fake = _FakeRun()
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        check_cli()
        assert fake.kwargs is not None
        assert fake.kwargs["capture_output"] is True
        assert fake.kwargs["text"] is True

    def test_runs_from_project_root_with_timeout(self, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        check_cli()
        assert fake.kwargs["cwd"] == str(acceptance.ROOT)
        assert fake.kwargs["timeout"] == 120

    def test_reports_duration_and_size(self, monkeypatch, capsys):
        fake = _FakeRun()
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        check_cli()
        out = capsys.readouterr().out
        assert "17000ms" in out
        assert "sample.wav" in out


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

class TestCheckCliFailures:
    def test_nonzero_returncode_exits_with_stderr(self, monkeypatch):
        fake = _FakeRun(returncode=3)
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        with pytest.raises(SystemExit) as exc_info:
            check_cli()
        assert "exit 3" in str(exc_info.value)
        assert "boom" in str(exc_info.value)

    def test_missing_wav_exits(self, monkeypatch):
        fake = _FakeRun(write_wav_ms=None)
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="no WAV"):
            check_cli()

    def test_non_timeline_true_wav_exits(self, monkeypatch):
        # 5s of audio against a 17s timeline → not timeline-true.
        fake = _FakeRun(write_wav_ms=5_000)
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="timeline"):
            check_cli()

    def test_wrong_segment_count_exits(self, monkeypatch):
        fake = _FakeRun(plan_segments=3)
        monkeypatch.setattr(acceptance.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="expected 5 segments"):
            check_cli()
