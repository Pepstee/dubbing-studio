from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

_SRT_SINGLE = """\
1
00:00:01,000 --> 00:00:03,000
Hello world

"""

_SRT_MULTI = """\
1
00:00:00,000 --> 00:00:02,000
First line

2
00:00:03,000 --> 00:00:05,000
Second line

"""


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "dubbing", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd or _PROJECT_ROOT),
    )


# ---------------------------------------------------------------------------
# dub subcommand — happy path
# ---------------------------------------------------------------------------

class TestCliDub:
    def test_exit_code_zero(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt))
        assert result.returncode == 0, result.stderr

    def test_stdout_is_non_empty(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt))
        assert result.stdout.strip() != ""

    def test_stdout_contains_segment_text(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt))
        assert "Hello world" in result.stdout

    def test_stdout_contains_timing_brackets(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt))
        assert "[" in result.stdout and "]" in result.stdout

    def test_stdout_one_line_per_segment(self, tmp_path):
        srt = tmp_path / "multi.srt"
        srt.write_text(_SRT_MULTI, encoding="utf-8")
        result = _run("dub", str(srt))
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        assert len(lines) == 2

    def test_stdout_contains_start_ms(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt))
        # start_ms for the single entry is 1000
        assert "1000" in result.stdout

    def test_mock_backend_flag_accepted(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt), "--backend", "mock")
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# dub subcommand — with --output
# ---------------------------------------------------------------------------

class TestCliDubOutput:
    def test_exit_code_zero_with_output(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        result = _run("dub", str(srt), "--output", str(out_dir))
        assert result.returncode == 0, result.stderr

    def test_stdout_mentions_wrote(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        result = _run("dub", str(srt), "--output", str(out_dir))
        assert "Wrote" in result.stdout or "wrote" in result.stdout.lower()

    def test_json_file_created(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        json_files = list(out_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_json_file_is_valid_json(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        json_file = next(out_dir.glob("*.json"))
        data = json.loads(json_file.read_text())
        assert isinstance(data, list)

    def test_json_contains_expected_keys(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        json_file = next(out_dir.glob("*.json"))
        data = json.loads(json_file.read_text())
        assert len(data) == 1
        entry = data[0]
        assert "start_ms" in entry
        assert "end_ms" in entry
        assert "text" in entry

    def test_json_start_ms_is_correct(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        json_file = next(out_dir.glob("*.json"))
        data = json.loads(json_file.read_text())
        assert data[0]["start_ms"] == 1000

    def test_json_end_ms_is_correct(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        json_file = next(out_dir.glob("*.json"))
        data = json.loads(json_file.read_text())
        assert data[0]["end_ms"] == 3000

    def test_json_stem_matches_srt_stem(self, tmp_path):
        srt = tmp_path / "mysubs.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        assert (out_dir / "mysubs.json").exists()


# ---------------------------------------------------------------------------
# batch subcommand
# ---------------------------------------------------------------------------

class TestCliBatch:
    def test_exit_code_zero(self, tmp_path):
        srt = tmp_path / "a.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"))
        assert result.returncode == 0, result.stderr

    def test_stdout_is_non_empty(self, tmp_path):
        srt = tmp_path / "a.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"))
        assert result.stdout.strip() != ""

    def test_stdout_mentions_segment_count(self, tmp_path):
        srt = tmp_path / "a.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"))
        assert "segment" in result.stdout.lower()

    def test_two_files_shows_two_lines(self, tmp_path):
        (tmp_path / "x.srt").write_text(_SRT_SINGLE, encoding="utf-8")
        (tmp_path / "y.srt").write_text(_SRT_MULTI, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"))
        assert result.returncode == 0, result.stderr
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        assert len(lines) == 2

    def test_multi_segment_file_shows_correct_count(self, tmp_path):
        (tmp_path / "m.srt").write_text(_SRT_MULTI, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"))
        assert "2 segment" in result.stdout

    def test_no_matching_glob_exits_nonzero(self, tmp_path):
        result = _run("batch", str(tmp_path / "*.srt"))
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestCliErrors:
    def test_unknown_backend_exits_nonzero(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt), "--backend", "nonexistent_backend")
        assert result.returncode != 0

    def test_no_subcommand_exits_nonzero(self):
        result = _run()
        assert result.returncode != 0

    def test_missing_srt_argument_exits_nonzero(self):
        result = _run("dub")
        assert result.returncode != 0

    def test_nonexistent_srt_file_exits_nonzero(self, tmp_path):
        result = _run("dub", str(tmp_path / "does_not_exist.srt"))
        assert result.returncode != 0
