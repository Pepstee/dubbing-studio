from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path



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
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_stdout_contains_start_ms(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt))
        # start_ms for the single entry is 1000
        assert "1000" in result.stdout

    def test_say_backend_flag_accepted(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt), "--backend", "say")
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

    def test_wav_file_created(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        assert (out_dir / "sample.wav").is_file()

    def test_wav_duration_matches_subtitle_timeline(self, tmp_path):
        import io as _io
        import wave as _wave

        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        data = (out_dir / "sample.wav").read_bytes()
        with _wave.open(_io.BytesIO(data)) as wf:
            duration_ms = int(wf.getnframes() * 1000 / wf.getframerate())
        # _SRT_SINGLE's only subtitle spans 1000–3000ms → timeline ends at 3000ms.
        assert abs(duration_ms - 3000) <= 50

    def test_json_includes_stretch_ratio(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        _run("dub", str(srt), "--output", str(out_dir))
        data = json.loads((out_dir / "sample.json").read_text())
        assert "stretch_ratio" in data[0]


# ---------------------------------------------------------------------------
# dub subcommand — --lang
# ---------------------------------------------------------------------------

class TestCliLang:
    def test_lang_flag_accepted_and_produces_wav(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        out_dir = tmp_path / "output"
        result = _run("dub", str(srt), "--lang", "en", "--output", str(out_dir))
        assert result.returncode == 0, result.stderr
        assert (out_dir / "sample.wav").is_file()

    def test_unsupported_lang_fails_with_clear_error(self, tmp_path):
        srt = tmp_path / "sample.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("dub", str(srt), "--lang", "zz-ZZ")
        assert result.returncode != 0
        assert "no installed 'say' voice" in result.stderr


# ---------------------------------------------------------------------------
# batch subcommand
# ---------------------------------------------------------------------------

class TestCliBatch:
    def test_exit_code_zero(self, tmp_path):
        srt = tmp_path / "a.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"), "--output", str(tmp_path / "out"))
        assert result.returncode == 0, result.stderr

    def test_stdout_is_non_empty(self, tmp_path):
        srt = tmp_path / "a.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"), "--output", str(tmp_path / "out"))
        assert result.stdout.strip() != ""

    def test_stdout_mentions_segment_count(self, tmp_path):
        srt = tmp_path / "a.srt"
        srt.write_text(_SRT_SINGLE, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"), "--output", str(tmp_path / "out"))
        assert "segment" in result.stdout.lower()

    def test_two_files_shows_two_lines(self, tmp_path):
        (tmp_path / "x.srt").write_text(_SRT_SINGLE, encoding="utf-8")
        (tmp_path / "y.srt").write_text(_SRT_MULTI, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"), "--output", str(tmp_path / "out"))
        assert result.returncode == 0, result.stderr
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_multi_segment_file_shows_correct_count(self, tmp_path):
        (tmp_path / "m.srt").write_text(_SRT_MULTI, encoding="utf-8")
        result = _run("batch", str(tmp_path / "*.srt"), "--output", str(tmp_path / "out"))
        assert "2 segment" in result.stdout

    def test_batch_writes_wav_and_json_per_input(self, tmp_path):
        (tmp_path / "x.srt").write_text(_SRT_SINGLE, encoding="utf-8")
        (tmp_path / "y.srt").write_text(_SRT_MULTI, encoding="utf-8")
        out_dir = tmp_path / "out"
        result = _run("batch", str(tmp_path / "*.srt"), "--output", str(out_dir))
        assert result.returncode == 0, result.stderr
        for stem in ("x", "y"):
            assert (out_dir / f"{stem}.wav").is_file(), f"{stem}.wav missing"
            assert (out_dir / f"{stem}.json").is_file(), f"{stem}.json missing"

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
