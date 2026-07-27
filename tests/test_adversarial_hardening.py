"""Adversarial round-4 regressions: hostile-input defects that previously
crashed or exhausted the process.

1. A tiny SRT carrying a huge timestamp (e.g. 99:59:59,999) used to force a
   ~16 GB silence allocation in `assemble_timeline` — memory-exhaustion DoS
   reachable through the 1 MiB-capped web upload (the cap bounds file size,
   not timestamp values). Now rejected with ValueError before any allocation.
2. A prosody tag value like "²" (superscript two) passes `str.isdigit()` but
   crashes `int()` — `<rate:²>` used to raise ValueError out of the backend.
   Now safely ignored.
3. `subprocess.TimeoutExpired` is NOT a `CalledProcessError`; a hung `say`
   used to escape the web handler as an unhandled 500. Now a 502.
"""

from __future__ import annotations

import io
import json
import subprocess
import wave

import pytest

from dubbing.aligner import TimedSegment
from dubbing.assembler import MAX_TIMELINE_MS, assemble_timeline
from dubbing.backends.say import _parse_int, _pbas_for_tags, _rate_for_tags
from dubbing.models import ProsodyTag, Segment, SRTEntry, TTSResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_wav(frames: int = 220) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00" * (2 * frames))
    return buf.getvalue()


def _pair(start_ms: int, end_ms: int) -> tuple[list[TimedSegment], list[TTSResult]]:
    entry = SRTEntry(index=1, start_ms=start_ms, end_ms=end_ms, text="hi")
    seg = Segment(entry=entry, tags=[], language="")
    timed = [TimedSegment(start_ms=start_ms, end_ms=end_ms, segment=seg)]
    results = [TTSResult(segment=seg, audio_bytes=_short_wav(), duration_ms=10)]
    return timed, results


# ---------------------------------------------------------------------------
# 1. Timeline-duration cap (hostile-timestamp memory DoS)
# ---------------------------------------------------------------------------

class TestTimelineCap:
    def test_99_hour_timestamp_rejected_without_allocation(self):
        # 99:59:59,999 → 359,999,999 ms → ~16 GB of silence pre-fix.
        timed, results = _pair(0, 359_999_999)
        with pytest.raises(ValueError, match="maximum supported timeline"):
            assemble_timeline(timed, results)

    def test_start_beyond_cap_rejected_even_with_degenerate_window(self):
        # end_ms <= start_ms (degenerate window) must not bypass the cap.
        timed, results = _pair(MAX_TIMELINE_MS + 1, 0)
        with pytest.raises(ValueError, match="maximum supported timeline"):
            assemble_timeline(timed, results)

    def test_one_ms_beyond_custom_cap_rejected(self):
        timed, results = _pair(0, 1_001)
        with pytest.raises(ValueError, match="maximum supported timeline"):
            assemble_timeline(timed, results, max_timeline_ms=1_000)

    def test_exactly_at_custom_cap_accepted(self):
        timed, results = _pair(0, 1_000)
        wav = assemble_timeline(timed, results, max_timeline_ms=1_000)
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnframes() == 22050  # exactly 1 s at 22050 Hz

    def test_error_names_offending_segment(self):
        timed, results = _pair(0, 359_999_999)
        with pytest.raises(ValueError, match="segment 1"):
            assemble_timeline(timed, results)

    def test_negative_start_rejected(self):
        timed, results = _pair(-1, 1_000)
        with pytest.raises(ValueError, match="negative start"):
            assemble_timeline(timed, results)

    def test_default_cap_is_four_hours(self):
        assert MAX_TIMELINE_MS == 4 * 60 * 60 * 1000


# ---------------------------------------------------------------------------
# 2. Unicode-digit prosody values (isdigit-passes-int-crashes)
# ---------------------------------------------------------------------------

class TestUnicodeDigitProsody:
    def test_parse_int_rejects_superscript_two(self):
        assert "²".isdigit()  # the trap: isdigit is True...
        assert _parse_int("²") is None  # ...but it is not an int

    def test_parse_int_accepts_plain_digits(self):
        assert _parse_int("175") == 175

    def test_superscript_rate_tag_ignored_not_crash(self):
        assert _rate_for_tags([ProsodyTag(name="rate", value="²")]) is None

    def test_superscript_pitch_tag_ignored_not_crash(self):
        assert _pbas_for_tags([ProsodyTag(name="pitch", value="²")]) is None

    def test_numeric_rate_tag_still_parses(self):
        assert _rate_for_tags([ProsodyTag(name="rate", value="200")]) == 200

    def test_numeric_pitch_tag_still_parses(self):
        assert _pbas_for_tags([ProsodyTag(name="pitch", value="50")]) == 50

    def test_named_rate_value_still_wins(self):
        assert _rate_for_tags([ProsodyTag(name="rate", value="fast")]) == 220

    def test_bad_value_does_not_clobber_earlier_good_value(self):
        tags = [ProsodyTag(name="rate", value="200"), ProsodyTag(name="rate", value="²")]
        assert _rate_for_tags(tags) == 200


# ---------------------------------------------------------------------------
# 3. Web: TimeoutExpired surfaces as 502, hostile SRT as clean error
# ---------------------------------------------------------------------------

flask = pytest.importorskip("flask")

from dubbing.backends.base import TTSBackend  # noqa: E402
from dubbing.web import _jobs, app  # noqa: E402

_SRT_HOSTILE_TIMESTAMP = """\
1
00:00:00,000 --> 99:59:59,999
sixteen gigabytes please

"""


class _TimingOutBackend(TTSBackend):
    def synthesize(self, segments):
        raise subprocess.TimeoutExpired(cmd=["say"], timeout=30)


class _ShortWavBackend(TTSBackend):
    def synthesize(self, segments):
        return [
            TTSResult(segment=seg, audio_bytes=_short_wav(), duration_ms=10)
            for seg in segments
        ]


@pytest.fixture()
def client():
    _jobs.clear()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    _jobs.clear()


def _post_srt(client, srt: str):
    return client.post(
        "/dub",
        data={"srt": (io.BytesIO(srt.encode()), "test.srt")},
        content_type="multipart/form-data",
    )


class TestWebHardening:
    def test_tts_timeout_returns_502_not_500(self, client, monkeypatch):
        monkeypatch.setattr(
            "dubbing.backends.select_backend",
            lambda: _TimingOutBackend(),
        )
        resp = _post_srt(client, "1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        assert resp.status_code == 502
        body = json.loads(resp.data)
        assert "error" in body
        assert _jobs == {}

    def test_hostile_timestamp_returns_error_not_oom(self, client, monkeypatch):
        monkeypatch.setattr("dubbing.backends.say.SayTTSBackend", _ShortWavBackend)
        resp = _post_srt(client, _SRT_HOSTILE_TIMESTAMP)
        assert resp.status_code == 502
        body = json.loads(resp.data)
        assert "error" in body and "timeline" in body["error"]
        assert _jobs == {}


# ---------------------------------------------------------------------------
# 4. CLI: hostile SRT exits with a clean error, never a traceback
# ---------------------------------------------------------------------------

class TestCliHardening:
    def test_hostile_timestamp_yields_clean_error(self, tmp_path):
        import sys
        from pathlib import Path

        srt = tmp_path / "hostile.srt"
        srt.write_text(_SRT_HOSTILE_TIMESTAMP, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "dubbing", "dub", str(srt), "--output", str(tmp_path / "out")],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0
        assert "error:" in result.stderr
        assert "Traceback" not in result.stderr
