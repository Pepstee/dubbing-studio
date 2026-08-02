"""Adversarial tests for prosody rate/pitch clamping in the say backend.

Each test inspects the *actual argument delivered to the mocked say process*
(the -r flag value and the [[ pbas N ]] embedded speech command on stdin),
not internal state.  This catches both missing clamping and serialisation
bugs independently of implementation details.
"""
from __future__ import annotations

import io
import subprocess
import wave
from pathlib import Path

import pytest

from dubbing.backends.say import _synthesize_with_say


# ---------------------------------------------------------------------------
# Shared fixture — captures every subprocess.run call; produces valid WAV
# output for afconvert so the backend can read it back without real macOS tools.
# ---------------------------------------------------------------------------

def _stub_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * 220)
    return buf.getvalue()


@pytest.fixture()
def captured_say(monkeypatch):
    """Mock shutil.which + subprocess.run; return captured call list."""
    calls: list[tuple[list[str], bytes | None]] = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs.get("input")))
        if cmd[0] == "afconvert":
            # The last arg is the output wav path — write a real WAV there.
            Path(cmd[-1]).write_bytes(_stub_wav_bytes())
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("dubbing.backends.say.subprocess.run", fake_run)
    return calls


def _say_cmd(calls: list) -> list[str]:
    return calls[0][0]


def _say_stdin(calls: list) -> str:
    raw = calls[0][1]
    return raw.decode("utf-8") if raw is not None else ""


# ---------------------------------------------------------------------------
# Rate clamping — assert -r value delivered to say process
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate_in, expected", [
    (0,        20),   # zero → floored to minimum
    (-1,       20),   # negative → floored to minimum
    (-9999,    20),   # large negative → floored to minimum
    (19,       20),   # just below minimum → floored
    (20,       20),   # exactly minimum → passes through unchanged
    (21,       21),   # one above minimum → passes through unchanged
    (175,     175),   # mid-range → passes through unchanged
    (499,     499),   # just below maximum → passes through unchanged
    (500,     500),   # exactly maximum → passes through unchanged
    (501,     500),   # just above maximum → capped
    (999999,  500),   # extreme → capped at maximum
])
def test_rate_clamping_delivered_to_say(rate_in, expected, captured_say):
    _synthesize_with_say("hello", rate_wpm=rate_in)
    cmd = _say_cmd(captured_say)
    assert "-r" in cmd, f"rate_wpm={rate_in}: expected -r flag in say command"
    delivered = int(cmd[cmd.index("-r") + 1])
    assert delivered == expected, (
        f"rate_wpm={rate_in}: expected say to receive -r {expected}, got -r {delivered}"
    )


# ---------------------------------------------------------------------------
# Pitch clamping — assert [[ pbas N ]] value in say's stdin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pitch_in, expected", [
    (-999999, -100),  # extreme negative → floored to minimum
    (-500,    -100),  # large negative → floored to minimum
    (-101,    -100),  # just below minimum → floored
    (-100,    -100),  # exactly minimum → passes through unchanged
    (-99,      -99),  # one above minimum → passes through unchanged
    (-50,      -50),  # mid-range negative → passes through unchanged
    (0,          0),  # neutral → passes through unchanged
    (50,        50),  # mid-range positive → passes through unchanged
    (99,        99),  # just below maximum → passes through unchanged
    (100,      100),  # exactly maximum → passes through unchanged
    (101,      100),  # just above maximum → capped
    (999999,   100),  # extreme positive → capped at maximum
])
def test_pitch_clamping_delivered_to_say(pitch_in, expected, captured_say):
    _synthesize_with_say("hello", pitch_pbas=pitch_in)
    stdin = _say_stdin(captured_say)
    assert f"[[ pbas {expected} ]]" in stdin, (
        f"pitch_pbas={pitch_in}: expected '[[ pbas {expected} ]]' in stdin, got: {stdin!r}"
    )


# ---------------------------------------------------------------------------
# None passthrough — omitting rate/pitch must not emit the flag at all
# ---------------------------------------------------------------------------

def test_rate_none_emits_no_r_flag(captured_say):
    _synthesize_with_say("hello", rate_wpm=None)
    assert "-r" not in _say_cmd(captured_say)


def test_pitch_none_emits_no_pbas(captured_say):
    _synthesize_with_say("hello", pitch_pbas=None)
    assert "[[ pbas" not in _say_stdin(captured_say)


# ---------------------------------------------------------------------------
# Boundary values — exact min/max must still emit the flag (regression guard:
# clamping must not accidentally suppress the flag when the value == bound)
# ---------------------------------------------------------------------------

def test_rate_min_boundary_flag_present(captured_say):
    _synthesize_with_say("hello", rate_wpm=20)
    cmd = _say_cmd(captured_say)
    assert "-r" in cmd
    assert cmd[cmd.index("-r") + 1] == "20"


def test_rate_max_boundary_flag_present(captured_say):
    _synthesize_with_say("hello", rate_wpm=500)
    cmd = _say_cmd(captured_say)
    assert "-r" in cmd
    assert cmd[cmd.index("-r") + 1] == "500"


def test_pitch_min_boundary_embedded_present(captured_say):
    _synthesize_with_say("hello", pitch_pbas=-100)
    assert "[[ pbas -100 ]]" in _say_stdin(captured_say)


def test_pitch_max_boundary_embedded_present(captured_say):
    _synthesize_with_say("hello", pitch_pbas=100)
    assert "[[ pbas 100 ]]" in _say_stdin(captured_say)


# ---------------------------------------------------------------------------
# Combined rate + pitch — both clamping paths run independently in one call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rate_in, pitch_in, rate_out, pitch_out", [
    (0,       -500,    20,   -100),  # both extremes, both clamped
    (999999,   999,   500,    100),  # both extremes in opposite direction
    (20,      -100,    20,   -100),  # both exactly at min boundary
    (500,      100,   500,    100),  # both exactly at max boundary
    (250,       50,   250,     50),  # both mid-range, no clamping
    (-1,       101,    20,    100),  # rate below min, pitch above max
    (501,     -101,   500,   -100),  # rate above max, pitch below min
])
def test_rate_and_pitch_clamped_independently(
    rate_in, pitch_in, rate_out, pitch_out, captured_say
):
    _synthesize_with_say("hello", rate_wpm=rate_in, pitch_pbas=pitch_in)
    cmd = _say_cmd(captured_say)
    stdin = _say_stdin(captured_say)

    assert "-r" in cmd, f"rate_wpm={rate_in}: expected -r flag"
    delivered_rate = int(cmd[cmd.index("-r") + 1])
    assert delivered_rate == rate_out, (
        f"rate_wpm={rate_in}: expected -r {rate_out}, got -r {delivered_rate}"
    )
    assert f"[[ pbas {pitch_out} ]]" in stdin, (
        f"pitch_pbas={pitch_in}: expected '[[ pbas {pitch_out} ]]', got stdin: {stdin!r}"
    )
