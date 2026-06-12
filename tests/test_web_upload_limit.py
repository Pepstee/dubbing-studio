"""Adversarial tests for the web upload size-limit gate (MAX_UPLOAD_BYTES).

Separate from test_web.py so the size-limit contract can be reviewed and
extended independently. Every test in this file exercises the REAL code path;
TTS synthesis is replaced by _FixedBackend to avoid subprocess calls.
"""

from __future__ import annotations

import io
import json
import wave

import pytest

flask = pytest.importorskip("flask")

from dubbing.backends.base import TTSBackend  # noqa: E402
from dubbing.models import Segment, TTSResult  # noqa: E402
from dubbing.web import MAX_UPLOAD_BYTES, _jobs, app  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers and test doubles
# ---------------------------------------------------------------------------


def _minimal_wav() -> bytes:
    """Single-frame silence WAV — valid but trivially small, no subprocess."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00" * 2)
    return buf.getvalue()


class _FixedBackend(TTSBackend):
    """Returns pre-built silence WAV; zero subprocess calls."""

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        wav = _minimal_wav()
        return [TTSResult(segment=seg, audio_bytes=wav, duration_ms=100) for seg in segments]


def _make_valid_srt_of_size(size: int) -> bytes:
    """Return a syntactically valid SRT whose UTF-8 encoding is exactly `size` bytes.

    The subtitle text line is padded with ASCII 'A' characters to hit the
    target.  The SRT parser, prosody scanner, and _FixedBackend all accept
    arbitrarily long text, so this round-trips through the full pipeline.
    """
    header = b"1\n00:00:01,000 --> 00:00:03,000\n"
    footer = b"\n\n"
    fill = size - len(header) - len(footer)
    if fill < 0:
        raise ValueError(f"Cannot build a valid SRT in {size} bytes (minimum {len(header) + len(footer)})")
    return header + b"A" * fill + footer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with real synthesis replaced by _FixedBackend."""
    monkeypatch.setattr("dubbing.backends.say.SayTTSBackend", _FixedBackend)
    _jobs.clear()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    _jobs.clear()


def _post(client, data: bytes, filename: str = "test.srt"):
    return client.post(
        "/dub",
        data={"srt": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


# ---------------------------------------------------------------------------
# Boundary: exact limit and one-byte over
# ---------------------------------------------------------------------------


class TestExactBoundary:
    """Fine-grained boundary checks around MAX_UPLOAD_BYTES."""

    def test_one_byte_over_returns_413(self, client):
        """MAX_UPLOAD_BYTES + 1 must be rejected."""
        payload = b"x" * (MAX_UPLOAD_BYTES + 1)
        assert _post(client, payload).status_code == 413

    def test_valid_srt_exactly_at_limit_returns_200(self, client):
        """A well-formed SRT whose size equals MAX_UPLOAD_BYTES must succeed."""
        payload = _make_valid_srt_of_size(MAX_UPLOAD_BYTES)
        assert len(payload) == MAX_UPLOAD_BYTES, "fixture must produce an exact-size payload"
        assert _post(client, payload).status_code == 200

    def test_exactly_at_limit_not_rejected_by_size_gate(self, client):
        """Payload of exactly MAX_UPLOAD_BYTES must not trigger a 413."""
        # Whitespace is valid UTF-8 but not valid SRT; pipeline may return 502,
        # but the SIZE gate must remain silent.
        resp = _post(client, b" " * MAX_UPLOAD_BYTES)
        assert resp.status_code != 413

    def test_one_byte_under_limit_not_rejected(self, client):
        """MAX_UPLOAD_BYTES - 1 must not trigger the size gate."""
        resp = _post(client, b" " * (MAX_UPLOAD_BYTES - 1))
        assert resp.status_code != 413

    def test_two_bytes_over_returns_413(self, client):
        resp = _post(client, b"x" * (MAX_UPLOAD_BYTES + 2))
        assert resp.status_code == 413

    def test_far_over_limit_returns_413(self, client):
        """Double the limit is still a 413, not a server crash."""
        resp = _post(client, b"x" * (MAX_UPLOAD_BYTES * 2))
        assert resp.status_code == 413

    def test_max_upload_bytes_constant_is_one_mib(self):
        """Guard the constant itself; a misconfigured limit would silently break the contract."""
        assert MAX_UPLOAD_BYTES == 1_048_576, (
            f"MAX_UPLOAD_BYTES changed to {MAX_UPLOAD_BYTES}; update this test if intentional"
        )


# ---------------------------------------------------------------------------
# Response format for 413
# ---------------------------------------------------------------------------


class TestRejectionResponseFormat:
    """The 413 response must be well-formed JSON with a meaningful error key."""

    def _oversized(self) -> bytes:
        return b"x" * (MAX_UPLOAD_BYTES + 1)

    def test_413_content_type_is_json(self, client):
        resp = _post(client, self._oversized())
        assert resp.content_type.startswith("application/json"), (
            f"Expected JSON content-type, got {resp.content_type!r}"
        )

    def test_413_body_is_parseable_json(self, client):
        resp = _post(client, self._oversized())
        try:
            json.loads(resp.data)
        except json.JSONDecodeError as exc:
            pytest.fail(f"413 response body is not valid JSON: {exc}")

    def test_413_json_has_error_key(self, client):
        body = json.loads(_post(client, self._oversized()).data)
        assert "error" in body, f"'error' key missing from 413 response: {body}"

    def test_413_error_value_is_nonempty_string(self, client):
        body = json.loads(_post(client, self._oversized()).data)
        assert isinstance(body["error"], str) and body["error"]

    def test_413_error_message_is_human_readable(self, client):
        """Error text must reference size or limit so the client can act on it."""
        body = json.loads(_post(client, self._oversized()).data)
        msg = body["error"].lower()
        assert any(word in msg for word in ("size", "limit", "exceed", "mb", "byte", "large")), (
            f"Error message is not informative about size: {body['error']!r}"
        )

    def test_413_body_contains_no_traceback(self, client):
        """Server internals must not leak in the response."""
        resp = _post(client, self._oversized())
        text = resp.data.decode(errors="replace").lower()
        assert "traceback" not in text
        assert "exception" not in text


# ---------------------------------------------------------------------------
# Side-effects: no job stored, TTS never invoked
# ---------------------------------------------------------------------------


class TestRejectionSideEffects:
    """Oversized uploads must leave the server state completely unmodified."""

    def test_no_job_stored_after_single_oversized_request(self, client):
        _post(client, b"x" * (MAX_UPLOAD_BYTES + 1))
        assert _jobs == {}

    def test_no_job_stored_after_repeated_oversized_requests(self, client):
        for _ in range(3):
            _post(client, b"x" * (MAX_UPLOAD_BYTES + 1))
        assert _jobs == {}

    def test_tts_backend_not_invoked_for_oversized_upload(self, client, monkeypatch):
        """The synthesis backend must never be called when the upload is too large."""
        invocations: list[int] = []

        class _SpyBackend(_FixedBackend):
            def synthesize(self, segments):
                invocations.append(len(segments))
                return super().synthesize(segments)

        monkeypatch.setattr("dubbing.backends.say.SayTTSBackend", _SpyBackend)
        _post(client, b"x" * (MAX_UPLOAD_BYTES + 1))
        assert invocations == [], (
            "TTS backend was called despite oversized payload; the size gate fired too late"
        )

    def test_oversized_rejection_does_not_poison_subsequent_valid_request(self, client):
        """A rejected upload must not corrupt in-flight or future state."""
        _post(client, b"x" * (MAX_UPLOAD_BYTES + 1))
        valid_srt = b"1\n00:00:01,000 --> 00:00:03,000\nHello\n\n"
        resp = _post(client, valid_srt)
        assert resp.status_code == 200
        assert len(_jobs) == 1

    def test_job_count_unchanged_by_oversized_upload(self, client):
        """Rejected uploads must be invisible to the job registry."""
        valid_srt = b"1\n00:00:01,000 --> 00:00:03,000\nHello\n\n"
        _post(client, valid_srt)
        before = set(_jobs.keys())

        _post(client, b"x" * (MAX_UPLOAD_BYTES + 1))
        assert set(_jobs.keys()) == before


# ---------------------------------------------------------------------------
# Edge-case payloads
# ---------------------------------------------------------------------------


class TestEdgeCasePayloads:
    """Verify the size gate behaves sensibly with unusual payload content."""

    def test_oversized_binary_payload_returns_413(self, client):
        """Binary data well over the limit must be rejected before decode attempt."""
        payload = bytes(range(256)) * ((MAX_UPLOAD_BYTES // 256) + 2)
        resp = _post(client, payload)
        assert resp.status_code == 413

    def test_oversized_utf8_multibyte_content_returns_413(self, client):
        """Multi-byte UTF-8 characters; the limit applies to raw bytes, not code points."""
        # U+00E9 (é) encodes to 2 bytes; ensure the byte count drives the gate.
        two_byte_char = "é".encode("utf-8")  # b'\xc3\xa9'
        payload = two_byte_char * ((MAX_UPLOAD_BYTES // 2) + 1)
        # payload is > MAX_UPLOAD_BYTES bytes even though code point count is ~half
        assert len(payload) > MAX_UPLOAD_BYTES
        resp = _post(client, payload)
        assert resp.status_code == 413

    def test_empty_file_is_not_rejected_by_size_gate(self, client):
        """Zero-byte upload is below the limit; the error must be a content error, not 413."""
        resp = _post(client, b"")
        assert resp.status_code != 413

    def test_single_byte_upload_is_not_rejected_by_size_gate(self, client):
        resp = _post(client, b"x")
        assert resp.status_code != 413

    def test_valid_srt_well_within_limit_returns_200(self, client):
        """A small, ordinary SRT must always succeed."""
        srt = b"1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n"
        assert len(srt) < MAX_UPLOAD_BYTES
        assert _post(client, srt).status_code == 200

    def test_oversized_payload_with_valid_srt_prefix_still_rejected(self, client):
        """Even if the start of the payload looks like valid SRT, the gate must still fire."""
        srt_prefix = b"1\n00:00:01,000 --> 00:00:03,000\nHello\n\n"
        padding = b"x" * (MAX_UPLOAD_BYTES - len(srt_prefix) + 1)
        payload = srt_prefix + padding
        assert len(payload) > MAX_UPLOAD_BYTES
        assert _post(client, payload).status_code == 413
