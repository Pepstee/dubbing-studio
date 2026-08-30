"""Tests for GET /stream/<job_id>.

Acceptance criteria:
  - GET /stream/<valid_id>  → 200, Content-Type audio/wav, Content-Disposition inline
  - GET /stream/<unknown>   → 404
"""

from __future__ import annotations

import io
import threading
import uuid
import wave

import pytest

flask = pytest.importorskip("flask")

from dubbing.backends.base import TTSBackend  # noqa: E402
from dubbing.models import Segment, TTSResult  # noqa: E402
from dubbing.apps.dubbing_web import _jobs, _store_job, app  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00" * 2)
    return buf.getvalue()


class _FixedBackend(TTSBackend):
    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        wav = _minimal_wav()
        return [TTSResult(segment=seg, audio_bytes=wav, duration_ms=100) for seg in segments]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with no real TTS."""
    monkeypatch.setattr("dubbing.backends.say.SayTTSBackend", _FixedBackend)
    _jobs.clear()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    _jobs.clear()


@pytest.fixture()
def stored_job(client) -> str:
    """Pre-store a minimal WAV and return the job_id.

    Depends on *client* so that _jobs is cleared before the job is added.
    """
    return _store_job(_minimal_wav())


@pytest.fixture()
def large_stored_job(client) -> str:
    """Pre-store a longer WAV (~0.5 s) so the body is meaningfully large."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00" * 2 * 11025)  # 0.5 s of silence
    return _store_job(buf.getvalue())


# ---------------------------------------------------------------------------
# GET /stream/<job_id> — happy path
# ---------------------------------------------------------------------------


class TestStreamSuccess:
    def test_returns_200_for_stored_job(self, client, stored_job):
        resp = client.get(f"/stream/{stored_job}")
        assert resp.status_code == 200

    def test_content_type_is_audio_wav(self, client, stored_job):
        resp = client.get(f"/stream/{stored_job}")
        assert resp.content_type == "audio/wav"

    def test_content_disposition_is_inline_not_attachment(self, client, stored_job):
        resp = client.get(f"/stream/{stored_job}")
        cd = resp.headers.get("Content-Disposition", "")
        # Inline disposition (or no header) → browser plays in-page.
        # Attachment disposition → browser triggers a file download.
        assert not cd.startswith("attachment"), f"Expected inline disposition, got: {cd!r}"

    def test_content_disposition_contains_inline(self, client, stored_job):
        # Flask sets "inline; filename=..." when as_attachment=False and
        # download_name is provided.
        resp = client.get(f"/stream/{stored_job}")
        cd = resp.headers.get("Content-Disposition", "")
        # Either the header is absent (browser default = inline) or it starts
        # with "inline".  It must NOT start with "attachment".
        assert cd == "" or cd.startswith("inline"), (
            f"Expected absent or inline Content-Disposition, got: {cd!r}"
        )

    def test_body_is_valid_wav(self, client, stored_job):
        resp = client.get(f"/stream/{stored_job}")
        with wave.open(io.BytesIO(resp.data)) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    def test_body_bytes_match_stored_audio(self, client):
        wav = _minimal_wav()
        job_id = _store_job(wav)
        resp = client.get(f"/stream/{job_id}")
        assert resp.data == wav

    def test_large_wav_body_is_returned_intact(self, client, large_stored_job):
        resp = client.get(f"/stream/{large_stored_job}")
        assert resp.status_code == 200
        assert len(resp.data) > 100  # more than header-only
        with wave.open(io.BytesIO(resp.data)) as wf:
            assert wf.getnframes() > 0

    def test_two_stored_jobs_can_both_be_retrieved_independently(self, client):
        job1 = _store_job(_minimal_wav())
        job2 = _store_job(_minimal_wav())
        resp1 = client.get(f"/stream/{job1}")
        resp2 = client.get(f"/stream/{job2}")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.data == resp2.data  # both stored same content → same bytes


# ---------------------------------------------------------------------------
# GET /stream/<job_id> — 404 paths
# ---------------------------------------------------------------------------


class TestStreamNotFound:
    def test_unknown_string_returns_404(self, client):
        resp = client.get("/stream/does-not-exist")
        assert resp.status_code == 404

    def test_uuid_not_in_registry_returns_404(self, client):
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/stream/{fake_id}")
        assert resp.status_code == 404

    def test_empty_registry_returns_404_for_any_id(self, client):
        _jobs.clear()
        resp = client.get("/stream/anything")
        assert resp.status_code == 404

    def test_evicted_job_returns_404(self, client):
        # Fill the registry past MAX_JOBS to force eviction of the first job.
        from dubbing.apps.dubbing_web import MAX_JOBS

        first_job = _store_job(_minimal_wav())
        for _ in range(MAX_JOBS):
            _store_job(_minimal_wav())
        # The first job should have been evicted.
        assert first_job not in _jobs
        resp = client.get(f"/stream/{first_job}")
        assert resp.status_code == 404

    def test_404_body_does_not_expose_traceback(self, client):
        resp = client.get("/stream/no-such-job")
        body = resp.data.decode(errors="replace")
        assert "Traceback" not in body
        assert "Exception" not in body

    def test_uppercase_uuid_shape_is_rejected_before_lookup(self, client):
        job_id = str(uuid.uuid4()).upper()
        _jobs[job_id] = _minimal_wav()
        assert client.get(f"/stream/{job_id}").status_code == 404


class TestConcurrentRegistry:
    def test_parallel_stores_create_distinct_retrievable_jobs(self, client):
        created: list[str] = []
        lock = threading.Lock()

        def store() -> None:
            job_id = _store_job(_minimal_wav())
            with lock:
                created.append(job_id)

        threads = [threading.Thread(target=store) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert len(created) == 8
        assert len(set(created)) == 8
        assert all(client.get(f"/stream/{job_id}").status_code == 200 for job_id in created)


# ---------------------------------------------------------------------------
# Contrast with /download — same job, different disposition
# ---------------------------------------------------------------------------


class TestStreamVsDownloadDisposition:
    def test_stream_is_inline_while_download_is_attachment(self, client, stored_job):
        stream_resp = client.get(f"/stream/{stored_job}")
        download_resp = client.get(f"/download/{stored_job}")

        stream_cd = stream_resp.headers.get("Content-Disposition", "")
        download_cd = download_resp.headers.get("Content-Disposition", "")

        # Stream should NOT be attachment.
        assert not stream_cd.startswith("attachment")
        # Download SHOULD be attachment.
        assert download_cd.startswith("attachment")

    def test_both_routes_return_audio_wav_content_type(self, client, stored_job):
        stream_resp = client.get(f"/stream/{stored_job}")
        download_resp = client.get(f"/download/{stored_job}")
        assert stream_resp.content_type == "audio/wav"
        assert download_resp.content_type == "audio/wav"

    def test_both_routes_return_identical_bytes(self, client, stored_job):
        stream_resp = client.get(f"/stream/{stored_job}")
        download_resp = client.get(f"/download/{stored_job}")
        assert stream_resp.data == download_resp.data
