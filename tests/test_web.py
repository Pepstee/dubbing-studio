from __future__ import annotations

import io
import json
import re
import uuid
import wave

import pytest

# Suite skips cleanly when Flask is absent (CI without optional deps).
flask = pytest.importorskip("flask")

from dubbing.backends.base import TTSBackend  # noqa: E402
from dubbing.models import Segment, TTSResult  # noqa: E402
from dubbing.web import (  # noqa: E402
    MAX_JOBS,
    MAX_JOBS_BYTES,
    MAX_UPLOAD_BYTES,
    _jobs,
    _store_job,
    app,
)


# ---------------------------------------------------------------------------
# Helpers
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


def _parse_job_id(resp) -> str:
    """Extract job_id from the HTML result page's /stream/<job_id> src."""
    html = resp.data.decode()
    m = re.search(r'/stream/([^"]+)', html)
    assert m, "Expected /stream/<job_id> in response HTML"
    return m.group(1)


def _is_valid_wav(data: bytes) -> bool:
    try:
        with wave.open(io.BytesIO(data)) as wf:
            return wf.getnframes() >= 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test double — named _FixedBackend, never Mock/fake/dummy/stub
# ---------------------------------------------------------------------------

class _FixedBackend(TTSBackend):
    """Returns pre-built silence WAV per segment; makes zero subprocess calls."""

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        wav = _minimal_wav()
        return [
            TTSResult(segment=seg, audio_bytes=wav, duration_ms=100)
            for seg in segments
        ]


# ---------------------------------------------------------------------------
# SRT fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    """Flask test client; real synthesis is replaced by _FixedBackend."""
    # Patch on the module that web.py imports from (inside the route body).
    monkeypatch.setattr("dubbing.backends.say.SayTTSBackend", _FixedBackend)
    _jobs.clear()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    _jobs.clear()


@pytest.fixture()
def job_id(client) -> str:
    """POST a single-entry SRT and return the resulting job id."""
    resp = client.post(
        "/dub",
        data={"srt": (io.BytesIO(_SRT_SINGLE.encode()), "test.srt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    return _parse_job_id(resp)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_body_contains_form_tag(self, client):
        assert "<form" in client.get("/").data.decode()

    def test_form_posts_to_dub_endpoint(self, client):
        assert 'action="/dub"' in client.get("/").data.decode()

    def test_form_method_is_post(self, client):
        body = client.get("/").data.decode().lower()
        assert 'method="post"' in body

    def test_body_contains_file_input(self, client):
        assert 'type="file"' in client.get("/").data.decode()

    def test_file_input_accepts_srt(self, client):
        assert ".srt" in client.get("/").data.decode()

    def test_body_has_html_doctype(self, client):
        assert "<!doctype html>" in client.get("/").data.decode().lower()

    def test_body_has_submit_button(self, client):
        body = client.get("/").data.decode()
        assert "<button" in body or 'type="submit"' in body

    def test_form_has_language_input(self, client):
        assert 'name="lang"' in client.get("/").data.decode()


# ---------------------------------------------------------------------------
# POST /dub — happy path
# ---------------------------------------------------------------------------

class TestDubRouteSuccess:
    def _post(self, client, srt: str = _SRT_SINGLE):
        return client.post(
            "/dub",
            data={"srt": (io.BytesIO(srt.encode()), "test.srt")},
            content_type="multipart/form-data",
        )

    def test_returns_200(self, client):
        assert self._post(client).status_code == 200

    def test_response_is_html(self, client):
        resp = self._post(client)
        assert resp.content_type.startswith("text/html")

    def test_html_has_audio_element(self, client):
        html = self._post(client).data.decode()
        assert "<audio" in html

    def test_html_has_download_link(self, client):
        html = self._post(client).data.decode()
        assert "Download WAV" in html

    def test_job_id_is_nonempty_string(self, client):
        job_id = _parse_job_id(self._post(client))
        assert isinstance(job_id, str) and job_id

    def test_stream_and_download_share_job_id(self, client):
        html = self._post(client).data.decode()
        stream_m = re.search(r'/stream/([^"]+)', html)
        download_m = re.search(r'/download/([^"]+)', html)
        assert stream_m and download_m
        assert stream_m.group(1) == download_m.group(1)

    def test_download_link_uses_download_path(self, client):
        html = self._post(client).data.decode()
        assert '/download/' in html

    def test_sequential_requests_produce_distinct_ids(self, client):
        id1 = _parse_job_id(self._post(client))
        id2 = _parse_job_id(self._post(client))
        assert id1 != id2

    def test_job_stored_in_jobs_registry(self, client):
        job_id = _parse_job_id(self._post(client))
        assert job_id in _jobs

    def test_job_audio_is_valid_wav(self, client):
        job_id = _parse_job_id(self._post(client))
        assert _is_valid_wav(_jobs[job_id])

    def test_multi_entry_srt_returns_200(self, client):
        assert self._post(client, _SRT_MULTI).status_code == 200

    def test_multi_entry_srt_has_download_link(self, client):
        html = self._post(client, _SRT_MULTI).data.decode()
        assert "Download WAV" in html


# ---------------------------------------------------------------------------
# POST /dub — error paths
# ---------------------------------------------------------------------------

class TestDubRouteErrors:
    def test_missing_srt_field_returns_400(self, client):
        resp = client.post("/dub", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_missing_srt_field_returns_json_error(self, client):
        resp = client.post("/dub", data={}, content_type="multipart/form-data")
        body = json.loads(resp.data)
        assert "error" in body

    def test_get_on_dub_returns_405(self, client):
        assert client.get("/dub").status_code == 405

    def test_non_utf8_upload_returns_400_not_500(self, client):
        """A binary (non-UTF-8) upload is a client error, never a crash."""
        resp = client.post(
            "/dub",
            data={"srt": (io.BytesIO(b"\xff\xfe\x00\x01RIFF junk"), "movie.srt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = json.loads(resp.data)
        assert "error" in body and "UTF-8" in body["error"]
        assert _jobs == {}, "no job must be stored for a rejected upload"

    def test_backend_failure_returns_502_json_error_not_silence(self, client, monkeypatch):
        """Synthesis failure must surface as an error — never silent audio."""

        class _FailingBackend(TTSBackend):
            def synthesize(self, segments):
                raise RuntimeError("'say' command not found; macOS TTS is unavailable")

        monkeypatch.setattr(
            "dubbing.backends.select_backend",
            lambda: _FailingBackend(),
        )
        resp = client.post(
            "/dub",
            data={"srt": (io.BytesIO(_SRT_SINGLE.encode()), "test.srt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 502
        body = json.loads(resp.data)
        assert "error" in body and "say" in body["error"]
        assert _jobs == {}, "no job must be stored when synthesis fails"

    def test_lang_field_forwarded_to_segments(self, client, monkeypatch):
        seen: list[str] = []

        class _RecordingBackend(_FixedBackend):
            def synthesize(self, segments):
                seen.extend(seg.language for seg in segments)
                return super().synthesize(segments)

        monkeypatch.setattr(
            "dubbing.backends.select_backend",
            lambda: _RecordingBackend(),
        )
        resp = client.post(
            "/dub",
            data={
                "srt": (io.BytesIO(_SRT_SINGLE.encode()), "test.srt"),
                "lang": "es",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert seen == ["es"]


# ---------------------------------------------------------------------------
# GET /download/<id> — happy path
# ---------------------------------------------------------------------------

class TestDownloadRouteSuccess:
    def test_valid_id_returns_200(self, client, job_id):
        assert client.get(f"/download/{job_id}").status_code == 200

    def test_content_type_is_audio_wav(self, client, job_id):
        resp = client.get(f"/download/{job_id}")
        assert resp.content_type.startswith("audio/wav")

    def test_response_body_is_valid_wav(self, client, job_id):
        resp = client.get(f"/download/{job_id}")
        assert _is_valid_wav(resp.data)

    def test_wav_has_mono_channel(self, client, job_id):
        resp = client.get(f"/download/{job_id}")
        with wave.open(io.BytesIO(resp.data)) as wf:
            assert wf.getnchannels() == 1

    def test_wav_has_22050_sample_rate(self, client, job_id):
        resp = client.get(f"/download/{job_id}")
        with wave.open(io.BytesIO(resp.data)) as wf:
            assert wf.getframerate() == 22050

    def test_download_url_from_dub_is_reachable(self, client):
        resp = client.post(
            "/dub",
            data={"srt": (io.BytesIO(_SRT_SINGLE.encode()), "test.srt")},
            content_type="multipart/form-data",
        )
        job_id = _parse_job_id(resp)
        assert client.get(f"/download/{job_id}").status_code == 200

    def test_download_url_content_type_is_audio_wav(self, client):
        resp = client.post(
            "/dub",
            data={"srt": (io.BytesIO(_SRT_SINGLE.encode()), "test.srt")},
            content_type="multipart/form-data",
        )
        job_id = _parse_job_id(resp)
        dl_resp = client.get(f"/download/{job_id}")
        assert dl_resp.content_type.startswith("audio/wav")


# ---------------------------------------------------------------------------
# GET /download/<id> — error paths
# ---------------------------------------------------------------------------

class TestDownloadRouteErrors:
    def test_unknown_uuid_returns_404(self, client):
        assert client.get(f"/download/{uuid.uuid4()}").status_code == 404

    def test_arbitrary_string_id_returns_404(self, client):
        assert client.get("/download/not-a-real-id").status_code == 404

    def test_empty_id_segment_returns_404(self, client):
        # /download/ with nothing after the slash hits the index or 404
        resp = client.get("/download/")
        assert resp.status_code in (404, 308)

    def test_previously_cleared_job_returns_404(self, client, job_id):
        _jobs.clear()
        assert client.get(f"/download/{job_id}").status_code == 404


# ---------------------------------------------------------------------------
# POST /dub — upload size limit
# ---------------------------------------------------------------------------

class TestDubUploadSizeLimit:
    def _oversized_payload(self) -> bytes:
        """One byte beyond MAX_UPLOAD_BYTES of valid ASCII text."""
        return b"x" * (MAX_UPLOAD_BYTES + 1)

    def test_oversized_upload_returns_413(self, client):
        resp = client.post(
            "/dub",
            data={"srt": (io.BytesIO(self._oversized_payload()), "big.srt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 413

    def test_oversized_upload_returns_json_error(self, client):
        resp = client.post(
            "/dub",
            data={"srt": (io.BytesIO(self._oversized_payload()), "big.srt")},
            content_type="multipart/form-data",
        )
        body = json.loads(resp.data)
        assert "error" in body

    def test_oversized_upload_no_job_stored(self, client):
        client.post(
            "/dub",
            data={"srt": (io.BytesIO(self._oversized_payload()), "big.srt")},
            content_type="multipart/form-data",
        )
        assert _jobs == {}

    def test_exactly_max_bytes_is_accepted(self, client):
        # MAX_UPLOAD_BYTES of whitespace is valid UTF-8 but not valid SRT;
        # the pipeline will reject it — that's a 502, not a 413.
        # We only verify the size gate itself does NOT trigger.
        resp = client.post(
            "/dub",
            data={"srt": (io.BytesIO(b" " * MAX_UPLOAD_BYTES), "limit.srt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code != 413

    def test_valid_srt_within_limit_returns_200(self, client):
        resp = client.post(
            "/dub",
            data={"srt": (io.BytesIO(_SRT_SINGLE.encode()), "test.srt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /dub — whole-request-body cap (MAX_CONTENT_LENGTH)
# ---------------------------------------------------------------------------

class TestRequestBodyCap:
    """The per-file cap alone is bypassable: request.files parses the whole
    multipart body first, so an oversized payload smuggled in any OTHER form
    field must be rejected by MAX_CONTENT_LENGTH before parsing."""

    def _post_huge_non_srt_field(self, client):
        return client.post(
            "/dub",
            data={
                "srt": (io.BytesIO(_SRT_SINGLE.encode()), "test.srt"),
                "lang": "x" * (2 * MAX_UPLOAD_BYTES),
            },
            content_type="multipart/form-data",
        )

    def test_max_content_length_is_configured(self):
        assert app.config["MAX_CONTENT_LENGTH"] is not None
        assert app.config["MAX_CONTENT_LENGTH"] >= MAX_UPLOAD_BYTES

    def test_huge_payload_in_non_srt_field_returns_413(self, client):
        assert self._post_huge_non_srt_field(client).status_code == 413

    def test_huge_payload_in_non_srt_field_returns_json_error(self, client):
        resp = self._post_huge_non_srt_field(client)
        body = json.loads(resp.data)
        assert "error" in body

    def test_huge_payload_in_non_srt_field_stores_no_job(self, client):
        self._post_huge_non_srt_field(client)
        assert _jobs == {}


# ---------------------------------------------------------------------------
# Job registry bounds — eviction by count and total bytes
# ---------------------------------------------------------------------------

class TestJobRegistryBounds:
    """Finished jobs must not accumulate without bound: a single legal
    4-hour timeline renders to ~600 MB, so unbounded retention is a
    memory-exhaustion DoS regardless of the upload-size cap."""

    def _post(self, client):
        return client.post(
            "/dub",
            data={"srt": (io.BytesIO(_SRT_SINGLE.encode()), "test.srt")},
            content_type="multipart/form-data",
        )

    def test_job_count_never_exceeds_max_jobs(self, client):
        for _ in range(MAX_JOBS + 5):
            assert self._post(client).status_code == 200
        assert len(_jobs) <= MAX_JOBS

    def test_oldest_job_evicted_first(self, client):
        first_id = _parse_job_id(self._post(client))
        for _ in range(MAX_JOBS):
            self._post(client)
        assert first_id not in _jobs

    def test_newest_job_survives_eviction(self, client):
        newest = None
        for _ in range(MAX_JOBS + 5):
            newest = _parse_job_id(self._post(client))
        assert newest in _jobs

    def test_evicted_job_download_returns_404(self, client):
        first_id = _parse_job_id(self._post(client))
        for _ in range(MAX_JOBS):
            self._post(client)
        assert client.get(f"/download/{first_id}").status_code == 404

    def test_newest_job_still_downloadable_after_eviction(self, client):
        newest_id = None
        for _ in range(MAX_JOBS + 5):
            newest_id = _parse_job_id(self._post(client))
        assert client.get(f"/download/{newest_id}").status_code == 200

    def test_store_job_evicts_on_total_bytes(self):
        _jobs.clear()
        try:
            half = MAX_JOBS_BYTES // 2 + 1
            first = _store_job(b"\x00" * half)
            second = _store_job(b"\x00" * half)
            assert first not in _jobs, "oldest job must be evicted at the byte cap"
            assert second in _jobs
        finally:
            _jobs.clear()

    def test_store_job_keeps_single_job_above_byte_cap(self):
        """One job larger than the cap is kept — the just-stored job is
        never evicted (the render is done; dropping it frees nothing sooner)."""
        _jobs.clear()
        try:
            job_id = _store_job(b"\x00" * (MAX_JOBS_BYTES + 1))
            assert job_id in _jobs
            assert len(_jobs) == 1
        finally:
            _jobs.clear()

    def test_store_job_under_caps_keeps_everything(self):
        _jobs.clear()
        try:
            ids = [_store_job(b"\x00" * 10) for _ in range(MAX_JOBS)]
            assert all(i in _jobs for i in ids)
        finally:
            _jobs.clear()
