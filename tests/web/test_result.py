"""Tests for the HTML result page returned by POST /dub.

Acceptance criteria:
  - response.content_type starts with 'text/html'
  - body contains '<audio' and '/stream/'
  - body contains a download link to '/download/'
"""
from __future__ import annotations

import io
import re
import wave

import pytest

flask = pytest.importorskip("flask")

from dubbing.backends.base import TTSBackend  # noqa: E402
from dubbing.models import Segment, TTSResult  # noqa: E402
from dubbing.apps.dubbing_web import _jobs, app  # noqa: E402


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
    """Returns a minimal WAV per segment; makes zero subprocess calls."""

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        wav = _minimal_wav()
        return [TTSResult(segment=seg, audio_bytes=wav, duration_ms=100) for seg in segments]


_SRT_MINIMAL = """\
1
00:00:01,000 --> 00:00:03,000
Hello world

"""

_SRT_MULTI = """\
1
00:00:00,000 --> 00:00:02,000
First subtitle

2
00:00:03,000 --> 00:00:05,000
Second subtitle

3
00:00:06,000 --> 00:00:08,000
Third subtitle

"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    """Flask test client; real TTS is replaced by _FixedBackend."""
    monkeypatch.setattr("dubbing.backends.say.SayTTSBackend", _FixedBackend)
    _jobs.clear()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    _jobs.clear()


def _post_srt(client, srt: str = _SRT_MINIMAL, lang: str = "") -> object:
    data: dict = {"srt": (io.BytesIO(srt.encode()), "test.srt")}
    if lang:
        data["lang"] = lang
    return client.post("/dub", data=data, content_type="multipart/form-data")


# ---------------------------------------------------------------------------
# Content-Type
# ---------------------------------------------------------------------------

class TestResultContentType:
    def test_content_type_starts_with_text_html(self, client):
        resp = _post_srt(client)
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")

    def test_content_type_specifies_charset(self, client):
        resp = _post_srt(client)
        assert "charset" in resp.content_type

    def test_multi_segment_srt_also_returns_html(self, client):
        resp = _post_srt(client, _SRT_MULTI)
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")


# ---------------------------------------------------------------------------
# Body — audio element
# ---------------------------------------------------------------------------

class TestResultBodyAudioElement:
    def test_body_contains_audio_tag(self, client):
        html = _post_srt(client).data.decode()
        assert "<audio" in html

    def test_audio_src_points_to_stream_path(self, client):
        html = _post_srt(client).data.decode()
        # <audio controls src="/stream/<job_id>">
        assert re.search(r'src="/stream/[^""]+"', html) is not None

    def test_audio_element_has_controls_attribute(self, client):
        html = _post_srt(client).data.decode()
        m = re.search(r'<audio\b([^>]*)>', html)
        assert m is not None
        assert "controls" in m.group(1)

    def test_stream_path_appears_in_body(self, client):
        html = _post_srt(client).data.decode()
        assert "/stream/" in html


# ---------------------------------------------------------------------------
# Body — download link
# ---------------------------------------------------------------------------

class TestResultBodyDownloadLink:
    def test_body_contains_download_path(self, client):
        html = _post_srt(client).data.decode()
        assert "/download/" in html

    def test_download_link_is_an_anchor(self, client):
        html = _post_srt(client).data.decode()
        assert re.search(r'<a\b[^>]*href="/download/[^""]+"', html) is not None

    def test_stream_and_download_share_the_same_job_id(self, client):
        html = _post_srt(client).data.decode()
        stream_m = re.search(r"/stream/([^\"<\s]+)", html)
        download_m = re.search(r"/download/([^\"<\s]+)", html)
        assert stream_m is not None and download_m is not None
        assert stream_m.group(1) == download_m.group(1)

    def test_job_id_in_html_is_present_in_registry(self, client):
        html = _post_srt(client).data.decode()
        m = re.search(r"/stream/([^\"<\s]+)", html)
        assert m is not None
        assert m.group(1) in _jobs


# ---------------------------------------------------------------------------
# Job-ID uniqueness and registry
# ---------------------------------------------------------------------------

class TestResultJobRegistry:
    def test_each_post_produces_a_distinct_job_id(self, client):
        html1 = _post_srt(client).data.decode()
        html2 = _post_srt(client).data.decode()
        m1 = re.search(r"/stream/([^\"<\s]+)", html1)
        m2 = re.search(r"/stream/([^\"<\s]+)", html2)
        assert m1 is not None and m2 is not None
        assert m1.group(1) != m2.group(1)

    def test_job_id_is_nonempty(self, client):
        html = _post_srt(client).data.decode()
        m = re.search(r"/stream/([^\"<\s]+)", html)
        assert m is not None
        assert m.group(1).strip() != ""

    def test_successful_post_increments_job_count(self, client):
        before = len(_jobs)
        _post_srt(client)
        assert len(_jobs) == before + 1


# ---------------------------------------------------------------------------
# Result page is valid HTML (smoke check, not a full validator)
# ---------------------------------------------------------------------------

class TestResultPageStructure:
    def test_result_page_has_doctype(self, client):
        html = _post_srt(client).data.decode().lower()
        assert "<!doctype html>" in html

    def test_result_page_has_html_tag(self, client):
        html = _post_srt(client).data.decode()
        assert "<html" in html

    def test_result_page_has_head_and_body(self, client):
        html = _post_srt(client).data.decode()
        assert "<head" in html and "<body" in html

    def test_result_page_title_is_present(self, client):
        html = _post_srt(client).data.decode()
        assert "<title" in html
