from __future__ import annotations

import io
import shutil
import subprocess
import wave
from unittest.mock import patch

import pytest

from dubbing.backends.base import TTSBackend
from dubbing.backends.say import (
    SayTTSBackend,
    _pbas_for_tags,
    _rate_for_tags,
    _synthesize_with_say,
    _voice_for_language,
)
from dubbing.models import ProsodyTag, Segment, SRTEntry, TTSResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entry(index: int = 1, start_ms: int = 0, end_ms: int = 1000, text: str = "Hello") -> SRTEntry:
    return SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text)


def make_segment(text: str = "Hello", tags: list[ProsodyTag] | None = None, language: str = "en") -> Segment:
    return Segment(entry=make_entry(text=text), tags=tags or [], language=language)


# ---------------------------------------------------------------------------
# Inheritance — no subprocess involvement
# ---------------------------------------------------------------------------

def test_say_backend_is_tts_backend():
    assert issubclass(SayTTSBackend, TTSBackend)


def test_instance_is_tts_backend():
    assert isinstance(SayTTSBackend(), TTSBackend)


# ---------------------------------------------------------------------------
# Raise-on-unavailable contract — _synthesize_with_say helper directly
# ---------------------------------------------------------------------------

def test_raises_runtime_error_when_say_not_found():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="'say' command not found"):
            _synthesize_with_say("hello")


def test_say_error_message_mentions_macos_tts():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="macOS TTS is unavailable"):
            _synthesize_with_say("test")


def test_raises_runtime_error_when_afconvert_not_found():
    def which_side_effect(name: str) -> str | None:
        return "/usr/bin/say" if name == "say" else None

    with patch("shutil.which", side_effect=which_side_effect):
        with pytest.raises(RuntimeError, match="'afconvert' command not found"):
            _synthesize_with_say("hello")


def test_afconvert_error_message_mentions_macos_tts():
    def which_side_effect(name: str) -> str | None:
        return "/usr/bin/say" if name == "say" else None

    with patch("shutil.which", side_effect=which_side_effect):
        with pytest.raises(RuntimeError, match="macOS TTS is unavailable"):
            _synthesize_with_say("hello")


def test_say_checked_before_afconvert_when_both_missing():
    """say is the first guard — its error takes precedence over afconvert's."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="'say' command not found"):
            _synthesize_with_say("hello")


def test_error_is_runtime_error_not_subclass():
    with patch("shutil.which", return_value=None):
        exc = pytest.raises(RuntimeError, _synthesize_with_say, "x")
        assert type(exc.value) is RuntimeError


# ---------------------------------------------------------------------------
# Raise-on-unavailable contract — SayTTSBackend.synthesize propagates
# ---------------------------------------------------------------------------

def test_synthesize_raises_when_say_unavailable():
    backend = SayTTSBackend()
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="'say' command not found"):
            backend.synthesize([make_segment()])


def test_synthesize_raises_when_afconvert_unavailable():
    backend = SayTTSBackend()

    def which_side_effect(name: str) -> str | None:
        return "/usr/bin/say" if name == "say" else None

    with patch("shutil.which", side_effect=which_side_effect):
        with pytest.raises(RuntimeError, match="'afconvert' command not found"):
            backend.synthesize([make_segment()])


def test_synthesize_no_silence_fallback_when_unavailable():
    """The old contract returned silent audio when unavailable; the new contract raises."""
    backend = SayTTSBackend()
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError):
            backend.synthesize([make_segment("anything")])


def test_synthesize_raises_on_first_segment_when_unavailable():
    backend = SayTTSBackend()
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="'say' command not found"):
            backend.synthesize([make_segment("first"), make_segment("second")])


def test_synthesize_raises_when_both_tools_missing():
    """Both say and afconvert absent — RuntimeError, not a partial list."""
    backend = SayTTSBackend()
    segments = [make_segment(f"seg {i}") for i in range(3)]
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError):
            backend.synthesize(segments)


# ---------------------------------------------------------------------------
# Empty segment list — no subprocess call; unavailable tools are irrelevant
# ---------------------------------------------------------------------------

def test_empty_segment_list_yields_empty_list():
    backend = SayTTSBackend()
    with patch("shutil.which", return_value=None):
        results = backend.synthesize([])
    assert results == []


def test_empty_segment_list_result_is_list():
    backend = SayTTSBackend()
    with patch("shutil.which", return_value=None):
        results = backend.synthesize([])
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Subprocess failure propagates as CalledProcessError
# ---------------------------------------------------------------------------

def test_synthesize_propagates_say_subprocess_error():
    """If say exits non-zero, CalledProcessError must surface — not be swallowed."""
    backend = SayTTSBackend()

    def which_ok(name: str) -> str:
        return f"/usr/bin/{name}"

    with patch("shutil.which", side_effect=which_ok):
        with patch("dubbing.backends.say.subprocess.run",
                   side_effect=subprocess.CalledProcessError(1, ["say"])):
            with pytest.raises(subprocess.CalledProcessError):
                backend.synthesize([make_segment("hello")])


# ---------------------------------------------------------------------------
# Prosody tag → say parameter mapping (pure functions, no subprocess)
# ---------------------------------------------------------------------------

class TestRateForTags:
    def test_no_tags_yields_none(self):
        assert _rate_for_tags([]) is None

    def test_rate_slow_maps_to_low_wpm(self):
        rate = _rate_for_tags([ProsodyTag(name="rate", value="slow")])
        assert rate is not None and rate < 175

    def test_rate_fast_maps_to_high_wpm(self):
        rate = _rate_for_tags([ProsodyTag(name="rate", value="fast")])
        assert rate is not None and rate > 175

    def test_numeric_rate_used_verbatim(self):
        assert _rate_for_tags([ProsodyTag(name="rate", value="190")]) == 190

    def test_emotion_excited_speeds_up(self):
        rate = _rate_for_tags([ProsodyTag(name="emotion", value="excited")])
        assert rate is not None and rate > 175

    def test_emotion_sad_slows_down(self):
        rate = _rate_for_tags([ProsodyTag(name="emotion", value="sad")])
        assert rate is not None and rate < 175

    def test_explicit_rate_overrides_emotion(self):
        tags = [ProsodyTag(name="emotion", value="excited"), ProsodyTag(name="rate", value="slow")]
        assert _rate_for_tags(tags) == _rate_for_tags([ProsodyTag(name="rate", value="slow")])

    def test_unknown_emotion_yields_none(self):
        assert _rate_for_tags([ProsodyTag(name="emotion", value="bewildered")]) is None


class TestPbasForTags:
    def test_no_tags_yields_none(self):
        assert _pbas_for_tags([]) is None

    def test_pitch_low_below_default(self):
        pbas = _pbas_for_tags([ProsodyTag(name="pitch", value="low")])
        assert pbas is not None and pbas < 46

    def test_pitch_high_above_default(self):
        pbas = _pbas_for_tags([ProsodyTag(name="pitch", value="high")])
        assert pbas is not None and pbas > 46

    def test_numeric_pitch_used_verbatim(self):
        assert _pbas_for_tags([ProsodyTag(name="pitch", value="40")]) == 40

    def test_non_pitch_tags_ignored(self):
        assert _pbas_for_tags([ProsodyTag(name="emotion", value="happy")]) is None


# ---------------------------------------------------------------------------
# Voice selection by language (voice list seeded, no subprocess)
# ---------------------------------------------------------------------------

_VOICE_FIXTURE = [
    ("Albert", "en_US"),
    ("Daniel", "en_GB"),
    ("Mónica", "es_ES"),
    ("Paulina", "es_MX"),
    ("Thomas", "fr_FR"),
]


@pytest.fixture()
def seeded_voices(monkeypatch):
    monkeypatch.setattr("dubbing.backends.say._voices_cache", list(_VOICE_FIXTURE))


class TestVoiceForLanguage:
    def test_empty_language_uses_system_default(self, seeded_voices):
        assert _voice_for_language("") is None

    def test_base_language_picks_first_matching_voice(self, seeded_voices):
        assert _voice_for_language("es") == "Mónica"

    def test_exact_locale_match_wins(self, seeded_voices):
        assert _voice_for_language("es-MX") == "Paulina"

    def test_locale_underscore_form_accepted(self, seeded_voices):
        assert _voice_for_language("en_GB") == "Daniel"

    def test_case_insensitive(self, seeded_voices):
        assert _voice_for_language("FR") == "Thomas"

    def test_unsupported_language_raises_with_available_list(self, seeded_voices):
        with pytest.raises(RuntimeError, match="no installed 'say' voice"):
            _voice_for_language("xx")

    def test_error_lists_available_languages(self, seeded_voices):
        with pytest.raises(RuntimeError, match="es"):
            _voice_for_language("xx")


# ---------------------------------------------------------------------------
# Command construction — subprocess captured, never executed
# ---------------------------------------------------------------------------

class TestSayCommandConstruction:
    def _capture_commands(self, monkeypatch, tmp_path):
        calls: list[list[str]] = []

        def record(cmd, **kwargs):
            calls.append(list(cmd))
            if cmd[0] == "afconvert":
                # produce a real WAV so the backend can read it back
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(b"\x00\x00" * 220)
                from pathlib import Path
                Path(cmd[-1]).write_bytes(buf.getvalue())
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr("dubbing.backends.say.subprocess.run", record)
        return calls

    def test_voice_flag_passed_to_say(self, monkeypatch, tmp_path):
        calls = self._capture_commands(monkeypatch, tmp_path)
        _synthesize_with_say("hola", voice="Mónica")
        say_cmd = calls[0]
        assert "-v" in say_cmd and say_cmd[say_cmd.index("-v") + 1] == "Mónica"

    def test_rate_flag_passed_to_say(self, monkeypatch, tmp_path):
        calls = self._capture_commands(monkeypatch, tmp_path)
        _synthesize_with_say("hello", rate_wpm=130)
        say_cmd = calls[0]
        assert "-r" in say_cmd and say_cmd[say_cmd.index("-r") + 1] == "130"

    def test_pitch_embedded_as_pbas_command(self, monkeypatch, tmp_path):
        calls = self._capture_commands(monkeypatch, tmp_path)
        _synthesize_with_say("hello", pitch_pbas=38)
        spoken = calls[0][-1]
        assert "[[ pbas 38 ]]" in spoken and "hello" in spoken

    def test_no_options_means_bare_command(self, monkeypatch, tmp_path):
        calls = self._capture_commands(monkeypatch, tmp_path)
        _synthesize_with_say("plain")
        say_cmd = calls[0]
        assert "-v" not in say_cmd and "-r" not in say_cmd
        assert say_cmd[-1] == "plain"

    def test_synthesize_applies_segment_tags(self, monkeypatch, tmp_path, seeded_voices):
        calls = self._capture_commands(monkeypatch, tmp_path)
        seg = Segment(
            entry=make_entry(text="Bonjour"),
            tags=[ProsodyTag(name="rate", value="slow"), ProsodyTag(name="pitch", value="low")],
            language="fr",
        )
        SayTTSBackend().synthesize([seg])
        say_cmd = calls[0]
        assert say_cmd[say_cmd.index("-v") + 1] == "Thomas"
        assert "-r" in say_cmd
        assert "[[ pbas" in say_cmd[-1]


# ---------------------------------------------------------------------------
# Full synthesis — macOS only (real say + afconvert required)
# ---------------------------------------------------------------------------

_macos_tts_available = (
    shutil.which("say") is not None and shutil.which("afconvert") is not None
)

macos_only = pytest.mark.skipif(
    not _macos_tts_available,
    reason="requires macOS say and afconvert",
)


@macos_only
def test_synthesize_returns_list_of_results():
    results = SayTTSBackend().synthesize([make_segment()])
    assert isinstance(results, list)


@macos_only
def test_single_segment_yields_one_result():
    results = SayTTSBackend().synthesize([make_segment()])
    assert len(results) == 1


@macos_only
def test_three_segments_yield_three_results():
    segments = [make_segment(f"test {i}") for i in range(3)]
    results = SayTTSBackend().synthesize(segments)
    assert len(results) == 3


@macos_only
def test_results_are_ttsresult_instances():
    results = SayTTSBackend().synthesize([make_segment(), make_segment("World")])
    assert all(isinstance(r, TTSResult) for r in results)


@macos_only
def test_audio_bytes_is_bytes():
    result = SayTTSBackend().synthesize([make_segment()])[0]
    assert isinstance(result.audio_bytes, bytes)


@macos_only
def test_audio_bytes_is_non_empty():
    result = SayTTSBackend().synthesize([make_segment()])[0]
    assert len(result.audio_bytes) > 0


@macos_only
def test_audio_is_valid_wav_file():
    """Output must parse as WAV with the declared encoding, not raw PCM or AIFF."""
    result = SayTTSBackend().synthesize([make_segment("hello")])[0]
    with wave.open(io.BytesIO(result.audio_bytes)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 22050
        assert wf.getsampwidth() == 2


@macos_only
def test_wav_starts_with_riff_header():
    result = SayTTSBackend().synthesize([make_segment("test")])[0]
    assert result.audio_bytes[:4] == b"RIFF"
    assert result.audio_bytes[8:12] == b"WAVE"


@macos_only
def test_duration_ms_positive():
    result = SayTTSBackend().synthesize([make_segment("Hello")])[0]
    assert result.duration_ms > 0


@macos_only
def test_duration_ms_is_integer():
    result = SayTTSBackend().synthesize([make_segment("Hello")])[0]
    assert isinstance(result.duration_ms, int)


@macos_only
def test_result_segment_is_input_segment_identity():
    seg = make_segment("test")
    result = SayTTSBackend().synthesize([seg])[0]
    assert result.segment is seg


@macos_only
def test_result_segments_order_matches_input():
    segments = [make_segment(f"text {i}") for i in range(5)]
    results = SayTTSBackend().synthesize(segments)
    for res, seg in zip(results, segments):
        assert res.segment is seg


@macos_only
def test_prosody_tags_preserved_in_result():
    tag = ProsodyTag(name="emotion", value="happy")
    seg = Segment(entry=make_entry(text="Hello"), tags=[tag], language="en")
    result = SayTTSBackend().synthesize([seg])[0]
    assert result.segment.tags == [tag]


@macos_only
def test_language_preserved_in_result():
    seg = Segment(entry=make_entry(text="Hola"), tags=[], language="es")
    result = SayTTSBackend().synthesize([seg])[0]
    assert result.segment.language == "es"


@macos_only
def test_direct_synthesize_helper_returns_bytes():
    data = _synthesize_with_say("Hello world")
    assert isinstance(data, bytes)
    assert len(data) > 0


@macos_only
def test_direct_synthesize_helper_returns_valid_wav():
    data = _synthesize_with_say("test audio")
    with wave.open(io.BytesIO(data)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 22050
        assert wf.getsampwidth() == 2
