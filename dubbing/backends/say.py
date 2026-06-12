from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from dubbing.backends.base import TTSBackend
from dubbing.models import ProsodyTag, Segment, TTSResult

_SAMPLE_RATE = 22050
_CHANNELS = 1
_SAMPLE_WIDTH = 2  # 16-bit PCM

# Speaking rate in words-per-minute for <rate:...> tag values.
_RATE_WPM = {
    "x-slow": 100,
    "slow": 130,
    "medium": 175,
    "normal": 175,
    "fast": 220,
    "x-fast": 260,
}

# Emotion profiles delivered as speaking-rate adjustments. `say` has no
# native emotion control, so emotions map onto pacing — the strongest
# expressive lever the engine exposes.
_EMOTION_WPM = {
    "excited": 215,
    "happy": 195,
    "neutral": 175,
    "calm": 150,
    "sad": 130,
}

# Pitch base (in `say`'s pbas units, default ≈ 46) for <pitch:...> values,
# delivered via the embedded speech command [[ pbas N ]].
_PITCH_PBAS = {
    "x-low": 30,
    "low": 38,
    "medium": 46,
    "normal": 46,
    "high": 54,
    "x-high": 62,
}

# `Name              lc_RG    # greeting` lines from `say -v ?`.
_VOICE_LINE_RE = re.compile(r"^(.*?)\s{2,}([a-zA-Z]{2,3}[-_][a-zA-Z]{2,})\s+#")

# Safe numeric bounds for `say` arguments.
_RATE_MIN = 20    # words per minute — say's documented minimum
_RATE_MAX = 500   # words per minute — say's documented maximum
_PITCH_MIN = -100
_PITCH_MAX = 100

_voices_cache: list[tuple[str, str]] | None = None


def _wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return int(wf.getnframes() * 1000 / wf.getframerate())


def _installed_voices() -> list[tuple[str, str]]:
    """Return (voice name, locale) pairs from `say -v ?`, cached per process."""
    global _voices_cache
    if _voices_cache is not None:
        return _voices_cache
    if shutil.which("say") is None:
        raise RuntimeError("'say' command not found; macOS TTS is unavailable")
    proc = subprocess.run(
        ["say", "-v", "?"], check=True, capture_output=True, text=True, timeout=30
    )
    voices: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        m = _VOICE_LINE_RE.match(line)
        if m:
            voices.append((m.group(1).strip(), m.group(2)))
    _voices_cache = voices
    return voices


def _voice_for_language(language: str) -> str | None:
    """Pick an installed `say` voice for a BCP-47 language code.

    Empty language → None (system default voice). Exact locale matches
    (e.g. "es-MX" → es_MX) win over base-language matches ("es" → first
    es_* voice). No installed voice for the language is an error — the
    backend never silently dubs in the wrong language.
    """
    if not language:
        return None
    norm = language.replace("-", "_").lower()
    base = norm.split("_")[0]
    voices = _installed_voices()
    for name, locale in voices:
        if locale.replace("-", "_").lower() == norm:
            return name
    for name, locale in voices:
        if locale.replace("-", "_").lower().split("_")[0] == base:
            return name
    available = ", ".join(sorted({loc.replace("-", "_").split("_")[0].lower() for _, loc in voices}))
    raise RuntimeError(
        f"no installed 'say' voice supports language {language!r}; "
        f"available languages: {available}"
    )


def _rate_for_tags(tags: list[ProsodyTag]) -> int | None:
    """Resolve <rate:...> / <emotion:...> tags to a speaking rate in WPM.

    An explicit rate tag (named value or integer WPM) overrides any
    emotion-derived pacing.
    """
    rate: int | None = None
    for tag in tags:
        if tag.name.lower() == "emotion":
            rate = _EMOTION_WPM.get(tag.value.lower(), rate)
    for tag in tags:
        if tag.name.lower() == "rate":
            value = tag.value.lower()
            if value in _RATE_WPM:
                rate = _RATE_WPM[value]
            elif value.isdigit():
                rate = int(value)
    return rate


def _pbas_for_tags(tags: list[ProsodyTag]) -> int | None:
    """Resolve <pitch:...> tags to a `say` pitch base (pbas) value."""
    pbas: int | None = None
    for tag in tags:
        if tag.name.lower() == "pitch":
            value = tag.value.lower()
            if value in _PITCH_PBAS:
                pbas = _PITCH_PBAS[value]
            elif value.lstrip("-").isdigit():
                pbas = int(value)
    return pbas


def _synthesize_with_say(
    text: str,
    voice: str | None = None,
    rate_wpm: int | None = None,
    pitch_pbas: int | None = None,
) -> bytes:
    if shutil.which("say") is None:
        raise RuntimeError("'say' command not found; macOS TTS is unavailable")
    if shutil.which("afconvert") is None:
        raise RuntimeError("'afconvert' command not found; macOS TTS is unavailable")

    if rate_wpm is not None:
        rate_wpm = max(_RATE_MIN, min(_RATE_MAX, rate_wpm))
    if pitch_pbas is not None:
        pitch_pbas = max(_PITCH_MIN, min(_PITCH_MAX, pitch_pbas))

    spoken = text
    if pitch_pbas is not None:
        # Embedded speech command: shift the pitch base for this utterance.
        spoken = f"[[ pbas {pitch_pbas} ]] {text}"

    with tempfile.TemporaryDirectory() as tmpdir:
        aiff_path = Path(tmpdir) / "out.aiff"
        wav_path = Path(tmpdir) / "out.wav"
        cmd = ["say", "-o", str(aiff_path)]
        if voice is not None:
            cmd += ["-v", voice]
        if rate_wpm is not None:
            cmd += ["-r", str(rate_wpm)]
        # The text is delivered on stdin, never as an argv item: subtitle
        # lines routinely start with "-" (dialogue dashes), which `say`
        # would parse as options — silently selecting a wrong voice/rate or
        # even redirecting -o output to an attacker-chosen path.
        subprocess.run(
            cmd, check=True, capture_output=True, timeout=30,
            input=spoken.encode("utf-8"),
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(aiff_path), str(wav_path)],
            check=True, capture_output=True, timeout=30,
        )
        return wav_path.read_bytes()


class SayTTSBackend(TTSBackend):
    """TTS backend using macOS `say` and `afconvert`.

    Honours `Segment.language` by selecting an installed voice for that
    language, and delivers prosody tags: <rate:...> and <emotion:...> set
    the speaking rate (`say -r`), <pitch:...> shifts the pitch base via an
    embedded `[[ pbas N ]]` speech command.
    """

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        results: list[TTSResult] = []
        for seg in segments:
            voice = _voice_for_language(seg.language)
            audio = _synthesize_with_say(
                seg.entry.text,
                voice=voice,
                rate_wpm=_rate_for_tags(seg.tags),
                pitch_pbas=_pbas_for_tags(seg.tags),
            )
            results.append(TTSResult(segment=seg, audio_bytes=audio, duration_ms=_wav_duration_ms(audio)))
        return results
