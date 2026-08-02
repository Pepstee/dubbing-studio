from __future__ import annotations

import io
import re
import shutil
import subprocess
import wave

from dubbing.backends.base import TTSBackend
from dubbing.models import ProsodyTag, Segment, TTSResult

_RATE_WPM = {
    "x-slow": 100,
    "slow": 130,
    "medium": 175,
    "normal": 175,
    "fast": 220,
    "x-fast": 260,
}
_EMOTION_WPM = {
    "excited": 215,
    "happy": 195,
    "neutral": 175,
    "calm": 150,
    "sad": 130,
}
_PITCH = {
    "x-low": 20,
    "low": 35,
    "medium": 50,
    "normal": 50,
    "high": 65,
    "x-high": 80,
}
_LANGUAGE_RE = re.compile(r"^\s*\d+\s+([A-Za-z][A-Za-z0-9_-]*)\s")


def _duration_ms(audio: bytes) -> int:
    with wave.open(io.BytesIO(audio)) as wav:
        return round(wav.getnframes() * 1000 / wav.getframerate())


def _integer(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _rate(tags: list[ProsodyTag]) -> int | None:
    rate: int | None = None
    for tag in tags:
        if tag.name.lower() == "emotion":
            rate = _EMOTION_WPM.get(tag.value.lower(), rate)
    for tag in tags:
        if tag.name.lower() == "rate":
            value = tag.value.lower()
            rate = _RATE_WPM.get(value, _integer(value))
    return max(80, min(450, rate)) if rate is not None else None


def _pitch(tags: list[ProsodyTag]) -> int | None:
    pitch: int | None = None
    for tag in tags:
        if tag.name.lower() == "pitch":
            value = tag.value.lower()
            pitch = _PITCH.get(value, _integer(value))
    return max(0, min(99, pitch)) if pitch is not None else None


class EspeakTTSBackend(TTSBackend):
    """Cross-platform local TTS using the real ``espeak-ng`` engine."""

    def __init__(self, executable: str = "espeak-ng") -> None:
        self.executable = executable
        self._languages: set[str] | None = None

    def _require_executable(self) -> str:
        executable = shutil.which(self.executable)
        if executable is None:
            raise RuntimeError(
                f"{self.executable!r} command not found; install eSpeak NG or choose another backend"
            )
        return executable

    def _available_languages(self, executable: str) -> set[str]:
        if self._languages is None:
            proc = subprocess.run(
                [executable, "--voices"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            languages: set[str] = set()
            for line in proc.stdout.splitlines():
                match = _LANGUAGE_RE.match(line)
                if match:
                    language = match.group(1).replace("_", "-").lower()
                    languages.add(language)
                    languages.add(language.split("-")[0])
            self._languages = languages
        return self._languages

    def _voice(self, executable: str, language: str) -> str | None:
        if not language:
            return None
        normalized = language.replace("_", "-").lower()
        base = normalized.split("-")[0]
        languages = self._available_languages(executable)
        if normalized not in languages and base not in languages:
            available = ", ".join(sorted(item for item in languages if "-" not in item))
            raise RuntimeError(
                f"no installed eSpeak voice supports language {language!r}; "
                f"available base languages: {available}"
            )
        return normalized

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        executable = self._require_executable()
        results: list[TTSResult] = []
        for segment in segments:
            command = [executable, "--stdout"]
            voice = self._voice(executable, segment.language)
            if voice is not None:
                command.extend(["-v", voice])
            rate = _rate(segment.tags)
            if rate is not None:
                command.extend(["-s", str(rate)])
            pitch = _pitch(segment.tags)
            if pitch is not None:
                command.extend(["-p", str(pitch)])
            proc = subprocess.run(
                command,
                input=segment.entry.text.encode("utf-8"),
                capture_output=True,
                timeout=60,
                check=True,
            )
            if not proc.stdout:
                raise RuntimeError("eSpeak NG produced no audio")
            results.append(
                TTSResult(
                    segment=segment,
                    audio_bytes=proc.stdout,
                    duration_ms=_duration_ms(proc.stdout),
                )
            )
        return results
