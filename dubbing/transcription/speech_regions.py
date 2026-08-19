from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dubbing.transcription.models import TranscriptionError


@dataclass(frozen=True)
class SpeechRegion:
    start_ms: int
    end_ms: int
    source_pass: str

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("speech region must satisfy 0 <= start_ms < end_ms")
        if self.source_pass not in {"strict", "sensitive"}:
            raise ValueError("speech region source_pass must be strict or sensitive")

    def to_dict(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "source_pass": self.source_pass,
        }


@dataclass(frozen=True)
class SpeechRegionPlan:
    detector_identity: str
    duration_ms: int
    strict_regions: tuple[SpeechRegion, ...]
    sensitive_regions: tuple[SpeechRegion, ...]
    selected_regions: tuple[SpeechRegion, ...]

    def to_dict(self) -> dict:
        return {
            "detector_identity": self.detector_identity,
            "duration_ms": self.duration_ms,
            "strict_regions": [item.to_dict() for item in self.strict_regions],
            "sensitive_regions": [item.to_dict() for item in self.sensitive_regions],
            "selected_regions": [item.to_dict() for item in self.selected_regions],
        }


class SpeechRegionDetector(Protocol):
    @property
    def identity(self) -> str: ...

    def detect(self, audio_path: str | Path) -> SpeechRegionPlan: ...


def _overlaps(left: SpeechRegion, right: SpeechRegion) -> bool:
    return left.start_ms < right.end_ms and left.end_ms > right.start_ms


def select_complementary_regions(
    strict_regions: tuple[SpeechRegion, ...],
    sensitive_regions: tuple[SpeechRegion, ...],
) -> tuple[SpeechRegion, ...]:
    """Prefer precise regions and add only genuinely new sensitive detections."""
    selected = list(strict_regions)
    for candidate in sensitive_regions:
        if any(_overlaps(candidate, existing) for existing in strict_regions):
            continue
        if any(_overlaps(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: (item.start_ms, item.end_ms)))


class FasterWhisperSileroSpeechRegionDetector:
    """Local semantic VAD using the Silero model bundled with Faster-Whisper."""

    def __init__(
        self,
        *,
        strict_threshold: float = 0.2,
        sensitive_threshold: float = 0.1,
        minimum_speech_ms: int = 40,
        minimum_silence_ms: int = 180,
        speech_pad_ms: int = 150,
        maximum_region_seconds: float = 8.0,
        sample_rate_hz: int = 16_000,
    ) -> None:
        if not 0 < sensitive_threshold < strict_threshold < 1:
            raise ValueError("VAD thresholds must satisfy 0 < sensitive < strict < 1")
        if minimum_speech_ms < 0 or minimum_silence_ms < 0 or speech_pad_ms < 0:
            raise ValueError("VAD timing parameters cannot be negative")
        if maximum_region_seconds <= 0 or sample_rate_hz <= 0:
            raise ValueError("VAD maximum region and sample rate must be positive")
        self.strict_threshold = strict_threshold
        self.sensitive_threshold = sensitive_threshold
        self.minimum_speech_ms = minimum_speech_ms
        self.minimum_silence_ms = minimum_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.maximum_region_seconds = maximum_region_seconds
        self.sample_rate_hz = sample_rate_hz

    @property
    def identity(self) -> str:
        return (
            "faster-whisper-silero:"
            f"strict={self.strict_threshold}:sensitive={self.sensitive_threshold}:"
            f"min_speech_ms={self.minimum_speech_ms}:"
            f"min_silence_ms={self.minimum_silence_ms}:pad_ms={self.speech_pad_ms}:"
            f"max_region_s={self.maximum_region_seconds}:rate={self.sample_rate_hz}"
        )

    @staticmethod
    def _dependencies():
        try:
            audio_module = importlib.import_module("faster_whisper.audio")
            vad_module = importlib.import_module("faster_whisper.vad")
        except ImportError as exc:
            raise TranscriptionError(
                "Faster-Whisper Silero VAD is unavailable; install transcription-faster"
            ) from exc
        return audio_module, vad_module

    def _decode_audio(self, path: Path):
        audio_module, _ = self._dependencies()
        return audio_module.decode_audio(
            str(path), sampling_rate=self.sample_rate_hz
        )

    def _timestamps(self, audio: Any, threshold: float) -> list[dict]:
        _, vad_module = self._dependencies()
        options = vad_module.VadOptions(
            threshold=threshold,
            min_speech_duration_ms=self.minimum_speech_ms,
            min_silence_duration_ms=self.minimum_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            max_speech_duration_s=self.maximum_region_seconds,
        )
        return vad_module.get_speech_timestamps(
            audio,
            options,
            sampling_rate=self.sample_rate_hz,
        )

    def _regions(self, rows: list[dict], source_pass: str) -> tuple[SpeechRegion, ...]:
        regions = []
        for row in rows:
            try:
                start_ms = round(int(row["start"]) * 1000 / self.sample_rate_hz)
                end_ms = round(int(row["end"]) * 1000 / self.sample_rate_hz)
            except (KeyError, TypeError, ValueError) as exc:
                raise TranscriptionError("Silero VAD returned malformed timestamps") from exc
            if end_ms > start_ms:
                regions.append(SpeechRegion(start_ms, end_ms, source_pass))
        return tuple(regions)

    def detect(self, audio_path: str | Path) -> SpeechRegionPlan:
        path = Path(audio_path)
        if not path.is_file():
            raise TranscriptionError(f"speech-region audio input not found: {path}")
        audio = self._decode_audio(path)
        duration_ms = round(len(audio) * 1000 / self.sample_rate_hz)
        strict = self._regions(
            self._timestamps(audio, self.strict_threshold), "strict"
        )
        sensitive = self._regions(
            self._timestamps(audio, self.sensitive_threshold), "sensitive"
        )
        return SpeechRegionPlan(
            detector_identity=self.identity,
            duration_ms=duration_ms,
            strict_regions=strict,
            sensitive_regions=sensitive,
            selected_regions=select_complementary_regions(strict, sensitive),
        )
