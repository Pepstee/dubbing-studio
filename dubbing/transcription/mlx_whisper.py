from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
from typing import Any

from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"


def _milliseconds(value: Any, field: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TranscriptionError(f"MLX Whisper returned invalid {field}") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise TranscriptionError(f"MLX Whisper returned invalid {field}")
    return round(numeric * 1000)


def _confidence(item: dict) -> float | None:
    probability = item.get("probability")
    if probability is not None:
        try:
            value = float(probability)
        except (TypeError, ValueError):
            return None
        return value if 0.0 <= value <= 1.0 else None
    average_logprob = item.get("avg_logprob")
    if average_logprob is None:
        return None
    try:
        value = math.exp(float(average_logprob))
    except (TypeError, ValueError, OverflowError):
        return None
    return min(1.0, max(0.0, value))


class MLXWhisperTranscriptionBackend(TranscriptionBackend):
    """Local Apple-Silicon ASR through ``mlx-whisper``."""

    def __init__(
        self,
        model: str = DEFAULT_MLX_MODEL,
        *,
        temperature: float = 0.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be blank")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        self.model = model
        self.temperature = temperature

    @property
    def identity(self) -> str:
        return f"mlx-whisper:{self.model}"

    @staticmethod
    def _dependency():
        try:
            return importlib.import_module("mlx_whisper")
        except ImportError as exc:
            raise TranscriptionError(
                "MLX Whisper is not installed. On Apple Silicon install with "
                "`pip install 'dubbing-studio[transcription-mlx]'`."
            ) from exc

    @staticmethod
    def _source_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _parse_word(item: dict) -> TranscriptWord | None:
        text = str(item.get("word", item.get("text", "")))
        if not text:
            return None
        start_ms = _milliseconds(item.get("start"), "word start")
        end_ms = _milliseconds(item.get("end"), "word end")
        if end_ms <= start_ms:
            return None
        return TranscriptWord(
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            confidence=_confidence(item),
        )

    @classmethod
    def _parse_segment(cls, item: dict) -> TranscriptSegment | None:
        text = str(item.get("text", "")).strip()
        if not text:
            return None
        start_ms = _milliseconds(item.get("start"), "segment start")
        end_ms = _milliseconds(item.get("end"), "segment end")
        if end_ms <= start_ms:
            return None
        words = tuple(
            word
            for word in (cls._parse_word(raw) for raw in item.get("words", []))
            if word is not None
        )
        return TranscriptSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            words=words,
            confidence=_confidence(item),
        )

    def transcribe(
        self,
        audio: str | Path,
        options: TranscriptionOptions | None = None,
    ) -> TranscriptionResult:
        path = Path(audio)
        if not path.is_file():
            raise TranscriptionError(f"audio input not found: {path}")
        options = options or TranscriptionOptions()
        mlx_whisper = self._dependency()
        kwargs: dict[str, Any] = {
            "path_or_hf_repo": self.model,
            "task": options.task,
            "word_timestamps": options.word_timestamps,
            "verbose": False,
            "temperature": self.temperature,
        }
        if options.language:
            kwargs["language"] = options.language
        if options.initial_prompt:
            kwargs["initial_prompt"] = options.initial_prompt
        try:
            raw = mlx_whisper.transcribe(str(path), **kwargs)
        except Exception as exc:
            raise TranscriptionError(f"MLX Whisper transcription failed: {exc}") from exc
        if not isinstance(raw, dict):
            raise TranscriptionError("MLX Whisper returned a non-object result")
        segments = tuple(
            segment
            for segment in (
                self._parse_segment(item) for item in raw.get("segments", [])
            )
            if segment is not None
        )
        text = str(raw.get("text", "")).strip()
        if not text and segments:
            text = " ".join(segment.text for segment in segments)
        duration_ms = max((segment.end_ms for segment in segments), default=None)
        confidence_available = any(
            segment.confidence is not None
            or any(word.confidence is not None for word in segment.words)
            for segment in segments
        )
        return TranscriptionResult(
            segments=segments,
            text=text,
            backend="mlx-whisper",
            model=self.model,
            device="apple-silicon",
            language=raw.get("language") or options.language,
            duration_ms=duration_ms,
            confidence_available=confidence_available,
            source_sha256=self._source_hash(path),
        )
