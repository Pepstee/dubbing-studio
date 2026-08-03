from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
from typing import Any

from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.models import (
    DecodeDiagnostics,
    TranscriptSegment,
    TranscriptWord,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MLX_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


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
        temperature: float | tuple[float, ...] = DEFAULT_MLX_TEMPERATURES,
        condition_on_previous_text: bool = False,
        hallucination_silence_threshold: float | None = 2.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be blank")
        temperatures = (temperature,) if isinstance(temperature, (int, float)) else temperature
        if not temperatures or any(value < 0 for value in temperatures):
            raise ValueError("temperature cannot be negative")
        if hallucination_silence_threshold is not None and hallucination_silence_threshold < 0:
            raise ValueError("hallucination_silence_threshold cannot be negative")
        self.model = model
        self.temperature = tuple(float(value) for value in temperatures)
        self.condition_on_previous_text = condition_on_previous_text
        self.hallucination_silence_threshold = hallucination_silence_threshold

    @property
    def identity(self) -> str:
        temperatures = ",".join(f"{value:g}" for value in self.temperature)
        return (
            f"mlx-whisper:{self.model}:temperature={temperatures}:"
            f"condition_on_previous_text={self.condition_on_previous_text}:"
            f"hallucination_silence_threshold={self.hallucination_silence_threshold}"
        )

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
    def _parse_segment(
        cls,
        item: dict,
        *,
        max_temperature: float | None = None,
        fallback_schedule: tuple[float, ...] = (),
    ) -> TranscriptSegment | None:
        text = str(item.get("text", "")).strip()
        if not text:
            return None
        start_ms = _milliseconds(item.get("start"), "segment start")
        end_ms = _milliseconds(item.get("end"), "segment end")
        if end_ms <= start_ms:
            return None
        words = tuple(
            sorted(
                (
                    word
                    for word in (
                        cls._parse_word(raw) for raw in item.get("words", [])
                    )
                    if word is not None
                ),
                key=lambda word: (word.start_ms, word.end_ms),
            )
        )
        temperature = item.get("temperature")
        try:
            parsed_temperature = float(temperature) if temperature is not None else None
        except (TypeError, ValueError):
            parsed_temperature = None
        return TranscriptSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            words=words,
            confidence=_confidence(item),
            diagnostics=DecodeDiagnostics(
                compression_ratio=item.get("compression_ratio"),
                avg_log_probability=item.get("avg_logprob"),
                no_speech_probability=item.get("no_speech_prob"),
                temperature=parsed_temperature,
                fallback_history=tuple(
                    {"configured_temperature": value} for value in fallback_schedule
                ),
                language_probabilities=item.get("language_probs")
                or item.get("language_probabilities"),
                fallback_exhausted=bool(
                    parsed_temperature is not None
                    and max_temperature is not None
                    and parsed_temperature >= max_temperature
                    and len(fallback_schedule) > 1
                ),
                backend_metadata={
                    key: item.get(key)
                    for key in ("seek", "tokens")
                    if key in item
                }
                or None,
            ),
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
            "condition_on_previous_text": self.condition_on_previous_text,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
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
            sorted(
                (
                    segment
                    for segment in (
                        self._parse_segment(
                            item,
                            max_temperature=max(self.temperature),
                            fallback_schedule=self.temperature,
                        )
                        for item in raw.get("segments", [])
                    )
                    if segment is not None
                ),
                key=lambda segment: (segment.start_ms, segment.end_ms),
            )
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
            diagnostics={
                "configured_temperature_fallbacks": list(self.temperature),
                "condition_on_previous_text": self.condition_on_previous_text,
                "hallucination_silence_threshold": self.hallucination_silence_threshold,
            },
            provenance={
                "provider": "mlx-whisper",
                "backend_identity": self.identity,
                "promotion_state": "experimental-fallback",
            },
        )
