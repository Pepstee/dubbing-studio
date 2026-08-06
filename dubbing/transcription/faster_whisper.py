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
from dubbing.transcription.windows_cuda import configure_nvidia_dlls

DEFAULT_FASTER_WHISPER_MODEL = "large-v3-turbo"


def _milliseconds(value: Any, field: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TranscriptionError(f"Faster-Whisper returned invalid {field}") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise TranscriptionError(f"Faster-Whisper returned invalid {field}")
    return round(numeric * 1000)


def _probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    return probability if 0.0 <= probability <= 1.0 else None


class FasterWhisperTranscriptionBackend(TranscriptionBackend):
    """Local cross-platform ASR through Faster-Whisper and CTranslate2."""

    def __init__(
        self,
        model: str = DEFAULT_FASTER_WHISPER_MODEL,
        *,
        device: str = "auto",
        compute_type: str = "default",
        temperature: float = 0.0,
        cpu_threads: int = 0,
        num_workers: int = 1,
        local_files_only: bool = False,
        model_revision: str | None = None,
        multilingual: bool = True,
        condition_on_previous_text: bool = False,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be blank")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not compute_type.strip():
            raise ValueError("compute_type cannot be blank")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if cpu_threads < 0:
            raise ValueError("cpu_threads cannot be negative")
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.temperature = temperature
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.local_files_only = local_files_only
        self.model_revision = model_revision
        self.multilingual = multilingual
        self.condition_on_previous_text = condition_on_previous_text
        self._model_instance = None
        self._observed_model_sha256: str | None = None

    def _model_sha256(self) -> str | None:
        if self._observed_model_sha256 is not None:
            return self._observed_model_sha256
        model_bin = Path(self.model) / "model.bin"
        if not model_bin.is_file():
            return None
        self._observed_model_sha256 = self._source_hash(model_bin)
        return self._observed_model_sha256

    @property
    def identity(self) -> str:
        model_fingerprint = self.model_revision or self._model_sha256() or "unversioned"
        return (
            f"faster-whisper:{self.model}:{self.device}:{self.compute_type}:"
            f"{self.temperature}:{self.cpu_threads}:{self.num_workers}:"
            f"local={self.local_files_only}:revision={model_fingerprint}:"
            f"multilingual={self.multilingual}:condition_previous={self.condition_on_previous_text}"
        )

    @staticmethod
    def _dependency():
        configure_nvidia_dlls()
        try:
            return importlib.import_module("faster_whisper")
        except ImportError as exc:
            raise TranscriptionError(
                "Faster-Whisper is not installed. Install with "
                "`pip install 'dubbing-studio[transcription-faster]'`."
            ) from exc

    @staticmethod
    def _source_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load_model(self):
        if self._model_instance is None:
            if self.local_files_only and not Path(self.model).is_dir():
                raise TranscriptionError(
                    f"offline Faster-Whisper model directory not found: {self.model}"
                )
            dependency = self._dependency()
            try:
                self._model_instance = dependency.WhisperModel(
                    self.model,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                )
            except Exception as exc:
                raise TranscriptionError(
                    f"Faster-Whisper could not load model {self.model!r} on "
                    f"{self.device}: {exc}"
                ) from exc
        return self._model_instance

    @staticmethod
    def _parse_word(item: Any) -> TranscriptWord | None:
        text = str(getattr(item, "word", ""))
        if not text:
            return None
        start_ms = _milliseconds(getattr(item, "start", None), "word start")
        end_ms = _milliseconds(getattr(item, "end", None), "word end")
        if end_ms <= start_ms:
            return None
        return TranscriptWord(
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            confidence=_probability(getattr(item, "probability", None)),
        )

    @classmethod
    def _parse_segment(cls, item: Any) -> TranscriptSegment | None:
        text = str(getattr(item, "text", "")).strip()
        if not text:
            return None
        start_ms = _milliseconds(getattr(item, "start", None), "segment start")
        end_ms = _milliseconds(getattr(item, "end", None), "segment end")
        if end_ms <= start_ms:
            return None
        words = tuple(
            word
            for word in (
                cls._parse_word(raw) for raw in (getattr(item, "words", None) or ())
            )
            if word is not None
        )
        average_logprob = getattr(item, "avg_logprob", None)
        confidence = None
        if average_logprob is not None:
            try:
                confidence = min(1.0, max(0.0, math.exp(float(average_logprob))))
            except (TypeError, ValueError, OverflowError):
                confidence = None
        return TranscriptSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            words=words,
            confidence=confidence,
            diagnostics=DecodeDiagnostics(
                compression_ratio=getattr(item, "compression_ratio", None),
                avg_log_probability=(
                    float(average_logprob) if average_logprob is not None else None
                ),
                no_speech_probability=getattr(item, "no_speech_prob", None),
                temperature=getattr(item, "temperature", None),
                fallback_exhausted=bool(getattr(item, "fallback_exhausted", False)),
                backend_metadata={
                    "seek": getattr(item, "seek", None),
                    "tokens": list(getattr(item, "tokens", ()) or ()),
                },
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
        model = self._load_model()
        kwargs: dict[str, Any] = {
            "task": options.task,
            "word_timestamps": options.word_timestamps,
            "temperature": self.temperature,
            "multilingual": self.multilingual and options.language is None,
            "condition_on_previous_text": self.condition_on_previous_text,
        }
        if options.language:
            kwargs["language"] = options.language
        if options.initial_prompt:
            kwargs["initial_prompt"] = options.initial_prompt
        try:
            raw_segments, info = model.transcribe(str(path), **kwargs)
            segments = tuple(
                segment
                for segment in (self._parse_segment(item) for item in raw_segments)
                if segment is not None
            )
        except Exception as exc:
            raise TranscriptionError(f"Faster-Whisper transcription failed: {exc}") from exc
        text = " ".join(segment.text for segment in segments).strip()
        duration = getattr(info, "duration", None)
        duration_ms = (
            _milliseconds(duration, "duration")
            if duration is not None
            else max((segment.end_ms for segment in segments), default=None)
        )
        confidence_available = any(
            segment.confidence is not None
            or any(word.confidence is not None for word in segment.words)
            for segment in segments
        )
        return TranscriptionResult(
            segments=segments,
            text=text,
            backend="faster-whisper",
            model=self.model,
            device=self.device,
            language=getattr(info, "language", None) or options.language,
            duration_ms=duration_ms,
            confidence_available=confidence_available,
            source_sha256=self._source_hash(path),
            diagnostics={
                "language_probability": getattr(info, "language_probability", None),
                "duration_after_vad": getattr(info, "duration_after_vad", None),
                "vad_options": getattr(info, "vad_options", None),
            },
            provenance={
                "provider": "faster-whisper",
                "backend_identity": self.identity,
                "persistent_model_instance": True,
                "model_revision": self.model_revision,
                "model_bin_sha256": self._model_sha256(),
                "local_files_only": self.local_files_only,
                "multilingual_segment_detection": kwargs["multilingual"],
                "condition_on_previous_text": self.condition_on_previous_text,
            },
        )
