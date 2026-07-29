from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dubbing.transcription.models import (
    TranscriptionOptions,
    TranscriptionResult,
)


class TranscriptionBackend(ABC):
    """Replaceable speech-to-text provider boundary."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Stable backend/model identity used by resumable checkpoints."""

    @abstractmethod
    def transcribe(
        self,
        audio: str | Path,
        options: TranscriptionOptions | None = None,
    ) -> TranscriptionResult:
        """Return a timestamped transcript for one local media file."""
