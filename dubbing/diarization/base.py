from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dubbing.diarization.models import DiarizationResult, SpeakerConstraints


class DiarizationBackend(ABC):
    """Interface for local or remote speaker diarization implementations."""

    @abstractmethod
    def diarize(
        self,
        audio: str | Path,
        constraints: SpeakerConstraints | None = None,
    ) -> DiarizationResult:
        """Return speaker turns for an audio or video file."""
