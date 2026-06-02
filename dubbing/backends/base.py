from __future__ import annotations

from abc import ABC, abstractmethod

from dubbing.models import Segment, TTSResult


class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        ...
