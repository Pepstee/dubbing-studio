from __future__ import annotations

from abc import ABC, abstractmethod


class LanguageDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> tuple[str | None, float | None]:
        """Return an ISO-639-1 language and confidence when sufficiently certain."""


class TranslationBackend(ABC):
    @property
    @abstractmethod
    def identity(self) -> str:
        """Stable provider:model identity used in evidence and cache keys."""

    @abstractmethod
    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text without altering or discarding the source."""
