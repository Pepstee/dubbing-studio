from __future__ import annotations

from tts_studio.models import Segment, SpeechResult, TTSBackend


class DubbingPipeline:
    def __init__(self, backend: TTSBackend, default_language: str = "en") -> None:
        self.backend = backend
        self.default_language = default_language

    def run(self, segments: list[Segment]) -> list[SpeechResult]:
        results: list[SpeechResult] = []
        for segment in segments:
            if not segment.language:
                segment.language = self.default_language
            results.append(self.backend.synthesize(segment))
        return results
