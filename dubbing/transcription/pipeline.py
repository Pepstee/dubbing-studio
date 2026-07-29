from __future__ import annotations

from pathlib import Path

from dubbing.diarization.base import DiarizationBackend
from dubbing.diarization.models import SpeakerConstraints
from dubbing.transcription.attribution import attribute_transcript
from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.models import TranscriptionOptions, TranscriptionResult


class AudioUnderstandingPipeline:
    """Compose replaceable ASR and optional diarisation backends."""

    def __init__(self, transcription_backend: TranscriptionBackend) -> None:
        self.transcription_backend = transcription_backend

    def run(
        self,
        audio: str | Path,
        *,
        options: TranscriptionOptions | None = None,
        diarizer: DiarizationBackend | None = None,
        speaker_constraints: SpeakerConstraints | None = None,
    ) -> TranscriptionResult:
        transcript = self.transcription_backend.transcribe(audio, options)
        if diarizer is None:
            return transcript
        diarization = diarizer.diarize(audio, constraints=speaker_constraints)
        return attribute_transcript(transcript, diarization)
