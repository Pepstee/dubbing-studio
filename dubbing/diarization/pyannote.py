from __future__ import annotations

import importlib
import os
from pathlib import Path

from dubbing.diarization.base import DiarizationBackend
from dubbing.diarization.models import (
    DiarizationError,
    DiarizationResult,
    SpeakerConstraints,
    SpeakerTurn,
)


class PyannoteCommunityBackend(DiarizationBackend):
    """Preferred local reference diarizer; model is loaded once and reused."""

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-community-1",
        *,
        device: str = "auto",
        token_environment_variable: str = "HF_TOKEN",
    ) -> None:
        self.model = model
        self.device = device
        self.token_environment_variable = token_environment_variable
        self._pipeline = None

    @property
    def identity(self) -> str:
        return f"pyannote-community-1:{self.model}:{self.device}"

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            module = importlib.import_module("pyannote.audio")
        except ImportError as exc:
            raise DiarizationError(
                "pyannote.audio is not installed; install the optional diarization-pyannote extra"
            ) from exc
        token = os.environ.get(self.token_environment_variable)
        is_local = Path(self.model).exists()
        if not is_local and not token:
            raise DiarizationError(
                "Community-1 requires accepted model terms and a locally configured HF_TOKEN; "
                "no credential or download was requested."
            )
        try:
            self._pipeline = module.Pipeline.from_pretrained(
                self.model, token=None if is_local else token
            )
            if self.device != "cpu":
                torch = importlib.import_module("torch")
                selected = (
                    "cuda"
                    if self.device == "auto" and torch.cuda.is_available()
                    else ("mps" if self.device == "auto" and torch.backends.mps.is_available() else self.device)
                )
                if selected not in {"auto", "cpu"}:
                    self._pipeline.to(torch.device(selected))
        except Exception as exc:
            raise DiarizationError(f"could not load pyannote Community-1: {exc}") from exc
        return self._pipeline

    def diarize(
        self,
        audio: str | Path,
        constraints: SpeakerConstraints | None = None,
    ) -> DiarizationResult:
        path = Path(audio)
        if not path.is_file():
            raise DiarizationError(f"audio input not found: {path}")
        constraints = constraints or SpeakerConstraints()
        kwargs = {
            key: value
            for key, value in {
                "num_speakers": constraints.num_speakers,
                "min_speakers": constraints.min_speakers,
                "max_speakers": constraints.max_speakers,
            }.items()
            if value is not None
        }
        try:
            output = self._load()(str(path), **kwargs)
            annotation = getattr(output, "speaker_diarization", output)
            turns = tuple(
                SpeakerTurn(
                    round(turn.start * 1000),
                    round(turn.end * 1000),
                    str(speaker),
                )
                for turn, _, speaker in annotation.itertracks(yield_label=True)
                if turn.end > turn.start
            )
        except Exception as exc:
            raise DiarizationError(f"pyannote diarization failed: {exc}") from exc
        return DiarizationResult(
            turns=tuple(sorted(turns, key=lambda item: (item.start_ms, item.end_ms, item.speaker))),
            backend="pyannote-community-1",
            model=self.model,
            device=self.device,
            confidence_available=False,
        )
