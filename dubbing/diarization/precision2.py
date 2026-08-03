from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from dubbing.diarization.base import DiarizationBackend
from dubbing.diarization.models import (
    DiarizationError,
    DiarizationResult,
    SpeakerConstraints,
    SpeakerTurn,
)


class Precision2Transport(Protocol):
    def __call__(self, audio: bytes, credential: str, options: dict) -> dict: ...


@dataclass(frozen=True)
class Precision2Authorization:
    recording_sha256: str
    operator_authorization_id: str
    cloud_allowed: bool = False


class Precision2Backend(DiarizationBackend):
    """Optional pyannoteAI provider boundary; disabled until per-recording authorization."""

    def __init__(
        self,
        authorization: Precision2Authorization,
        *,
        transport: Precision2Transport,
        receipt_dir: str | Path,
        credential_environment_variable: str = "PYANNOTEAI_API_KEY",
    ) -> None:
        self.authorization = authorization
        self.transport = transport
        self.receipt_dir = Path(receipt_dir)
        self.credential_environment_variable = credential_environment_variable

    @property
    def identity(self) -> str:
        return "pyannote-precision-2:remote:explicit-authorization"

    def diarize(
        self,
        audio: str | Path,
        constraints: SpeakerConstraints | None = None,
    ) -> DiarizationResult:
        path = Path(audio)
        if not self.authorization.cloud_allowed:
            raise DiarizationError("Precision-2 cloud_allowed=false; no audio was uploaded")
        if not self.authorization.operator_authorization_id.strip():
            raise DiarizationError("Precision-2 requires explicit operator authorization")
        credential = os.environ.get(self.credential_environment_variable)
        if not credential:
            raise DiarizationError(
                f"{self.credential_environment_variable} is not configured locally; no upload occurred"
            )
        constraints = constraints or SpeakerConstraints()
        options = {
            "num_speakers": constraints.num_speakers,
            "min_speakers": constraints.min_speakers,
            "max_speakers": constraints.max_speakers,
        }
        audio_bytes = path.read_bytes()
        response = self.transport(audio_bytes, credential, options)
        turns = tuple(
            SpeakerTurn(
                round(float(item["start"]) * 1000),
                round(float(item["end"]) * 1000),
                str(item["speaker"]),
                item.get("confidence"),
            )
            for item in response.get("turns", [])
        )
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(audio_bytes).hexdigest()
        receipt = {
            "schema_version": "dubbing.remote-diarization-receipt.v1",
            "recording_sha256": self.authorization.recording_sha256,
            "uploaded_audio_sha256": digest,
            "provider": "pyannoteAI",
            "model": "precision-2",
            "operator_authorization_id": self.authorization.operator_authorization_id,
            "retention_mode": response.get("retention_mode", "provider-configuration-unverified"),
            "request_receipt": response.get("request_receipt"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        destination = self.receipt_dir / f"{digest}-precision-2.json"
        destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return DiarizationResult(
            turns=tuple(sorted(turns, key=lambda item: (item.start_ms, item.end_ms, item.speaker))),
            backend="pyannoteAI",
            model="precision-2",
            device="remote-authorized",
            confidence_available=any(item.confidence is not None for item in turns),
        )
