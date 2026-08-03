from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from dubbing.transcription.models import TranscriptionError


@dataclass(frozen=True)
class CloudAuthorization:
    recording_sha256: str
    provider: str
    operator_authorization_id: str
    cloud_allowed: bool = False
    failed_spans_only: bool = True


@dataclass(frozen=True)
class CloudSpan:
    path: Path
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


class CloudTransport(Protocol):
    def __call__(self, request: dict, audio: bytes, credential: str) -> dict: ...


@dataclass(frozen=True)
class CloudProviderAdapter:
    name: str
    model: str
    credential_environment_variable: str
    endpoint: str
    retention_mode: str
    request_options: dict

    def build_request(self) -> dict:
        return {
            "provider": self.name,
            "model": self.model,
            "endpoint": self.endpoint,
            "retention_mode": self.retention_mode,
            "options": self.request_options,
        }


PROVIDERS = {
    "deepgram": CloudProviderAdapter(
        "deepgram",
        "nova-3-multilingual",
        "DEEPGRAM_API_KEY",
        "https://api.deepgram.com/v1/listen",
        "provider-default; operator must verify current project retention",
        {"model": "nova-3", "language": "multi", "diarize": True, "smart_format": True},
    ),
    "assemblyai": CloudProviderAdapter(
        "assemblyai",
        "universal-2",
        "ASSEMBLYAI_API_KEY",
        "https://api.assemblyai.com/v2/transcript",
        "provider-default; operator must verify current project retention",
        {"speech_model": "universal-2", "language_detection": True, "speaker_labels": True},
    ),
    "elevenlabs": CloudProviderAdapter(
        "elevenlabs",
        "scribe-v2",
        "ELEVENLABS_API_KEY",
        "https://api.elevenlabs.io/v1/speech-to-text",
        "provider-default; operator must verify current project retention",
        {"model_id": "scribe_v2", "diarize": True, "tag_audio_events": True},
    ),
    "openai": CloudProviderAdapter(
        "openai",
        "gpt-4o-transcribe-diarize",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1/audio/transcriptions",
        "provider-default; operator must verify current project retention",
        {"model": "gpt-4o-transcribe-diarize", "response_format": "diarized_json"},
    ),
}


class CloudAdjudicator:
    """Optional span-only cloud boundary; construction grants no upload authority."""

    def __init__(
        self,
        receipt_dir: str | Path,
        *,
        transport: CloudTransport,
        providers: dict[str, CloudProviderAdapter] | None = None,
    ) -> None:
        self.receipt_dir = Path(receipt_dir)
        self.transport = transport
        self.providers = providers or PROVIDERS

    def adjudicate(
        self,
        span: CloudSpan,
        authorization: CloudAuthorization,
    ) -> tuple[dict, Path]:
        if not authorization.cloud_allowed:
            raise TranscriptionError("cloud_allowed=false; no audio was uploaded")
        if not authorization.operator_authorization_id.strip():
            raise TranscriptionError("explicit operator authorization ID is required")
        if authorization.provider not in self.providers:
            raise TranscriptionError("operator-selected cloud provider is unsupported")
        if not authorization.failed_spans_only:
            raise TranscriptionError("whole-recording cloud upload is disabled by this boundary")
        if not 20_000 <= span.duration_ms <= 60_000:
            raise TranscriptionError("cloud adjudication spans must be between 20 and 60 seconds")
        provider = self.providers[authorization.provider]
        credential = os.environ.get(provider.credential_environment_variable)
        if not credential:
            raise TranscriptionError(
                f"{provider.credential_environment_variable} is not configured locally; no upload occurred"
            )
        request = provider.build_request()
        response = self.transport(request, span.path.read_bytes(), credential)
        receipt = {
            "schema_version": "dubbing.cloud-adjudication-receipt.v1",
            "recording_sha256": authorization.recording_sha256,
            "source_span_sha256": span.sha256,
            "start_ms": span.start_ms,
            "end_ms": span.end_ms,
            "provider": provider.name,
            "model": provider.model,
            "retention_mode": provider.retention_mode,
            "request_configuration": provider.request_options,
            "operator_authorization_id": authorization.operator_authorization_id,
            "request_receipt": response.get("request_receipt"),
            "cost": response.get("cost"),
            "free_credit_consumed": response.get("free_credit_consumed"),
            "candidate_provenance": response.get("provenance"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        path = self.receipt_dir / f"{span.sha256}-{provider.name}.json"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return response["candidate"], path


class OfflineMockTransport:
    """Deterministic contract-test transport; it never performs network I/O."""

    def __init__(self, candidate: dict | None = None) -> None:
        self.candidate = candidate or {"text": "mock candidate", "segments": []}
        self.calls: list[dict] = []

    def __call__(self, request: dict, audio: bytes, credential: str) -> dict:
        self.calls.append({"request": request, "audio_sha256": hashlib.sha256(audio).hexdigest()})
        return {
            "candidate": self.candidate,
            "request_receipt": "offline-mock",
            "cost": 0,
            "free_credit_consumed": 0,
            "provenance": {"transport": "offline-mock"},
        }
