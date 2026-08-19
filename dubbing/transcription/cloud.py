from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from dubbing.media import ffmpeg_executable
from dubbing.transcription.job import source_sha256
from dubbing.transcription.models import (
    DecodeDiagnostics,
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)
from dubbing.transcription.quality import (
    TranscriptQualityStatus,
    evaluate_transcript_quality,
)


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


_UNCERTAIN_PREFIX = "[UNCERTAIN:"
_MINIMUM_PACKET_MS = 20_000
_MAXIMUM_PACKET_MS = 60_000
_LANGUAGE_ALIASES = {"eng": "en", "rus": "ru", "ron": "ro", "kor": "ko"}


def _language_code(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-").split("-", 1)[0]
    return _LANGUAGE_ALIASES.get(normalized, normalized or None)


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
        "provider-default; model-improvement opt-out must be attested before upload",
        {
            "model_id": "scribe_v2",
            "diarize": True,
            "tag_audio_events": False,
            "timestamps_granularity": "word",
            "temperature": 0,
            "seed": 0,
        },
    ),
    "openai": CloudProviderAdapter(
        "openai",
        "gpt-4o-transcribe-diarize",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1/audio/transcriptions",
        "provider-default; operator must verify current project retention",
        {
            "model": "gpt-4o-transcribe-diarize",
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        },
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
        candidate = response.get("candidate")
        if not isinstance(candidate, dict):
            raise TranscriptionError("cloud transport returned no structured candidate")
        provenance = response.get("provenance")
        if not isinstance(provenance, dict):
            raise TranscriptionError("cloud transport returned no candidate provenance")
        request_receipt = response.get("request_receipt")
        if not isinstance(request_receipt, str) or not request_receipt.strip():
            raise TranscriptionError("cloud transport returned no request receipt")
        candidate_sha256 = hashlib.sha256(
            json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
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
            "request_receipt": request_receipt,
            "cost": response.get("cost"),
            "free_credit_consumed": response.get("free_credit_consumed"),
            "candidate_sha256": candidate_sha256,
            "candidate_provenance": provenance,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        path = self.receipt_dir / f"{span.sha256}-{provider.name}.json"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return candidate, path


class OpenAIHTTPTransport:
    """Small stdlib transport for the official audio-transcriptions endpoint.

    The transport deliberately accepts only the OpenAI adapter. Provider-specific
    transports can implement the same callable contract without changing policy code.
    """

    def __init__(self, *, timeout_seconds: int = 120) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _multipart(request: dict, audio: bytes) -> tuple[bytes, str]:
        boundary = f"dubbing-{uuid.uuid4().hex}"
        body = bytearray()

        def field(name: str, value: object) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            )
            if isinstance(value, bool):
                value = str(value).lower()
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        for name, value in request["options"].items():
            if value is not None:
                field(name, value)
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            b'Content-Disposition: form-data; name="file"; filename="span.wav"\r\n'
        )
        body.extend(b"Content-Type: audio/wav\r\n\r\n")
        body.extend(audio)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        return bytes(body), boundary

    @staticmethod
    def _candidate(document: dict) -> dict:
        raw_segments = document.get("segments")
        if not isinstance(raw_segments, list):
            raise TranscriptionError(
                "OpenAI diarized response did not contain timestamped segments"
            )
        segments = []
        for item in raw_segments:
            if not isinstance(item, dict):
                raise TranscriptionError("OpenAI returned a malformed diarized segment")
            try:
                start_ms = round(float(item["start"]) * 1000)
                end_ms = round(float(item["end"]) * 1000)
                text = str(item["text"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise TranscriptionError(
                    "OpenAI returned a malformed diarized segment"
                ) from exc
            if start_ms < 0 or end_ms <= start_ms or not text:
                raise TranscriptionError("OpenAI returned invalid segment content")
            segments.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "speaker": item.get("speaker"),
                }
            )
        return {
            "text": str(document.get("text", "")).strip()
            or " ".join(item["text"] for item in segments),
            "segments": segments,
        }

    def __call__(self, request: dict, audio: bytes, credential: str) -> dict:
        if request.get("provider") != "openai":
            raise TranscriptionError("OpenAI transport cannot call another provider")
        body, boundary = self._multipart(request, audio)
        http_request = urllib.request.Request(
            request["endpoint"],
            data=body,
            headers={
                "Authorization": f"Bearer {credential}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "dubbing-studio/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.timeout_seconds
            ) as response:
                document = json.loads(response.read().decode("utf-8"))
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise TranscriptionError(
                f"OpenAI transcription request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise TranscriptionError("OpenAI transcription request failed") from exc
        return {
            "candidate": self._candidate(document),
            "request_receipt": request_id,
            "cost": None,
            "free_credit_consumed": None,
            "provenance": {
                "transport": "openai-audio-transcriptions-http",
                "response_format": request["options"].get("response_format"),
            },
        }


class ElevenLabsHTTPTransport:
    """Official synchronous Scribe v2 multipart transport.

    The provider response is normalized into the same timestamped candidate shape
    used by the existing cloud policy boundary. Credentials are accepted only by
    the callable interface and are never serialized into receipts.
    """

    def __init__(self, *, timeout_seconds: int = 7_200) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _multipart(request: dict, audio: bytes) -> tuple[bytes, str]:
        boundary = f"dubbing-{uuid.uuid4().hex}"
        body = bytearray()

        def field(name: str, value: object) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            )
            if isinstance(value, bool):
                value = str(value).lower()
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        for name, value in request["options"].items():
            if value is not None:
                field(name, value)
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            b'Content-Disposition: form-data; name="file"; filename="recording.flac"\r\n'
        )
        body.extend(b"Content-Type: audio/flac\r\n\r\n")
        body.extend(audio)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        return bytes(body), boundary

    @staticmethod
    def _candidate(document: dict) -> dict:
        raw_words = document.get("words")
        if not isinstance(raw_words, list):
            raise TranscriptionError(
                "ElevenLabs Scribe response did not contain timestamped words"
            )
        segments = []
        for item in raw_words:
            if not isinstance(item, dict) or item.get("type", "word") != "word":
                continue
            try:
                start_ms = round(float(item["start"]) * 1000)
                end_ms = round(float(item["end"]) * 1000)
                value = str(item["text"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise TranscriptionError(
                    "ElevenLabs returned a malformed timestamped word"
                ) from exc
            if start_ms < 0 or end_ms <= start_ms or not value:
                raise TranscriptionError("ElevenLabs returned invalid word content")
            segments.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": value,
                    "speaker": item.get("speaker_id"),
                    "language": _language_code(
                        item.get("language_code") or document.get("language_code")
                    ),
                    "confidence": (
                        None
                        if item.get("logprob") is None
                        else min(1.0, max(0.0, pow(2.718281828, float(item["logprob"]))))
                    ),
                }
            )
        if not segments:
            raise TranscriptionError("ElevenLabs returned no timestamped speech words")
        return {
            "text": str(document.get("text", "")).strip()
            or " ".join(item["text"] for item in segments),
            "segments": segments,
            "language": _language_code(document.get("language_code")),
            "language_probability": document.get("language_probability"),
        }

    def __call__(self, request: dict, audio: bytes, credential: str) -> dict:
        if request.get("provider") != "elevenlabs":
            raise TranscriptionError("ElevenLabs transport cannot call another provider")
        body, boundary = self._multipart(request, audio)
        http_request = urllib.request.Request(
            request["endpoint"],
            data=body,
            headers={
                "xi-api-key": credential,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "dubbing-studio/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.timeout_seconds
            ) as response:
                response_bytes = response.read()
                document = json.loads(response_bytes.decode("utf-8"))
                request_id = response.headers.get("request-id") or response.headers.get(
                    "x-request-id"
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise TranscriptionError(
                f"ElevenLabs transcription request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise TranscriptionError("ElevenLabs transcription request failed") from exc
        response_sha256 = hashlib.sha256(response_bytes).hexdigest()
        return {
            "candidate": self._candidate(document),
            "request_receipt": request_id or f"response-sha256:{response_sha256}",
            "cost": None,
            "free_credit_consumed": None,
            "provenance": {
                "transport": "elevenlabs-speech-to-text-http",
                "timestamps_granularity": request["options"].get(
                    "timestamps_granularity"
                ),
                "response_sha256": response_sha256,
            },
        }

def unresolved_intervals(result: TranscriptionResult) -> tuple[tuple[int, int], ...]:
    """Return merged source intervals that remain explicitly uncertain."""

    intervals = [
        (segment.start_ms, segment.end_ms)
        for segment in result.segments
        if segment.uncertain or segment.text.startswith(_UNCERTAIN_PREFIX)
    ]
    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in sorted(intervals):
        if merged and start_ms <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    bounded = []
    for start_ms, end_ms in merged:
        cursor = start_ms
        while end_ms - cursor > _MAXIMUM_PACKET_MS:
            bounded.append((cursor, cursor + _MAXIMUM_PACKET_MS))
            cursor += _MAXIMUM_PACKET_MS
        if end_ms > cursor:
            bounded.append((cursor, end_ms))
    return tuple(bounded)


def context_packet_interval(
    target: tuple[int, int], *, duration_ms: int
) -> tuple[int, int]:
    """Build the minimum legal cloud packet around an unresolved target."""

    start_ms, end_ms = target
    if not 0 <= start_ms < end_ms <= duration_ms:
        raise TranscriptionError("cloud target is outside the source recording")
    if duration_ms < _MINIMUM_PACKET_MS:
        raise TranscriptionError(
            "recording is shorter than the minimum cloud adjudication packet"
        )
    target_duration = end_ms - start_ms
    if target_duration > _MAXIMUM_PACKET_MS:
        raise TranscriptionError("unresolved target exceeds the maximum cloud packet")
    packet_duration = max(_MINIMUM_PACKET_MS, target_duration)
    centre = start_ms + target_duration // 2
    packet_start = max(0, centre - packet_duration // 2)
    packet_end = min(duration_ms, packet_start + packet_duration)
    packet_start = max(0, packet_end - packet_duration)
    return packet_start, packet_end


def plan_context_packets(
    targets: tuple[tuple[int, int], ...], *, duration_ms: int
) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]:
    """Coalesce overlapping context windows so the same audio is never uploaded twice."""

    planned: list[list] = []
    for target in targets:
        packet_start, packet_end = context_packet_interval(
            target, duration_ms=duration_ms
        )
        if (
            planned
            and packet_start <= planned[-1][1]
            and max(packet_end, planned[-1][1]) - planned[-1][0]
            <= _MAXIMUM_PACKET_MS
        ):
            planned[-1][1] = max(packet_end, planned[-1][1])
            planned[-1][2].append(target)
        else:
            planned.append([packet_start, packet_end, [target]])
    return tuple(
        (start_ms, end_ms, tuple(packet_targets))
        for start_ms, end_ms, packet_targets in planned
    )


def _extract_packet(
    source: Path, *, start_ms: int, end_ms: int, destination: Path
) -> None:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise TranscriptionError("ffmpeg is required for cloud packet extraction")
    process = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(source),
            "-t",
            f"{(end_ms - start_ms) / 1000:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=max(120, (end_ms - start_ms) // 1000 + 60),
    )
    if process.returncode:
        raise TranscriptionError(
            f"cloud packet extraction failed: {process.stderr.strip()}"
        )


def _candidate_target_segments(
    candidate: dict,
    *,
    packet_start_ms: int,
    target: tuple[int, int],
    provider: str,
    model: str,
) -> tuple[TranscriptSegment, ...]:
    raw_segments = candidate.get("segments")
    if not isinstance(raw_segments, list):
        raise TranscriptionError("cloud candidate must contain timestamped segments")
    target_start, target_end = target
    admitted = []
    for item in raw_segments:
        try:
            local_start = int(item["start_ms"])
            local_end = int(item["end_ms"])
            text = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError("cloud candidate segment is malformed") from exc
        source_start = packet_start_ms + local_start
        source_end = packet_start_ms + local_end
        if source_end <= target_start or source_start >= target_end:
            continue
        start_ms = max(target_start, source_start)
        end_ms = min(target_end, source_end)
        if end_ms <= start_ms or not text or text.startswith(_UNCERTAIN_PREFIX):
            continue
        admitted.append(
            TranscriptSegment(
                start_ms,
                end_ms,
                text,
                speaker=item.get("speaker"),
                speaker_status=(
                    "attributed" if item.get("speaker") else "not_requested"
                ),
                uncertain=False,
                diagnostics=DecodeDiagnostics(
                    backend_metadata={
                        "cloud_adjudication": True,
                        "provider": provider,
                        "model": model,
                    }
                ),
            )
        )
    return tuple(sorted(admitted, key=lambda item: (item.start_ms, item.end_ms)))


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class TargetedCloudAdjudication:
    """Automatically packetize and adjudicate only unresolved transcript intervals."""

    def __init__(self, adjudicator: CloudAdjudicator, output_dir: str | Path) -> None:
        self.adjudicator = adjudicator
        self.output_dir = Path(output_dir)

    def run(
        self,
        source: str | Path,
        local_result: TranscriptionResult,
        authorization: CloudAuthorization,
    ) -> tuple[TranscriptionResult, dict]:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise TranscriptionError(f"media input not found: {source_path}")
        digest = source_sha256(source_path)
        if digest != authorization.recording_sha256:
            raise TranscriptionError("cloud authorization does not match source recording")
        if local_result.source_sha256 != digest:
            raise TranscriptionError("local transcript does not match source recording")
        duration_ms = local_result.duration_ms
        if duration_ms is None or duration_ms <= 0:
            raise TranscriptionError("local transcript must preserve source duration")
        targets = unresolved_intervals(local_result)
        if not targets:
            raise TranscriptionError("transcript has no unresolved intervals to adjudicate")

        replacements: list[TranscriptSegment] = []
        decisions = []
        provider = self.adjudicator.providers.get(authorization.provider)
        if provider is None:
            raise TranscriptionError("operator-selected cloud provider is unsupported")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dubbing-cloud-adjudication-") as directory:
            temporary_dir = Path(directory)
            packets = plan_context_packets(targets, duration_ms=duration_ms)
            for index, (packet_start, packet_end, packet_targets) in enumerate(packets):
                packet = temporary_dir / f"{index:06d}.wav"
                _extract_packet(
                    source_path,
                    start_ms=packet_start,
                    end_ms=packet_end,
                    destination=packet,
                )
                candidate, receipt_path = self.adjudicator.adjudicate(
                    CloudSpan(packet, packet_start, packet_end), authorization
                )
                packet_sha256 = hashlib.sha256(packet.read_bytes()).hexdigest()
                for target in packet_targets:
                    candidate_segments = _candidate_target_segments(
                        candidate,
                        packet_start_ms=packet_start,
                        target=target,
                        provider=provider.name,
                        model=provider.model,
                    )
                    status = (
                        "ADMITTED" if candidate_segments else "REJECTED_EMPTY_TARGET"
                    )
                    if candidate_segments:
                        replacements.extend(candidate_segments)
                    decisions.append(
                        {
                            "target_start_ms": target[0],
                            "target_end_ms": target[1],
                            "packet_index": index,
                            "packet_target_count": len(packet_targets),
                            "packet_start_ms": packet_start,
                            "packet_end_ms": packet_end,
                            "packet_sha256": packet_sha256,
                            "receipt": str(receipt_path),
                            "status": status,
                            "candidate_segments": [
                                item.to_dict() for item in candidate_segments
                            ],
                        }
                    )

        retained = [
            segment
            for segment in local_result.segments
            if not any(
                segment.start_ms < end_ms and segment.end_ms > start_ms
                for start_ms, end_ms in targets
            )
        ]
        unresolved_targets = {
            (item["target_start_ms"], item["target_end_ms"])
            for item in decisions
            if item["status"] != "ADMITTED"
        }
        retained.extend(
            segment
            for segment in local_result.segments
            if any(
                segment.start_ms < end_ms and segment.end_ms > start_ms
                for start_ms, end_ms in unresolved_targets
            )
        )
        segments = tuple(
            sorted(retained + replacements, key=lambda item: (item.start_ms, item.end_ms))
        )
        result = TranscriptionResult(
            segments=segments,
            text=" ".join(item.text for item in segments),
            backend=local_result.backend,
            model=local_result.model,
            device=local_result.device,
            language=local_result.language,
            duration_ms=duration_ms,
            confidence_available=local_result.confidence_available,
            source_sha256=digest,
            diarization=local_result.diarization,
            diagnostics=local_result.diagnostics,
            provenance={
                **(local_result.provenance or {}),
                "targeted_cloud_adjudication": {
                    "provider": provider.name,
                    "model": provider.model,
                    "operator_authorization_id": authorization.operator_authorization_id,
                    "target_count": len(targets),
                },
            },
        )
        quality = evaluate_transcript_quality(result, expected_duration_ms=duration_ms)
        admitted = (
            all(item["status"] == "ADMITTED" for item in decisions)
            and quality.status is TranscriptQualityStatus.PASS
        )
        report = {
            "schema_version": "dubbing.targeted-cloud-adjudication.v1",
            "recording_sha256": digest,
            "provider": provider.name,
            "model": provider.model,
            "operator_authorization_id": authorization.operator_authorization_id,
            "failed_spans_only": authorization.failed_spans_only,
            "decisions": decisions,
            "quality": quality.to_dict(),
            "admission_status": "PASS" if admitted else "FAIL_CLOSED",
            "giga_admission_allowed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(self.output_dir / "result.json", result.to_dict())
        _atomic_json(self.output_dir / "adjudication-report.json", report)
        return result, report


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
