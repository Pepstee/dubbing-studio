from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.models import (
    DecodeDiagnostics,
    TranscriptSegment,
    TranscriptWord,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
)


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_tree_fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(block)
                digest.update(block)
    return digest.hexdigest(), total


def _loopback_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("WhisperKit server URL must use local loopback HTTP")
    return value.rstrip("/")


class WhisperKitServerProcess:
    """Own one supported Argmax local-server process for many transcription calls."""

    def __init__(
        self,
        *,
        executable: str = "argmax-cli",
        model_path: str | Path,
        host: str = "127.0.0.1",
        port: int = 50060,
        startup_timeout_seconds: float = 120.0,
    ) -> None:
        resolved = shutil.which(executable) if Path(executable).name == executable else executable
        if not resolved or not Path(resolved).is_file():
            raise TranscriptionError(
                "WhisperKit CLI is not installed. Install the official argmax-cli/"
                "whisperkit-cli boundary, then retry; no model download was attempted."
            )
        self.executable = str(Path(resolved).resolve())
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise TranscriptionError(f"WhisperKit model directory not found: {self.model_path}")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("WhisperKit server must bind to loopback")
        if not 1 <= port <= 65535:
            raise ValueError("invalid WhisperKit server port")
        self.host = host
        self.port = port
        self.startup_timeout_seconds = startup_timeout_seconds
        version = subprocess.run(
            [self.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.version = (version.stdout or version.stderr).strip() or "unknown"
        self.model_sha256, self.model_size_bytes = _model_tree_fingerprint(self.model_path)
        self.process: subprocess.Popen | None = None
        atexit.register(self.close)

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        command = [
            self.executable,
            "serve",
            "--model-path",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--word-timestamps",
            "--temperature",
            "0",
            "--temperature-fallback-count",
            "0",
            "--chunking-strategy",
            "vad",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                detail = (self.process.stderr.read() if self.process.stderr else "").strip()
                raise TranscriptionError(f"WhisperKit server exited during startup: {detail}")
            try:
                with socket.create_connection((self.host, self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.25)
        self.close()
        raise TranscriptionError("WhisperKit local server did not become ready in time")

    def close(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


class WhisperKitTranscriptionBackend(TranscriptionBackend):
    """Official WhisperKit OpenAI-compatible local-server adapter."""

    def __init__(
        self,
        *,
        model: str,
        endpoint: str = "http://127.0.0.1:50060",
        server: WhisperKitServerProcess | None = None,
        request_timeout_seconds: int = 900,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be blank")
        self.model = model
        self.endpoint = _loopback_url(endpoint)
        self.server = server
        self.request_timeout_seconds = request_timeout_seconds

    @property
    def identity(self) -> str:
        return f"whisperkit-local-server:{self.model}:{self.endpoint}"

    @staticmethod
    def _multipart(path: Path, fields: list[tuple[str, str]]) -> tuple[bytes, str]:
        boundary = f"dubbing-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields:
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode("utf-8"),
                    b"\r\n",
                ]
            )
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; name="file"; '
                    f'filename="{path.name}"\r\n'
                ).encode(),
                b"Content-Type: application/octet-stream\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        return b"".join(chunks), boundary

    @staticmethod
    def _diagnostics(item: dict) -> DecodeDiagnostics:
        metadata = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "id",
                "seek",
                "start",
                "end",
                "text",
                "tokens",
                "words",
            }
        }
        history = item.get("fallback_history") or item.get("fallbacks") or []
        return DecodeDiagnostics(
            compression_ratio=item.get("compression_ratio"),
            avg_log_probability=item.get("avg_logprob", item.get("avg_log_probability")),
            no_speech_probability=item.get("no_speech_prob", item.get("no_speech_probability")),
            temperature=item.get("temperature"),
            fallback_history=tuple(history) if isinstance(history, list) else (),
            language_probabilities=item.get("language_probabilities"),
            fallback_exhausted=bool(item.get("fallback_exhausted", False)),
            backend_metadata=metadata or None,
        )

    @classmethod
    def _parse_segment(cls, item: dict, language: str | None) -> TranscriptSegment | None:
        text = str(item.get("text", "")).strip()
        start_ms = round(float(item.get("start", 0)) * 1000)
        end_ms = round(float(item.get("end", 0)) * 1000)
        if not text or end_ms <= start_ms:
            return None
        words = []
        for word in item.get("words", []):
            word_start = round(float(word.get("start", 0)) * 1000)
            word_end = round(float(word.get("end", 0)) * 1000)
            word_text = str(word.get("word", word.get("text", "")))
            if word_text and word_end > word_start:
                words.append(TranscriptWord(word_start, word_end, word_text, word.get("probability")))
        return TranscriptSegment(
            start_ms,
            end_ms,
            text,
            words=tuple(sorted(words, key=lambda value: (value.start_ms, value.end_ms))),
            language=item.get("language") or language,
            language_confidence=item.get("language_probability"),
            diagnostics=cls._diagnostics(item),
        )

    @staticmethod
    def _parse_words(items: list[dict]) -> tuple[TranscriptWord, ...]:
        words = []
        for item in items:
            start_ms = round(float(item.get("start", 0)) * 1000)
            end_ms = round(float(item.get("end", 0)) * 1000)
            text = str(item.get("word", item.get("text", "")))
            if text and end_ms > start_ms:
                words.append(
                    TranscriptWord(
                        start_ms,
                        end_ms,
                        text,
                        item.get("probability", item.get("confidence")),
                    )
                )
        return tuple(sorted(words, key=lambda item: (item.start_ms, item.end_ms)))

    def transcribe(
        self, audio: str | Path, options: TranscriptionOptions | None = None
    ) -> TranscriptionResult:
        path = Path(audio).resolve()
        if not path.is_file():
            raise TranscriptionError(f"audio input not found: {path}")
        options = options or TranscriptionOptions()
        if self.server is not None:
            self.server.start()
        fields = [
            ("model", self.model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "segment"),
            ("timestamp_granularities[]", "word"),
        ]
        if options.language:
            fields.append(("language", options.language))
        if options.initial_prompt:
            fields.append(("prompt", options.initial_prompt))
        body, boundary = self._multipart(path, fields)
        request = urllib.request.Request(
            f"{self.endpoint}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                document = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise TranscriptionError(f"WhisperKit local-server request failed: {exc}") from exc
        language = document.get("language") or options.language
        segments = tuple(
            sorted(
                (
                    segment
                    for segment in (
                        self._parse_segment(item, language)
                        for item in document.get("segments", [])
                    )
                    if segment is not None
                ),
                key=lambda item: (item.start_ms, item.end_ms),
            )
        )
        top_level_words = self._parse_words(document.get("words", []))
        if top_level_words:
            segments = tuple(
                replace(
                    segment,
                    words=segment.words
                    or tuple(
                        word
                        for word in top_level_words
                        if segment.start_ms
                        <= word.start_ms + (word.end_ms - word.start_ms) // 2
                        < segment.end_ms
                    ),
                )
                for segment in segments
            )
        text = str(document.get("text", "")).strip() or " ".join(item.text for item in segments)
        duration = document.get("duration")
        return TranscriptionResult(
            segments=segments,
            text=text,
            backend="whisperkit-local-server",
            model=self.model,
            device="apple-silicon",
            language=language,
            duration_ms=(round(float(duration) * 1000) if duration is not None else max((item.end_ms for item in segments), default=None)),
            confidence_available=any(item.confidence is not None for segment in segments for item in segment.words),
            source_sha256=_source_hash(path),
            diagnostics={
                "provider_response_fields": sorted(document.keys()),
                "top_level_word_count": len(top_level_words),
            },
            provenance={
                "provider": "WhisperKit",
                "boundary": "official-openai-compatible-local-server",
                "endpoint": self.endpoint,
                "model": self.model,
                "model_path": str(self.server.model_path) if self.server else None,
                "model_tree_sha256": self.server.model_sha256 if self.server else None,
                "model_size_bytes": self.server.model_size_bytes if self.server else None,
                "cli_executable": self.server.executable if self.server else None,
                "cli_version": self.server.version if self.server else None,
                "temperature_fallback_count": 0 if self.server else None,
            },
        )
