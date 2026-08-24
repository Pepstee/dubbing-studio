from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat as stat_module
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from dubbing.media import ffmpeg_executable
from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
    transcription_result_from_dict,
)

_CHECKPOINT_SCHEMA = "dubbing.transcription-checkpoint.v1"
_MAX_AUDIO_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class SourceStatIdentity:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> SourceStatIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size_bytes=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "inode": self.inode,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class SourceBinding:
    """A digest and stable filesystem identity created from one trusted open file."""

    path: Path
    sha256: str
    stat: SourceStatIdentity

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "stat": self.stat.to_dict(),
        }


def _validate_expected_digest(expected_sha256: str | None) -> None:
    if expected_sha256 is not None and (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise TranscriptionError("expected source digest must be a lowercase SHA-256 digest")


def _resolved_regular_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        candidate_stat = candidate.lstat()
    except OSError as exc:
        raise TranscriptionError(f"media input not found: {candidate}") from exc
    if stat_module.S_ISLNK(candidate_stat.st_mode):
        raise TranscriptionError("media input must not be a symbolic link")
    if not stat_module.S_ISREG(candidate_stat.st_mode):
        raise TranscriptionError("media input must be a regular file")
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise TranscriptionError(f"could not resolve media input: {candidate}") from exc


def create_source_binding(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> SourceBinding:
    """Hash a stable regular file descriptor and bind it to its resolved path."""

    _validate_expected_digest(expected_sha256)
    candidate = Path(path).expanduser()
    resolved = _resolved_regular_path(candidate)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise TranscriptionError(f"could not open media input safely: {candidate}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode):
            raise TranscriptionError("media input must be a regular file")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = SourceStatIdentity.from_stat(before)
    after_identity = SourceStatIdentity.from_stat(after)
    if before_identity != after_identity:
        raise TranscriptionError("media source changed while its digest was computed")
    try:
        path_after = candidate.lstat()
        resolved_after = candidate.resolve(strict=True)
    except OSError as exc:
        raise TranscriptionError("media source path changed while its digest was computed") from exc
    if (
        stat_module.S_ISLNK(path_after.st_mode)
        or not stat_module.S_ISREG(path_after.st_mode)
        or path_after.st_dev != after.st_dev
        or path_after.st_ino != after.st_ino
        or resolved_after != resolved
    ):
        raise TranscriptionError("media source path changed while its digest was computed")
    computed = digest.hexdigest()
    if expected_sha256 is not None and computed != expected_sha256:
        raise TranscriptionError("media source digest does not match expected SHA-256")
    return SourceBinding(path=resolved, sha256=computed, stat=after_identity)


def validate_source_binding(path: str | Path, binding: SourceBinding) -> None:
    """Check path and current stat identity without re-reading source bytes."""

    if not isinstance(binding, SourceBinding):
        raise TranscriptionError("source_binding must be a SourceBinding")
    resolved = _resolved_regular_path(path)
    if resolved != binding.path:
        raise TranscriptionError("source binding path does not match media input")
    try:
        current = SourceStatIdentity.from_stat(resolved.stat(follow_symlinks=False))
    except OSError as exc:
        raise TranscriptionError("could not stat source-bound media input") from exc
    if current != binding.stat:
        raise TranscriptionError("source binding stat identity no longer matches media input")


def verify_source_binding(binding: SourceBinding) -> SourceBinding:
    """Re-hash a binding and require both digest and stat identity to remain stable."""

    verified = create_source_binding(binding.path, expected_sha256=binding.sha256)
    if verified.stat != binding.stat:
        raise TranscriptionError("media source stat identity changed after admission")
    return verified


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def media_duration_ms(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise TranscriptionError("ffprobe is required for resumable transcription")
    try:
        process = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        seconds = float(process.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        raise TranscriptionError(f"could not determine duration of {path.name}") from exc
    if seconds <= 0:
        raise TranscriptionError("audio duration must be positive")
    if seconds > _MAX_AUDIO_SECONDS:
        raise TranscriptionError(
            f"audio duration exceeds the {_MAX_AUDIO_SECONDS}s safety limit"
        )
    return round(seconds * 1000)


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, indent=2, sort_keys=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ResumableTranscriptionJob:
    """Source-bound chunk runner for recordings too valuable to restart."""

    def __init__(
        self,
        backend: TranscriptionBackend,
        checkpoint_dir: str | Path,
        *,
        chunk_seconds: int = 30 * 60,
        overlap_seconds: int = 5,
    ) -> None:
        if not 30 <= chunk_seconds <= 60 * 60:
            raise ValueError("chunk_seconds must be between 30 and 3600")
        self.backend = backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.chunk_seconds = chunk_seconds
        if not 0 <= overlap_seconds < chunk_seconds // 2:
            raise ValueError("overlap_seconds must be non-negative and below half a chunk")
        self.overlap_seconds = overlap_seconds

    @staticmethod
    def _extract_chunk(source: Path, start_ms: int, end_ms: int, output: Path) -> None:
        ffmpeg = ffmpeg_executable()
        if ffmpeg is None:
            raise TranscriptionError("ffmpeg is required for chunked transcription")
        try:
            subprocess.run(
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
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=max(300, (end_ms - start_ms) // 1000),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "unknown ffmpeg error"
            raise TranscriptionError(f"could not extract audio chunk: {detail}") from exc

    def _manifest(
        self,
        source: Path,
        duration_ms: int,
        digest: str,
        options: TranscriptionOptions,
    ) -> dict:
        return {
            "schema_version": _CHECKPOINT_SCHEMA,
            "source_name": source.name,
            "source_sha256": digest,
            "duration_ms": duration_ms,
            "backend_identity": self.backend.identity,
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
            "options": {
                "language": options.language,
                "task": options.task,
                "initial_prompt": options.initial_prompt,
                "word_timestamps": options.word_timestamps,
            },
        }

    def _admit_manifest(self, expected: dict) -> None:
        path = self.checkpoint_dir / "manifest.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TranscriptionError("checkpoint manifest is malformed") from exc
            if existing != expected:
                raise TranscriptionError(
                    "checkpoint does not match source, backend, or chunk configuration"
                )
        else:
            _atomic_json(path, expected)

    def run(
        self,
        audio: str | Path,
        options: TranscriptionOptions | None = None,
        *,
        source_digest: str | None = None,
        duration_ms: int | None = None,
    ) -> TranscriptionResult:
        source = Path(audio)
        if not source.is_file():
            raise TranscriptionError(f"audio input not found: {source}")
        options = options or TranscriptionOptions()
        duration_ms = media_duration_ms(source) if duration_ms is None else duration_ms
        digest = source_sha256(source) if source_digest is None else source_digest
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._admit_manifest(self._manifest(source, duration_ms, digest, options))

        chunk_ms = self.chunk_seconds * 1000
        segments: list[TranscriptSegment] = []
        texts: list[str] = []
        language = options.language
        confidence_available = False
        total_chunks = (duration_ms + chunk_ms - 1) // chunk_ms
        for index, start_ms in enumerate(range(0, duration_ms, chunk_ms)):
            end_ms = min(duration_ms, start_ms + chunk_ms)
            checkpoint = self.checkpoint_dir / "chunks" / f"{index:06d}.json"
            if checkpoint.exists():
                try:
                    result = transcription_result_from_dict(
                        json.loads(checkpoint.read_text(encoding="utf-8"))
                    )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise TranscriptionError(
                        f"chunk checkpoint is malformed: {checkpoint.name}"
                    ) from exc
            else:
                overlap_ms = self.overlap_seconds * 1000
                extract_start_ms = max(0, start_ms - overlap_ms)
                extract_end_ms = min(duration_ms, end_ms + overlap_ms)
                with tempfile.TemporaryDirectory(
                    prefix="dubbing-transcription-"
                ) as directory:
                    chunk = Path(directory) / f"{index:06d}.wav"
                    self._extract_chunk(
                        source,
                        extract_start_ms,
                        extract_end_ms,
                        chunk,
                    )
                    local = self.backend.transcribe(chunk, options)
                shifted = local.shifted(extract_start_ms)
                admitted = tuple(
                    segment
                    for segment in shifted.segments
                    if start_ms
                    <= segment.start_ms + (segment.end_ms - segment.start_ms) // 2
                    < end_ms
                )
                result = replace(
                    shifted,
                    segments=admitted,
                    text=" ".join(segment.text for segment in admitted).strip(),
                    duration_ms=end_ms,
                    source_sha256=digest,
                )
                _atomic_json(checkpoint, result.to_dict())
            segments.extend(result.segments)
            if result.text.strip():
                texts.append(result.text.strip())
            language = language or result.language
            confidence_available = confidence_available or result.confidence_available
            _atomic_json(
                self.checkpoint_dir / "progress.json",
                {
                    "schema_version": "dubbing.transcription-progress.v1",
                    "completed_chunks": index + 1,
                    "total_chunks": total_chunks,
                    "completed_through_ms": end_ms,
                },
            )

        final = TranscriptionResult(
            segments=tuple(
                sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
            ),
            text=" ".join(texts),
            backend=self.backend.identity.split(":", 1)[0],
            model=self.backend.identity.split(":", 1)[-1],
            device="chunked",
            language=language,
            duration_ms=duration_ms,
            confidence_available=confidence_available,
            source_sha256=digest,
        )
        _atomic_json(self.checkpoint_dir / "result.json", final.to_dict())
        return final
