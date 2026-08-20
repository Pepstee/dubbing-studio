from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from array import array
from dataclasses import dataclass, replace
from pathlib import Path

from dubbing.evaluation.metrics import token_agreement, word_tokens
from dubbing.media import ffmpeg_executable
from dubbing.transcription.audio_candidates import build_audio_candidates
from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.job import source_sha256
from dubbing.transcription.language import annotate_transcript_languages, script_evidence
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptionError,
    TranscriptionOptions,
    TranscriptionResult,
    transcription_result_from_dict,
)
from dubbing.transcription.quality import (
    QUALITY_POLICY_VERSION,
    TranscriptQualityReport,
    TranscriptQualityStatus,
    evaluate_transcript_quality,
)
from dubbing.transcription.speech_regions import SpeechRegionDetector


_FAILED_SPAN_TEXT = "[UNCERTAIN: LOCAL TRANSCRIPTION FAILED]"
_DISAGREEMENT_SPAN_TEXT = "[UNCERTAIN: INDEPENDENT TRANSCRIPTIONS DISAGREE]"
_UNISOLATED_SPAN_TEXT = "[UNCERTAIN: SPEECH REGION NOT ISOLATED]"
_TARGET_RETRY_MIN_MS = 20_000
_TARGET_RETRY_TARGET_MS = 45_000
_TARGET_RETRY_MAX_MS = 60_000
_TARGET_RETRY_PADDING_MS = 2_000
_INDEPENDENT_AGREEMENT_THRESHOLD = 0.75
_INDEPENDENT_MINIMUM_ACOUSTIC_CONFIDENCE = 0.35
_INDEPENDENT_MINIMUM_CONSENSUS_TOKENS = 2
_LANGUAGE_RETRY_CONTEXT_MS = 1_000
_COORDINATOR_VERSION = "adaptive-long-form-v17"
_CHUNK_RECEIPT_SCHEMA = "dubbing.adaptive-chunk-receipt.v1"
_SILENCE_ADMISSION_POLICY = "full-energy-coverage-plus-empty-semantic-vad-v1"
_AUDIO_BOUNDS_POLICY = "drop-wholly-outside-clamp-overlap-v1"


class _UnsetSpeechRegionDetector:
    pass


_UNSET_SPEECH_REGION_DETECTOR = _UnsetSpeechRegionDetector()


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str
    channels: int
    sample_rate_hz: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "codec": self.codec,
            "channels": self.channels,
            "sample_rate_hz": self.sample_rate_hz,
        }


@dataclass(frozen=True)
class MediaProbe:
    duration_ms: int
    size_bytes: int
    audio_streams: tuple[AudioStream, ...]

    def to_dict(self) -> dict:
        return {
            "duration_ms": self.duration_ms,
            "size_bytes": self.size_bytes,
            "audio_streams": [item.to_dict() for item in self.audio_streams],
        }


@dataclass(frozen=True)
class AdaptiveChunk:
    index: int
    start_ms: int
    end_ms: int
    extract_start_ms: int
    extract_end_ms: int
    boundary_reason: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class _SpanCandidate:
    audio_candidate_id: str
    audio_candidate_sha256: str
    audio_processing: dict
    backend_identity: str
    language: str | None
    segments: tuple[TranscriptSegment, ...]
    text: str
    quality: TranscriptQualityReport
    average_log_probability: float
    average_acoustic_confidence: float | None
    attempt_index: int

    @property
    def score(self) -> tuple[float, ...]:
        return (
            2.0 if self.quality.status is TranscriptQualityStatus.PASS else 1.0,
            -float(sum(item.uncertain for item in self.segments)),
            self.average_acoustic_confidence or -1.0,
            self.average_log_probability,
        )


def probe_media(path: str | Path) -> MediaProbe:
    source = Path(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        if source.suffix.casefold() == ".wav":
            return _probe_pcm_wave(source)
        raise TranscriptionError("ffprobe is required for non-WAV adaptive transcription")
    try:
        process = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=index,codec_type,codec_name,channels,sample_rate",
                "-of",
                "json",
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        document = json.loads(process.stdout)
        duration_ms = round(float(document["format"]["duration"]) * 1000)
        size_bytes = int(document["format"].get("size", source.stat().st_size))
        streams = tuple(
            AudioStream(
                index=int(item["index"]),
                codec=str(item.get("codec_name", "unknown")),
                channels=int(item.get("channels", 1)),
                sample_rate_hz=int(item.get("sample_rate", 16000)),
            )
            for item in document.get("streams", [])
            if item.get("codec_type") == "audio"
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        raise TranscriptionError(f"could not probe media: {source.name}") from exc
    if duration_ms <= 0 or not streams:
        raise TranscriptionError("media must contain at least one positive-duration audio stream")
    return MediaProbe(duration_ms, size_bytes, streams)


def _probe_pcm_wave(source: Path) -> MediaProbe:
    try:
        with wave.open(str(source), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise TranscriptionError("native WAV fallback requires uncompressed PCM")
            duration_ms = round(audio.getnframes() * 1000 / audio.getframerate())
            stream = AudioStream(
                index=0,
                codec=f"pcm_s{audio.getsampwidth() * 8}le",
                channels=audio.getnchannels(),
                sample_rate_hz=audio.getframerate(),
            )
    except (EOFError, wave.Error) as error:
        raise TranscriptionError(f"could not probe WAV media: {source.name}") from error
    if duration_ms <= 0:
        raise TranscriptionError("media must have positive duration")
    return MediaProbe(duration_ms, source.stat().st_size, (stream,))


def detect_silence_centres(
    path: str | Path,
    *,
    minimum_silence_seconds: float = 0.7,
    noise_db: float = -35.0,
) -> tuple[int, ...]:
    return tuple(
        round((start + end) * 500)
        for start, end in _detect_silence_seconds(
            path,
            minimum_silence_seconds=minimum_silence_seconds,
            noise_db=noise_db,
        )
    )


def detect_silence_intervals(
    path: str | Path,
    *,
    minimum_silence_seconds: float = 0.7,
    noise_db: float = -35.0,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (round(start * 1000), round(end * 1000))
        for start, end in _detect_silence_seconds(
            path,
            minimum_silence_seconds=minimum_silence_seconds,
            noise_db=noise_db,
        )
    )


def _interval_coverage_ms(
    start_ms: int,
    end_ms: int,
    intervals: tuple[tuple[int, int], ...],
) -> int:
    clipped = sorted(
        (max(start_ms, start), min(end_ms, end))
        for start, end in intervals
        if start < end_ms and end > start_ms
    )
    if not clipped:
        return 0
    covered = 0
    cursor_start, cursor_end = clipped[0]
    for next_start, next_end in clipped[1:]:
        if next_start <= cursor_end:
            cursor_end = max(cursor_end, next_end)
        else:
            covered += cursor_end - cursor_start
            cursor_start, cursor_end = next_start, next_end
    return covered + cursor_end - cursor_start


def _detect_silence_seconds(
    path: str | Path,
    *,
    minimum_silence_seconds: float,
    noise_db: float,
) -> tuple[tuple[float, float], ...]:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        source = Path(path)
        if source.suffix.casefold() == ".wav":
            return _detect_pcm_wave_silence(
                source,
                minimum_silence_seconds=minimum_silence_seconds,
                noise_db=noise_db,
            )
        raise TranscriptionError("ffmpeg is required for non-WAV silence-aware chunking")
    process = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-af",
            f"silencedetect=noise={noise_db}dB:d={minimum_silence_seconds}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    starts = [float(value) for value in re.findall(r"silence_start: ([0-9.]+)", process.stderr)]
    ends = [float(value) for value in re.findall(r"silence_end: ([0-9.]+)", process.stderr)]
    return tuple((start, end) for start, end in zip(starts, ends) if end > start)


def _detect_pcm_wave_silence(
    source: Path,
    *,
    minimum_silence_seconds: float,
    noise_db: float,
) -> tuple[tuple[float, float], ...]:
    with wave.open(str(source), "rb") as audio:
        if audio.getcomptype() != "NONE" or audio.getsampwidth() != 2:
            raise TranscriptionError("native silence detection requires 16-bit PCM WAV")
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        samples = array("h")
        samples.frombytes(audio.readframes(audio.getnframes()))
    window_frames = max(1, round(sample_rate * 0.01))
    threshold = 32767 * 10 ** (noise_db / 20)
    minimum_windows = max(1, math.ceil(minimum_silence_seconds / 0.01))
    intervals = []
    silence_start: int | None = None
    frame_count = len(samples) // channels
    for window_start in range(0, frame_count, window_frames):
        sample_start = window_start * channels
        sample_end = min(frame_count, window_start + window_frames) * channels
        silent = (
            max((abs(value) for value in samples[sample_start:sample_end]), default=0) <= threshold
        )
        if silent and silence_start is None:
            silence_start = window_start
        if not silent and silence_start is not None:
            windows = (window_start - silence_start) / window_frames
            if windows >= minimum_windows:
                intervals.append((silence_start / sample_rate, window_start / sample_rate))
            silence_start = None
    if silence_start is not None:
        windows = (frame_count - silence_start) / window_frames
        if windows >= minimum_windows:
            intervals.append((silence_start / sample_rate, frame_count / sample_rate))
    return tuple(intervals)


def _extract_pcm_wave(source: Path, start_ms: int, end_ms: int, output: Path) -> None:
    with wave.open(str(source), "rb") as audio:
        if audio.getcomptype() != "NONE":
            raise TranscriptionError("native WAV extraction requires uncompressed PCM")
        sample_rate = audio.getframerate()
        start_frame = round(start_ms * sample_rate / 1000)
        end_frame = round(end_ms * sample_rate / 1000)
        audio.setpos(min(start_frame, audio.getnframes()))
        content = audio.readframes(max(0, end_frame - start_frame))
        parameters = audio.getparams()
    with wave.open(str(output), "wb") as destination:
        destination.setparams(parameters)
        destination.writeframes(content)


class AdaptiveChunkPlanner:
    def __init__(
        self,
        *,
        target_seconds: int = 240,
        minimum_seconds: int = 60,
        maximum_seconds: int = 480,
        overlap_seconds: float = 2.0,
        hard_boundary_silence_seconds: float = 0.7,
    ) -> None:
        if not 1 <= minimum_seconds <= target_seconds <= maximum_seconds <= 1800:
            raise ValueError("chunk bounds must satisfy 1 <= minimum <= target <= maximum <= 1800")
        if overlap_seconds < 0 or overlap_seconds >= minimum_seconds / 2:
            raise ValueError("invalid overlap_seconds")
        if hard_boundary_silence_seconds <= 0:
            raise ValueError("hard_boundary_silence_seconds must be positive")
        self.target_ms = target_seconds * 1000
        self.minimum_ms = minimum_seconds * 1000
        self.maximum_ms = maximum_seconds * 1000
        self.overlap_ms = round(overlap_seconds * 1000)
        self.hard_boundary_silence_ms = round(hard_boundary_silence_seconds * 1000)

    def plan(
        self,
        duration_ms: int,
        silence_centres: tuple[int, ...],
        *,
        silence_intervals: tuple[tuple[int, int], ...] = (),
    ) -> tuple[AdaptiveChunk, ...]:
        boundaries = [0]
        cursor = 0
        silences = sorted(point for point in silence_centres if 0 < point < duration_ms)
        silence_durations = {
            round((start_ms + end_ms) / 2): end_ms - start_ms
            for start_ms, end_ms in silence_intervals
            if 0 <= start_ms < end_ms <= duration_ms
        }
        reasons: list[str] = []
        while duration_ms - cursor > self.target_ms + self.minimum_ms:
            remaining = duration_ms - cursor
            minimum = cursor + self.minimum_ms
            maximum = min(
                cursor + self.maximum_ms,
                duration_ms - self.minimum_ms,
            )
            target = cursor + self.target_ms
            candidates = [point for point in silences if minimum <= point <= maximum]
            if candidates:
                hard_boundaries = [
                    point
                    for point in candidates
                    if silence_durations.get(point, 0) >= self.hard_boundary_silence_ms
                ]
                if hard_boundaries:
                    boundary = min(hard_boundaries)
                    reason = "long-silence"
                else:
                    boundary = min(
                        candidates,
                        key=lambda point: (
                            -silence_durations.get(point, 0),
                            abs(point - target),
                            point,
                        ),
                    )
                    reason = "silence"
            elif remaining > self.maximum_ms:
                boundary = cursor + self.maximum_ms
                reason = "maximum-duration"
            else:
                break
            boundaries.append(boundary)
            reasons.append(reason)
            cursor = boundary
        boundaries.append(duration_ms)
        reasons.append("end-of-media")
        chunks = []
        for index, (start_ms, end_ms) in enumerate(zip(boundaries, boundaries[1:])):
            chunks.append(
                AdaptiveChunk(
                    index=index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    extract_start_ms=max(0, start_ms - self.overlap_ms),
                    extract_end_ms=min(duration_ms, end_ms + self.overlap_ms),
                    boundary_reason=reasons[index],
                )
            )
        return tuple(chunks)

    def to_dict(self) -> dict:
        return {
            "target_ms": self.target_ms,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "overlap_ms": self.overlap_ms,
            "hard_boundary_silence_ms": self.hard_boundary_silence_ms,
            "boundary_selection": "earliest-hard-silence-else-longest-v1",
        }


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _document_sha256(document: dict) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def reconcile_chunks(
    chunk_results: list[tuple[AdaptiveChunk, TranscriptionResult]],
) -> tuple[TranscriptSegment, ...]:
    admitted: list[TranscriptSegment] = []
    for chunk, result in chunk_results:
        shifted = result.shifted(chunk.extract_start_ms)
        for segment in shifted.segments:
            midpoint = segment.start_ms + (segment.end_ms - segment.start_ms) // 2
            if not chunk.start_ms <= midpoint < chunk.end_ms:
                continue
            # The overlap is decoder context, not output territory.  Clamp admitted
            # timestamps to the chunk's logical (non-overlapping) interval so two
            # adjacent chunks cannot manufacture a timestamp collision.
            start_ms = max(segment.start_ms, chunk.start_ms)
            end_ms = min(segment.end_ms, chunk.end_ms)
            if end_ms <= start_ms:
                continue
            words = []
            for word in segment.words:
                word_start = max(word.start_ms, start_ms)
                word_end = min(word.end_ms, end_ms)
                if word_end > word_start:
                    words.append(
                        replace(
                            word,
                            start_ms=word_start,
                            end_ms=word_end,
                        )
                    )
            words.sort(key=lambda item: (item.start_ms, item.end_ms))
            segment = replace(
                segment,
                start_ms=start_ms,
                end_ms=end_ms,
                words=tuple(words),
            )
            if admitted and _normalize(admitted[-1].text) == _normalize(segment.text):
                if segment.start_ms <= admitted[-1].end_ms + 2000:
                    continue
            admitted.append(segment)
    return tuple(sorted(admitted, key=lambda item: (item.start_ms, item.end_ms)))


class AdaptiveLongFormCoordinator:
    """Source-bound coordinator that retries only rejected spans."""

    def __init__(
        self,
        backend: TranscriptionBackend,
        checkpoint_dir: str | Path,
        *,
        planner: AdaptiveChunkPlanner | None = None,
        retry_backend: TranscriptionBackend | None = None,
        speech_region_detector: SpeechRegionDetector | None = None,
        silence_verification_detector: (
            SpeechRegionDetector | None | _UnsetSpeechRegionDetector
        ) = _UNSET_SPEECH_REGION_DETECTOR,
        targeted_retry_region_detector: (
            SpeechRegionDetector | None | _UnsetSpeechRegionDetector
        ) = _UNSET_SPEECH_REGION_DETECTOR,
        candidate_languages: tuple[str, ...] = ("en", "ru", "ro", "ko"),
        minimum_silence_seconds: float = 0.7,
        language_retry_policy: dict[str, str] | None = None,
        audio_candidate_policies: tuple[str, ...] = (
            "raw",
            "downmix",
            "channels",
        ),
        maximum_audio_candidate_channels: int = 4,
    ) -> None:
        if minimum_silence_seconds <= 0:
            raise ValueError("minimum_silence_seconds must be positive")
        retry_policy = language_retry_policy or {}
        if any(mode not in {"always", "confidence"} for mode in retry_policy.values()):
            raise ValueError("language retry policy must use always or confidence")
        self.backend = backend
        self.retry_backend = retry_backend
        # `speech_region_detector` remains the backward-compatible alias for
        # callers that intentionally use one detector for both decisions.
        self.speech_region_detector = speech_region_detector
        self.silence_verification_detector = (
            speech_region_detector
            if isinstance(silence_verification_detector, _UnsetSpeechRegionDetector)
            else silence_verification_detector
        )
        self.targeted_retry_region_detector = (
            speech_region_detector
            if isinstance(targeted_retry_region_detector, _UnsetSpeechRegionDetector)
            else targeted_retry_region_detector
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.planner = planner or AdaptiveChunkPlanner()
        self.candidate_languages = candidate_languages
        self.minimum_silence_seconds = minimum_silence_seconds
        self.language_retry_policy = dict(sorted(retry_policy.items()))
        if (
            not audio_candidate_policies
            or audio_candidate_policies[0] != "raw"
            or len(set(audio_candidate_policies)) != len(audio_candidate_policies)
        ):
            raise ValueError("audio candidate policies must start with unique raw")
        if not 1 <= maximum_audio_candidate_channels <= 8:
            raise ValueError("maximum_audio_candidate_channels must be between 1 and 8")
        self.audio_candidate_policies = audio_candidate_policies
        self.maximum_audio_candidate_channels = maximum_audio_candidate_channels

    @staticmethod
    def _extract(
        source: Path,
        streams: tuple[AudioStream, ...],
        chunk: AdaptiveChunk,
        output: Path,
    ) -> None:
        ffmpeg = ffmpeg_executable()
        if ffmpeg is None:
            if source.suffix.casefold() == ".wav" and len(streams) == 1:
                _extract_pcm_wave(source, chunk.extract_start_ms, chunk.extract_end_ms, output)
                return
            raise TranscriptionError("ffmpeg is required for non-WAV adaptive transcription")
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{chunk.extract_start_ms / 1000:.3f}",
            "-i",
            str(source),
            "-t",
            f"{(chunk.extract_end_ms - chunk.extract_start_ms) / 1000:.3f}",
        ]
        if len(streams) == 1:
            stream = streams[0]
            command.extend(("-map", f"0:{stream.index}"))
        else:
            inputs = "".join(f"[0:{stream.index}]" for stream in streams)
            command.extend(
                (
                    "-filter_complex",
                    f"{inputs}amerge=inputs={len(streams)}[all_audio]",
                    "-map",
                    "[all_audio]",
                )
            )
        command.extend(
            (
                "-vn",
                "-ac",
                str(sum(stream.channels for stream in streams)),
                "-ar",
                str(max(stream.sample_rate_hz for stream in streams)),
                "-c:a",
                "pcm_s16le",
                "-y",
                str(output),
            )
        )
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(300, (chunk.extract_end_ms - chunk.extract_start_ms) // 1000),
        )
        if process.returncode:
            raise TranscriptionError(f"chunk extraction failed: {process.stderr.strip()}")

    def _manifest(
        self,
        source: Path,
        digest: str,
        probe: MediaProbe,
        chunks: tuple[AdaptiveChunk, ...],
        processing_end_ms: int | None = None,
    ) -> dict:
        if processing_end_ms is None:
            processing_end_ms = probe.duration_ms
        manifest = {
            "schema_version": "dubbing.adaptive-transcription-checkpoint.v2",
            "coordinator_version": _COORDINATOR_VERSION,
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "source_name": source.name,
            "source_sha256": digest,
            "probe": probe.to_dict(),
            "processing_scope": {
                "start_ms": 0,
                "end_ms": processing_end_ms,
                "processed_duration_ms": processing_end_ms,
                "full_source_duration_ms": probe.duration_ms,
                "full_source_sha256_bound": True,
            },
            "backend_identity": self.backend.identity,
            "retry_backend_identity": self.retry_backend.identity if self.retry_backend else None,
            "silence_verification_detector_identity": (
                self.silence_verification_detector.identity
                if self.silence_verification_detector is not None
                else None
            ),
            "targeted_retry_region_detector_identity": (
                self.targeted_retry_region_detector.identity
                if self.targeted_retry_region_detector is not None
                else None
            ),
            "candidate_languages": list(self.candidate_languages),
            "audio_candidates": {
                "policies": list(self.audio_candidate_policies),
                "maximum_channels": self.maximum_audio_candidate_channels,
                "promotion_policy": "raw-consensus-first-else-independent-unanimity-v1",
            },
            "targeted_retry": {
                "minimum_ms": _TARGET_RETRY_MIN_MS,
                "target_ms": _TARGET_RETRY_TARGET_MS,
                "maximum_ms": _TARGET_RETRY_MAX_MS,
                "padding_ms": _TARGET_RETRY_PADDING_MS,
                "maximum_rounds": 2,
                "independent_backend_required": True,
                "independent_agreement_threshold": _INDEPENDENT_AGREEMENT_THRESHOLD,
                "independent_minimum_acoustic_confidence": (
                    _INDEPENDENT_MINIMUM_ACOUSTIC_CONFIDENCE
                ),
                "independent_minimum_consensus_tokens": (_INDEPENDENT_MINIMUM_CONSENSUS_TOKENS),
            },
            "silence_admission": {
                "policy": _SILENCE_ADMISSION_POLICY,
                "required_energy_coverage_ratio": 1.0,
                "semantic_vad_zero_regions_required": True,
            },
            "planner": self.planner.to_dict(),
            "chunks": [item.to_dict() for item in chunks],
        }
        if self.minimum_silence_seconds != 0.7:
            manifest["silence_detection"] = {
                "minimum_silence_seconds": self.minimum_silence_seconds
            }
        if self.language_retry_policy:
            manifest["language_retry_policy"] = self.language_retry_policy
        return manifest

    def _admit_manifest(self, expected: dict) -> None:
        path = self.checkpoint_dir / "manifest.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TranscriptionError("adaptive checkpoint manifest is malformed") from exc
            if existing != expected:
                raise TranscriptionError(
                    "adaptive checkpoint does not match source or configuration"
                )
        else:
            _atomic_json(path, expected)

    @staticmethod
    def _critical_failure_intervals(
        result: TranscriptionResult,
        quality: TranscriptQualityReport,
    ) -> list[tuple[int, int]]:
        critical_codes = {
            issue.code for issue in quality.issues if issue.severity in {"critical", "fatal"}
        }
        if not critical_codes:
            return []
        intervals: list[tuple[int, int]] = [
            (issue.start_ms, issue.end_ms)
            for issue in quality.issues
            if issue.severity in {"critical", "fatal"}
            and issue.start_ms is not None
            and issue.end_ms is not None
        ]
        previous: TranscriptSegment | None = None
        for segment in result.segments:
            if segment.text == _FAILED_SPAN_TEXT or (
                segment.diagnostics and segment.diagnostics.fallback_exhausted
            ):
                intervals.append((segment.start_ms, segment.end_ms))
            if previous is not None:
                if _normalize(previous.text) == _normalize(segment.text):
                    intervals.extend(
                        [
                            (previous.start_ms, previous.end_ms),
                            (segment.start_ms, segment.end_ms),
                        ]
                    )
                if (
                    "excessive_timestamp_overlap" in critical_codes
                    and segment.start_ms < previous.end_ms
                ):
                    intervals.append(
                        (
                            min(previous.start_ms, segment.start_ms),
                            max(previous.end_ms, segment.end_ms),
                        )
                    )
            previous = segment
        if not intervals:
            duration_ms = result.duration_ms or max(
                (segment.end_ms for segment in result.segments), default=1
            )
            intervals.append((0, max(1, duration_ms)))
        return intervals

    @staticmethod
    def _bounded_retry_spans(
        intervals: list[tuple[int, int]],
        *,
        duration_ms: int,
        silence_centres: tuple[int, ...],
    ) -> tuple[tuple[int, int], ...]:
        padded = []
        for start_ms, end_ms in sorted(intervals):
            start_ms = max(0, start_ms - _TARGET_RETRY_PADDING_MS)
            end_ms = min(duration_ms, end_ms + _TARGET_RETRY_PADDING_MS)
            if duration_ms >= _TARGET_RETRY_MIN_MS and end_ms - start_ms < _TARGET_RETRY_MIN_MS:
                centre = start_ms + (end_ms - start_ms) // 2
                start_ms = max(0, centre - _TARGET_RETRY_MIN_MS // 2)
                end_ms = min(duration_ms, start_ms + _TARGET_RETRY_MIN_MS)
                start_ms = max(0, end_ms - _TARGET_RETRY_MIN_MS)
            if padded and start_ms <= padded[-1][1] + _TARGET_RETRY_PADDING_MS:
                padded[-1] = (padded[-1][0], max(padded[-1][1], end_ms))
            else:
                padded.append((start_ms, end_ms))

        bounded: list[tuple[int, int]] = []
        for start_ms, end_ms in padded:
            group_start_index = len(bounded)
            cursor = start_ms
            while end_ms - cursor > _TARGET_RETRY_MAX_MS:
                eligible = [
                    point
                    for point in silence_centres
                    if cursor + _TARGET_RETRY_MIN_MS <= point <= cursor + _TARGET_RETRY_MAX_MS
                ]
                target = cursor + _TARGET_RETRY_TARGET_MS
                boundary = (
                    min(eligible, key=lambda point: (abs(point - target), point))
                    if eligible
                    else cursor + _TARGET_RETRY_MAX_MS
                )
                bounded.append((cursor, boundary))
                cursor = boundary
            if end_ms > cursor:
                if len(bounded) > group_start_index and end_ms - cursor < _TARGET_RETRY_MIN_MS:
                    previous_start, _ = bounded[-1]
                    if end_ms - previous_start <= _TARGET_RETRY_MAX_MS:
                        bounded[-1] = (previous_start, end_ms)
                        continue
                    revised_boundary = end_ms - _TARGET_RETRY_MIN_MS
                    if revised_boundary - previous_start >= _TARGET_RETRY_MIN_MS:
                        bounded[-1] = (previous_start, revised_boundary)
                        cursor = revised_boundary
                bounded.append((cursor, end_ms))
        return tuple(bounded)

    @staticmethod
    def _splice_retry_spans(
        original: TranscriptionResult,
        spans: tuple[tuple[int, int], ...],
        replacements: list[TranscriptSegment],
    ) -> TranscriptionResult:
        retained = [
            segment
            for segment in original.segments
            if not any(
                segment.start_ms < end_ms and segment.end_ms > start_ms
                for start_ms, end_ms in spans
            )
        ]
        segments = tuple(
            sorted(retained + replacements, key=lambda item: (item.start_ms, item.end_ms))
        )
        return replace(original, segments=segments, text=" ".join(item.text for item in segments))

    def _target_languages(self, text: str) -> tuple[str, ...]:
        evidence = script_evidence(text)
        preferred: tuple[str, ...] = ()
        if evidence["cyrillic"] > evidence["latin"]:
            preferred = ("ru",)
        elif evidence["hangul"] > evidence["latin"]:
            preferred = ("ko",)
        elif evidence["latin"]:
            preferred = ("en", "ro")
        return preferred + tuple(
            language for language in self.candidate_languages if language not in preferred
        )

    def _decode_target_span(
        self,
        audio: Path,
        span: tuple[int, int],
        original_text: str,
        options: TranscriptionOptions,
        attempts: list[dict],
        *,
        _allow_isolation: bool = True,
    ) -> tuple[TranscriptSegment, ...] | None:
        start_ms, end_ms = span
        candidates: list[tuple[TranscriptionBackend, TranscriptionOptions]] = [
            (self.backend, options)
        ]
        independent_retry = (
            self.retry_backend
            if self.retry_backend is not None
            and self.retry_backend.identity != self.backend.identity
            else None
        )
        if independent_retry is None:
            attempts.append(
                {
                    "kind": "targeted-span-adjudication",
                    "source_start_ms": start_ms,
                    "source_end_ms": end_ms,
                    "status": "NO_INDEPENDENT_BACKEND_CONFIGURED",
                    "independent_backend_count": 0,
                    "selected": False,
                }
            )
            return None
        if independent_retry is not None:
            candidates.append((self.retry_backend, options))
        retry_backend = independent_retry or self.backend
        for language in self._target_languages(original_text):
            candidate_options = replace(options, language=language)
            for backend in (self.backend, retry_backend):
                if not any(
                    existing_backend.identity == backend.identity
                    and existing_options == candidate_options
                    for existing_backend, existing_options in candidates
                ):
                    candidates.append((backend, candidate_options))

        eligible: list[_SpanCandidate] = []
        with tempfile.TemporaryDirectory(prefix="dubbing-targeted-retry-") as directory:
            retry_audio = Path(directory) / "span.wav"
            marker = TranscriptSegment(start_ms, end_ms, "targeted retry span")
            self._extract_language_span(audio, marker, retry_audio)
            if _allow_isolation and self.targeted_retry_region_detector is not None:
                plan = self.targeted_retry_region_detector.detect(retry_audio)
                source_regions = tuple(
                    (region_start, region_end)
                    for item in plan.selected_regions
                    for region_start, region_end in (
                        (
                            max(start_ms, min(end_ms, start_ms + item.start_ms)),
                            max(start_ms, min(end_ms, start_ms + item.end_ms)),
                        ),
                    )
                    if region_end > region_start
                )
                attempts.append(
                    {
                        "kind": "targeted-speech-region-plan",
                        "source_start_ms": start_ms,
                        "source_end_ms": end_ms,
                        "plan": plan.to_dict(),
                        "source_regions": [
                            {"start_ms": region_start, "end_ms": region_end}
                            for region_start, region_end in source_regions
                        ],
                    }
                )
                total_region_ms = sum(
                    region_end - region_start for region_start, region_end in source_regions
                )
                use_isolation = bool(source_regions) and (
                    len(source_regions) > 1 or total_region_ms < 0.8 * (end_ms - start_ms)
                )
                if use_isolation:
                    replacements: list[TranscriptSegment] = []
                    cursor = start_ms
                    for region_start, region_end in source_regions:
                        if region_start - cursor >= 1000:
                            replacements.append(
                                TranscriptSegment(
                                    cursor,
                                    region_start,
                                    _UNISOLATED_SPAN_TEXT,
                                    uncertain=True,
                                )
                            )
                        replacement = self._decode_target_span(
                            audio,
                            (region_start, region_end),
                            original_text,
                            options,
                            attempts,
                            _allow_isolation=False,
                        )
                        if replacement is None:
                            replacements.append(
                                TranscriptSegment(
                                    region_start,
                                    region_end,
                                    _FAILED_SPAN_TEXT,
                                    uncertain=True,
                                )
                            )
                        else:
                            region_cursor = region_start
                            for item in sorted(
                                replacement,
                                key=lambda candidate: (
                                    candidate.start_ms,
                                    candidate.end_ms,
                                ),
                            ):
                                if item.start_ms - region_cursor >= 1000:
                                    replacements.append(
                                        TranscriptSegment(
                                            region_cursor,
                                            item.start_ms,
                                            _UNISOLATED_SPAN_TEXT,
                                            uncertain=True,
                                        )
                                    )
                                replacements.append(item)
                                region_cursor = max(region_cursor, item.end_ms)
                            if region_end - region_cursor >= 1000:
                                replacements.append(
                                    TranscriptSegment(
                                        region_cursor,
                                        region_end,
                                        _UNISOLATED_SPAN_TEXT,
                                        uncertain=True,
                                    )
                                )
                        cursor = max(cursor, region_end)
                    if end_ms - cursor >= 1000:
                        replacements.append(
                            TranscriptSegment(
                                cursor,
                                end_ms,
                                _UNISOLATED_SPAN_TEXT,
                                uncertain=True,
                            )
                        )
                    attempts.append(
                        {
                            "kind": "targeted-speech-region-result",
                            "source_start_ms": start_ms,
                            "source_end_ms": end_ms,
                            "selected_region_count": len(source_regions),
                            "selected_region_coverage_ms": total_region_ms,
                            "replacement_segment_count": len(replacements),
                            "uncertain_segment_count": sum(item.uncertain for item in replacements),
                        }
                    )
                    return tuple(
                        sorted(
                            replacements,
                            key=lambda item: (item.start_ms, item.end_ms),
                        )
                    )
            audio_candidates = build_audio_candidates(
                retry_audio,
                Path(directory) / "audio-candidates",
                policies=self.audio_candidate_policies,
                maximum_channels=self.maximum_audio_candidate_channels,
            )
            attempts.append(
                {
                    "kind": "targeted-audio-candidate-plan",
                    "source_start_ms": start_ms,
                    "source_end_ms": end_ms,
                    **audio_candidates.to_dict(),
                }
            )
            for audio_candidate in audio_candidates.candidates:
                for backend, retry_options in candidates:
                    candidate = annotate_transcript_languages(
                        backend.transcribe(audio_candidate.path, retry_options)
                    )
                    quality = evaluate_transcript_quality(
                        candidate, expected_duration_ms=end_ms - start_ms
                    )
                    attempts.append(
                        {
                            "kind": "targeted-span-redecode",
                            "source_start_ms": start_ms,
                            "source_end_ms": end_ms,
                            "backend": backend.identity,
                            "language": retry_options.language,
                            "audio_candidate": audio_candidate.identifier,
                            "audio_candidate_sha256": audio_candidate.candidate_sha256,
                            "audio_processing": audio_candidate.processing,
                            "quality": quality.to_dict(),
                            "candidate_text": candidate.text,
                        }
                    )
                    if (
                        not candidate.segments
                        or quality.status is not TranscriptQualityStatus.PASS
                        or any(item.uncertain for item in candidate.segments)
                    ):
                        continue
                    log_probabilities = [
                        item.diagnostics.avg_log_probability
                        for item in candidate.segments
                        if item.diagnostics and item.diagnostics.avg_log_probability is not None
                    ]
                    average_log_probability = (
                        sum(log_probabilities) / len(log_probabilities)
                        if log_probabilities
                        else -10.0
                    )
                    word_confidences = [
                        word.confidence
                        for item in candidate.segments
                        for word in item.words
                        if word.confidence is not None
                    ]
                    segment_confidences = [
                        item.confidence
                        for item in candidate.segments
                        if item.confidence is not None
                    ]
                    acoustic_confidences = word_confidences or segment_confidences
                    average_acoustic_confidence = (
                        sum(acoustic_confidences) / len(acoustic_confidences)
                        if acoustic_confidences
                        else None
                    )
                    attempts[-1]["average_acoustic_confidence"] = (
                        round(average_acoustic_confidence, 6)
                        if average_acoustic_confidence is not None
                        else None
                    )
                    shifted = tuple(item.shifted(start_ms) for item in candidate.segments)
                    eligible.append(
                        _SpanCandidate(
                            audio_candidate.identifier,
                            audio_candidate.candidate_sha256,
                            audio_candidate.processing,
                            backend.identity,
                            candidate.language or retry_options.language,
                            shifted,
                            candidate.text,
                            quality,
                            average_log_probability,
                            average_acoustic_confidence,
                            len(attempts) - 1,
                        )
                    )

        best_by_backend_language: dict[tuple[str, str, str | None], _SpanCandidate] = {}
        for candidate in eligible:
            key = (
                candidate.audio_candidate_id,
                candidate.backend_identity,
                candidate.language,
            )
            previous = best_by_backend_language.get(key)
            if previous is None or candidate.score > previous.score:
                best_by_backend_language[key] = candidate
        healthy_independent = [
            candidate
            for (_, identity, _), candidate in best_by_backend_language.items()
            if identity != self.backend.identity
        ]
        if not healthy_independent:
            attempts.append(
                {
                    "kind": "targeted-span-adjudication",
                    "source_start_ms": start_ms,
                    "source_end_ms": end_ms,
                    "status": "NO_HEALTHY_INDEPENDENT_CANDIDATE",
                    "independent_backend_count": 0,
                    "agreement_threshold": _INDEPENDENT_AGREEMENT_THRESHOLD,
                    "minimum_acoustic_confidence": (_INDEPENDENT_MINIMUM_ACOUSTIC_CONFIDENCE),
                    "minimum_consensus_tokens": (_INDEPENDENT_MINIMUM_CONSENSUS_TOKENS),
                    "selected": False,
                }
            )
            return None

        paired_candidates = []
        for independent_candidate in healthy_independent:
            primary_candidate = best_by_backend_language.get(
                (
                    independent_candidate.audio_candidate_id,
                    self.backend.identity,
                    independent_candidate.language,
                )
            )
            agreement = (
                token_agreement(primary_candidate.text, independent_candidate.text)
                if primary_candidate is not None
                else None
            )
            confidence_floor = (
                min(
                    primary_candidate.average_acoustic_confidence,
                    independent_candidate.average_acoustic_confidence,
                )
                if primary_candidate is not None
                and primary_candidate.average_acoustic_confidence is not None
                and independent_candidate.average_acoustic_confidence is not None
                else None
            )
            consensus_token_count = (
                min(
                    len(word_tokens(primary_candidate.text)),
                    len(word_tokens(independent_candidate.text)),
                )
                if primary_candidate is not None
                else 0
            )
            paired_candidates.append(
                (
                    agreement,
                    confidence_floor,
                    consensus_token_count,
                    primary_candidate,
                    independent_candidate,
                )
            )
        corroborated = [
            item
            for item in paired_candidates
            if item[0] is not None
            and item[0] >= _INDEPENDENT_AGREEMENT_THRESHOLD
            and (item[1] is None or item[1] >= _INDEPENDENT_MINIMUM_ACOUSTIC_CONFIDENCE)
            and item[2] >= _INDEPENDENT_MINIMUM_CONSENSUS_TOKENS
        ]
        best_corroborated_by_audio = {}
        for item in corroborated:
            audio_candidate_id = item[4].audio_candidate_id
            incumbent = best_corroborated_by_audio.get(audio_candidate_id)
            ranking = (
                item[0] if item[0] is not None else -1.0,
                item[1] if item[1] is not None else -1.0,
                item[2],
                item[4].score,
            )
            if incumbent is None or ranking > (
                incumbent[0] if incumbent[0] is not None else -1.0,
                incumbent[1] if incumbent[1] is not None else -1.0,
                incumbent[2],
                incumbent[4].score,
            ):
                best_corroborated_by_audio[audio_candidate_id] = item
        raw_corroborated = [
            item
            for audio_candidate_id, item in best_corroborated_by_audio.items()
            if audio_candidate_id == "raw"
        ]
        processed_corroborated = [
            item
            for audio_candidate_id, item in best_corroborated_by_audio.items()
            if audio_candidate_id != "raw"
        ]
        divergent_audio_candidates = False
        if raw_corroborated:
            ranked = raw_corroborated
        elif processed_corroborated:
            for index, left in enumerate(processed_corroborated):
                for right in processed_corroborated[index + 1 :]:
                    if token_agreement(left[4].text, right[4].text) < (
                        _INDEPENDENT_AGREEMENT_THRESHOLD
                    ):
                        divergent_audio_candidates = True
            ranked = processed_corroborated
        else:
            ranked = paired_candidates
        (
            agreement,
            confidence_floor,
            consensus_token_count,
            primary_candidate,
            selected,
        ) = max(
            ranked,
            key=lambda item: (
                item[0] if item[0] is not None else -1.0,
                item[1] if item[1] is not None else -1.0,
                item[2],
                item[4].score,
            ),
        )
        consensus = (
            agreement is not None
            and agreement >= _INDEPENDENT_AGREEMENT_THRESHOLD
            and (
                confidence_floor is None
                or confidence_floor >= _INDEPENDENT_MINIMUM_ACOUSTIC_CONFIDENCE
            )
            and consensus_token_count >= _INDEPENDENT_MINIMUM_CONSENSUS_TOKENS
            and not divergent_audio_candidates
        )
        status = (
            "CONSENSUS_PASS"
            if consensus
            else (
                "AUDIO_CANDIDATE_DISAGREEMENT"
                if divergent_audio_candidates
                else "INDEPENDENT_DISAGREEMENT"
                if primary_candidate is not None
                else "INDEPENDENT_CANDIDATE_UNCORROBORATED"
            )
        )
        selected_segments = selected.segments
        if not consensus:
            selected_segments = (
                TranscriptSegment(
                    start_ms,
                    end_ms,
                    _DISAGREEMENT_SPAN_TEXT,
                    uncertain=True,
                ),
            )
        attempts[selected.attempt_index]["selected"] = True
        attempts.append(
            {
                "kind": "targeted-span-adjudication",
                "source_start_ms": start_ms,
                "source_end_ms": end_ms,
                "status": status,
                "primary_backend": self.backend.identity,
                "primary_language": (
                    primary_candidate.language if primary_candidate is not None else None
                ),
                "selected_backend": selected.backend_identity,
                "selected_language": selected.language,
                "selected_audio_candidate": selected.audio_candidate_id,
                "selected_audio_candidate_sha256": selected.audio_candidate_sha256,
                "selected_audio_processing": selected.audio_processing,
                "raw_consensus_available": bool(raw_corroborated),
                "processed_consensus_count": len(processed_corroborated),
                "divergent_audio_candidates": divergent_audio_candidates,
                "independent_backend_count": len(
                    {item.backend_identity for item in healthy_independent}
                ),
                "agreement": round(agreement, 6) if agreement is not None else None,
                "agreement_threshold": _INDEPENDENT_AGREEMENT_THRESHOLD,
                "acoustic_confidence_floor": (
                    round(confidence_floor, 6) if confidence_floor is not None else None
                ),
                "minimum_acoustic_confidence": (_INDEPENDENT_MINIMUM_ACOUSTIC_CONFIDENCE),
                "consensus_token_count": consensus_token_count,
                "minimum_consensus_tokens": (_INDEPENDENT_MINIMUM_CONSENSUS_TOKENS),
                "selected_uncertain": not consensus,
            }
        )
        return selected_segments

    def _repair_rejected_spans(
        self,
        audio: Path,
        result: TranscriptionResult,
        quality: TranscriptQualityReport,
        expected_duration_ms: int,
        options: TranscriptionOptions,
        attempts: list[dict],
    ) -> tuple[TranscriptionResult, TranscriptQualityReport]:
        duration_ms = expected_duration_ms
        silence_centres = detect_silence_centres(
            audio, minimum_silence_seconds=self.minimum_silence_seconds
        )
        repaired = result
        report = quality
        for round_index in range(2):
            intervals = self._critical_failure_intervals(repaired, report)
            if not intervals:
                break
            spans = self._bounded_retry_spans(
                intervals,
                duration_ms=duration_ms,
                silence_centres=silence_centres,
            )
            replacements: list[TranscriptSegment] = []
            for span in spans:
                original_text = " ".join(
                    segment.text
                    for segment in repaired.segments
                    if segment.start_ms < span[1] and segment.end_ms > span[0]
                )
                replacement = self._decode_target_span(
                    audio, span, original_text, options, attempts
                )
                if replacement is None:
                    replacements.append(
                        TranscriptSegment(
                            span[0],
                            span[1],
                            _FAILED_SPAN_TEXT,
                            uncertain=True,
                        )
                    )
                else:
                    replacements.extend(replacement)
            repaired = self._splice_retry_spans(repaired, spans, replacements)
            report = evaluate_transcript_quality(repaired, expected_duration_ms=duration_ms)
            attempts.append(
                {
                    "kind": "targeted-repair-round",
                    "round": round_index + 1,
                    "spans": [
                        {"start_ms": start_ms, "end_ms": end_ms} for start_ms, end_ms in spans
                    ],
                    "quality": report.to_dict(),
                }
            )
            if report.status not in {
                TranscriptQualityStatus.FAILED,
                TranscriptQualityStatus.REPROCESS_REQUIRED,
            }:
                break
        return repaired, report

    def _decode_chunk(
        self,
        audio: Path,
        chunk: AdaptiveChunk,
        options: TranscriptionOptions,
        *,
        energy_silence_coverage_ms: int = 0,
    ) -> tuple[TranscriptionResult, dict]:
        attempts: list[dict] = []
        duration_ms = chunk.extract_end_ms - chunk.extract_start_ms
        if (
            energy_silence_coverage_ms == duration_ms
            and self.silence_verification_detector is not None
        ):
            plan = self.silence_verification_detector.detect(audio)
            silence_attempt = {
                "kind": "chunk-silence-classification",
                "status": (
                    "CONFIRMED_SILENCE" if not plan.selected_regions else "SEMANTIC_SPEECH_DETECTED"
                ),
                "policy": _SILENCE_ADMISSION_POLICY,
                "energy_silence_coverage_ms": energy_silence_coverage_ms,
                "energy_silence_coverage_ratio": 1.0,
                "speech_region_plan": plan.to_dict(),
            }
            attempts.append(silence_attempt)
            if not plan.selected_regions:
                result = TranscriptionResult(
                    segments=(),
                    text="",
                    backend=self.backend.identity.split(":", 1)[0],
                    model=self.backend.identity,
                    device="adaptive-local",
                    language=options.language,
                    duration_ms=duration_ms,
                    confidence_available=False,
                    provenance={
                        "coordinator": _COORDINATOR_VERSION,
                        "classification": "confirmed_silence",
                        "silence_admission_policy": _SILENCE_ADMISSION_POLICY,
                        "energy_silence_coverage_ms": energy_silence_coverage_ms,
                        "silence_verification_detector": plan.detector_identity,
                    },
                )
                return result, {
                    "attempts": attempts,
                    "selected_attempt": 0,
                    "targeted_retry_exhausted": False,
                }
        result = annotate_transcript_languages(self.backend.transcribe(audio, options))
        raw_quality = evaluate_transcript_quality(result, expected_duration_ms=duration_ms)
        result = self._enforce_audio_bounds(result, duration_ms, raw_quality, attempts)
        result = self._retry_detected_chunk_language(audio, result, options, duration_ms, attempts)
        result = self._resolve_uncertain_turns(
            audio,
            result,
            self.backend,
            options,
            attempts,
        )
        quality = evaluate_transcript_quality(result, expected_duration_ms=duration_ms)
        attempts.append(
            {
                "kind": "chunk",
                "backend": self.backend.identity,
                "language": options.language,
                "quality": quality.to_dict(),
            }
        )
        if quality.status in {
            TranscriptQualityStatus.FAILED,
            TranscriptQualityStatus.REPROCESS_REQUIRED,
        }:
            result, quality = self._repair_rejected_spans(
                audio, result, quality, duration_ms, options, attempts
            )
        selected_attempt = None
        if quality.status not in {
            TranscriptQualityStatus.FAILED,
            TranscriptQualityStatus.REPROCESS_REQUIRED,
        }:
            selected_attempt = len(attempts) - 1
        return result, {
            "attempts": attempts,
            "selected_attempt": selected_attempt,
            "targeted_retry_exhausted": selected_attempt is None,
        }

    @staticmethod
    def _enforce_audio_bounds(
        result: TranscriptionResult,
        duration_ms: int,
        raw_quality: TranscriptQualityReport,
        attempts: list[dict],
    ) -> TranscriptionResult:
        """Reject decoder timestamps that cannot refer to the extracted audio."""
        normalized: list[TranscriptSegment] = []
        changes: list[dict] = []
        for segment_index, segment in enumerate(result.segments):
            if segment.start_ms >= duration_ms:
                changes.append(
                    {
                        "action": "DROPPED_WHOLE_SEGMENT",
                        "segment_index": segment_index,
                        "observed_start_ms": segment.start_ms,
                        "observed_end_ms": segment.end_ms,
                    }
                )
                continue
            if segment.end_ms <= duration_ms:
                normalized.append(segment)
                continue
            words = []
            for word in segment.words:
                midpoint_ms = word.start_ms + (word.end_ms - word.start_ms) // 2
                if not 0 <= midpoint_ms < duration_ms:
                    continue
                start_ms = max(0, word.start_ms)
                end_ms = min(duration_ms, word.end_ms)
                if end_ms > start_ms:
                    words.append(replace(word, start_ms=start_ms, end_ms=end_ms))
            text = segment.text
            if segment.words and words:
                text = "".join(word.text for word in words).strip()
            normalized.append(
                replace(
                    segment,
                    end_ms=duration_ms,
                    text=text,
                    words=tuple(words),
                )
            )
            changes.append(
                {
                    "action": "CLAMPED_OVERLAPPING_SEGMENT",
                    "segment_index": segment_index,
                    "observed_start_ms": segment.start_ms,
                    "observed_end_ms": segment.end_ms,
                    "normalized_start_ms": segment.start_ms,
                    "normalized_end_ms": duration_ms,
                    "dropped_word_count": len(segment.words) - len(words),
                }
            )
        if not changes:
            return result
        attempts.append(
            {
                "kind": "out-of-audio-segment-rejection",
                "policy": _AUDIO_BOUNDS_POLICY,
                "audio_duration_ms": duration_ms,
                "raw_quality": raw_quality.to_dict(),
                "changes": changes,
            }
        )
        segments = tuple(normalized)
        return replace(
            result,
            segments=segments,
            text=" ".join(segment.text for segment in segments),
            duration_ms=duration_ms,
        )

    @staticmethod
    def _average_log_probability(result: TranscriptionResult) -> float | None:
        values = [
            segment.diagnostics.avg_log_probability
            for segment in result.segments
            if segment.diagnostics and segment.diagnostics.avg_log_probability is not None
        ]
        return sum(values) / len(values) if values else None

    def _retry_detected_chunk_language(
        self,
        audio: Path,
        automatic: TranscriptionResult,
        options: TranscriptionOptions,
        duration_ms: int,
        attempts: list[dict],
    ) -> TranscriptionResult:
        if options.language is not None:
            return automatic
        language = automatic.language
        mode = self.language_retry_policy.get(language or "")
        if not mode:
            return automatic
        forced = annotate_transcript_languages(
            self.backend.transcribe(audio, replace(options, language=language))
        )
        forced_quality = evaluate_transcript_quality(forced, expected_duration_ms=duration_ms)
        automatic_score = self._average_log_probability(automatic)
        forced_score = self._average_log_probability(forced)
        eligible = forced_quality.status not in {
            TranscriptQualityStatus.FAILED,
            TranscriptQualityStatus.REPROCESS_REQUIRED,
        }
        selected = eligible and (
            mode == "always"
            or (
                forced_score is not None
                and automatic_score is not None
                and forced_score > automatic_score
            )
        )
        attempts.append(
            {
                "kind": "detected-chunk-language-redecode",
                "language": language,
                "mode": mode,
                "automatic_avg_log_probability": automatic_score,
                "forced_avg_log_probability": forced_score,
                "forced_quality": forced_quality.to_dict(),
                "selected": selected,
            }
        )
        return forced if selected else automatic

    @staticmethod
    def _extract_language_span(
        source: Path,
        segment: TranscriptSegment,
        destination: Path,
    ) -> None:
        ffmpeg = ffmpeg_executable()
        if ffmpeg is None:
            if source.suffix.casefold() == ".wav":
                _extract_pcm_wave(source, segment.start_ms, segment.end_ms, destination)
                return
            raise TranscriptionError("ffmpeg is required for non-WAV language retry")
        process = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{segment.start_ms / 1000:.3f}",
                "-i",
                str(source),
                "-t",
                f"{(segment.end_ms - segment.start_ms) / 1000:.3f}",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(destination),
            ],
            capture_output=True,
            text=True,
            timeout=max(120, (segment.end_ms - segment.start_ms) // 1000),
        )
        if process.returncode:
            raise TranscriptionError(f"language retry extraction failed: {process.stderr.strip()}")

    @staticmethod
    def _crop_language_candidate(
        candidate: TranscriptionResult,
        target_start_ms: int,
        target_end_ms: int,
    ) -> TranscriptionResult:
        """Keep only target-turn speech and normalize it to the target interval."""
        segments = []
        for segment in candidate.segments:
            words = []
            for word in segment.words:
                midpoint = word.start_ms + (word.end_ms - word.start_ms) // 2
                if not target_start_ms <= midpoint < target_end_ms:
                    continue
                word_start = max(word.start_ms, target_start_ms)
                word_end = min(word.end_ms, target_end_ms)
                if word_end > word_start:
                    words.append(
                        replace(
                            word,
                            start_ms=word_start - target_start_ms,
                            end_ms=word_end - target_start_ms,
                        )
                    )
            if segment.words:
                if not words:
                    continue
                start_ms = min(word.start_ms for word in words) + target_start_ms
                end_ms = max(word.end_ms for word in words) + target_start_ms
                text = "".join(word.text for word in words).strip()
            else:
                if segment.start_ms < target_start_ms or segment.end_ms > target_end_ms:
                    continue
                start_ms = segment.start_ms
                end_ms = segment.end_ms
                text = segment.text
            segments.append(
                replace(
                    segment,
                    start_ms=start_ms - target_start_ms,
                    end_ms=end_ms - target_start_ms,
                    text=text,
                    words=tuple(words),
                )
            )
        normalized = tuple(sorted(segments, key=lambda item: (item.start_ms, item.end_ms)))
        return replace(
            candidate,
            segments=normalized,
            text=" ".join(item.text for item in normalized),
            duration_ms=target_end_ms - target_start_ms,
        )

    def _resolve_uncertain_turns(
        self,
        audio: Path,
        result: TranscriptionResult,
        backend: TranscriptionBackend,
        base_options: TranscriptionOptions,
        attempts: list[dict],
    ) -> TranscriptionResult:
        resolved: list[TranscriptSegment] = []
        changed = False
        for segment in result.segments:
            if not segment.uncertain:
                resolved.append(segment)
                continue
            evidence = script_evidence(segment.text)
            if evidence["cyrillic"] > evidence["latin"]:
                languages = ("ru",)
            elif evidence["hangul"] > evidence["latin"]:
                languages = ("ko",)
            else:
                languages = ()
            if not languages:
                resolved.append(segment)
                continue
            if result.duration_ms is not None and segment.end_ms > result.duration_ms:
                attempts.append(
                    {
                        "kind": "turn-language-redecode-skipped",
                        "source_start_ms": segment.start_ms,
                        "source_end_ms": segment.end_ms,
                        "audio_duration_ms": result.duration_ms,
                        "reason": "TARGET_OUTSIDE_AUDIO_DURATION",
                    }
                )
                resolved.append(segment)
                continue
            eligible_replacements: list[
                tuple[tuple[float, ...], str, tuple[TranscriptSegment, ...]]
            ] = []
            with tempfile.TemporaryDirectory(prefix="dubbing-language-retry-") as directory:
                span = Path(directory) / "span.wav"
                context_start_ms = max(0, segment.start_ms - _LANGUAGE_RETRY_CONTEXT_MS)
                context_end_ms = min(
                    result.duration_ms or segment.end_ms + _LANGUAGE_RETRY_CONTEXT_MS,
                    segment.end_ms + _LANGUAGE_RETRY_CONTEXT_MS,
                )
                self._extract_language_span(
                    audio,
                    replace(
                        segment,
                        start_ms=context_start_ms,
                        end_ms=context_end_ms,
                    ),
                    span,
                )
                target_start_ms = segment.start_ms - context_start_ms
                target_end_ms = segment.end_ms - context_start_ms
                for language in languages:
                    candidate = self._crop_language_candidate(
                        backend.transcribe(span, replace(base_options, language=language)),
                        target_start_ms,
                        target_end_ms,
                    )
                    candidate = annotate_transcript_languages(candidate)
                    quality = evaluate_transcript_quality(
                        candidate,
                        expected_duration_ms=segment.end_ms - segment.start_ms,
                    )
                    attempts.append(
                        {
                            "kind": "turn-language-redecode",
                            "source_start_ms": segment.start_ms,
                            "source_end_ms": segment.end_ms,
                            "context_start_ms": context_start_ms,
                            "context_end_ms": context_end_ms,
                            "backend": backend.identity,
                            "language": language,
                            "quality": quality.to_dict(),
                        }
                    )
                    if (
                        candidate.segments
                        and not any(item.uncertain for item in candidate.segments)
                        and quality.status
                        not in {
                            TranscriptQualityStatus.FAILED,
                            TranscriptQualityStatus.REPROCESS_REQUIRED,
                        }
                    ):
                        candidate_segments = tuple(
                            item.shifted(segment.start_ms) for item in candidate.segments
                        )
                        candidate_text = " ".join(item.text for item in candidate.segments)
                        candidate_script = script_evidence(candidate_text)
                        letters = max(1, sum(candidate_script.values()))
                        expected_script = {
                            "ru": "cyrillic",
                            "ko": "hangul",
                            "en": "latin",
                            "ro": "latin",
                        }[language]
                        script_ratio = candidate_script[expected_script] / letters
                        log_probabilities = [
                            item.diagnostics.avg_log_probability
                            for item in candidate.segments
                            if item.diagnostics and item.diagnostics.avg_log_probability is not None
                        ]
                        average_log_probability = (
                            sum(log_probabilities) / len(log_probabilities)
                            if log_probabilities
                            else -10.0
                        )
                        language_matches = sum(
                            item.language == language for item in candidate.segments
                        ) / len(candidate.segments)
                        status_score = {
                            TranscriptQualityStatus.PASS: 2.0,
                            TranscriptQualityStatus.HUMAN_REVIEW_REQUIRED: 1.0,
                            TranscriptQualityStatus.PASS_WITH_UNCERTAIN_SPANS: 0.0,
                        }.get(quality.status, -10.0)
                        score = (
                            status_score,
                            script_ratio,
                            language_matches,
                            average_log_probability,
                        )
                        attempts[-1]["candidate_score"] = list(score)
                        eligible_replacements.append((score, language, candidate_segments))
            if eligible_replacements:
                _, selected_language, replacement = max(
                    eligible_replacements, key=lambda item: item[0]
                )
                attempts.append(
                    {
                        "kind": "turn-language-selection",
                        "source_start_ms": segment.start_ms,
                        "source_end_ms": segment.end_ms,
                        "selected_language": selected_language,
                    }
                )
                resolved.extend(replacement)
                changed = True
            else:
                resolved.append(segment)
        if not changed:
            return result
        ordered = tuple(sorted(resolved, key=lambda item: (item.start_ms, item.end_ms)))
        return replace(result, segments=ordered, text=" ".join(item.text for item in ordered))

    def run(
        self,
        media: str | Path,
        options: TranscriptionOptions | None = None,
        *,
        source_digest: str | None = None,
        processing_end_ms: int | None = None,
    ) -> tuple[TranscriptionResult, dict]:
        source = Path(media).resolve()
        if not source.is_file():
            raise TranscriptionError(f"media input not found: {source}")
        options = options or TranscriptionOptions()
        digest = source_digest or source_sha256(source)
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise TranscriptionError("source_digest must be a lowercase SHA-256 digest")
        probe = probe_media(source)
        if processing_end_ms is None:
            effective_duration_ms = probe.duration_ms
        elif (
            isinstance(processing_end_ms, bool)
            or not isinstance(processing_end_ms, int)
            or processing_end_ms <= 0
            or processing_end_ms > probe.duration_ms
        ):
            raise TranscriptionError(
                "processing_end_ms must satisfy 0 < processing_end_ms <= source duration"
            )
        else:
            effective_duration_ms = processing_end_ms
        detected_silence_intervals = detect_silence_intervals(
            source, minimum_silence_seconds=self.minimum_silence_seconds
        )
        silence_intervals = tuple(
            (max(0, start_ms), min(effective_duration_ms, end_ms))
            for start_ms, end_ms in detected_silence_intervals
            if start_ms < effective_duration_ms and end_ms > 0
        )
        silence_centres = tuple(
            round((start_ms + end_ms) / 2) for start_ms, end_ms in silence_intervals
        )
        chunks = self.planner.plan(
            effective_duration_ms,
            silence_centres,
            silence_intervals=silence_intervals,
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        expected_manifest = self._manifest(
            source,
            digest,
            probe,
            chunks,
            effective_duration_ms,
        )
        self._admit_manifest(expected_manifest)
        manifest_sha256 = _document_sha256(expected_manifest)
        streams = probe.audio_streams
        completed: list[tuple[AdaptiveChunk, TranscriptionResult]] = []
        for chunk in chunks:
            checkpoint = self.checkpoint_dir / "chunks" / f"{chunk.index:06d}.json"
            receipt_path = self.checkpoint_dir / "receipts" / f"{chunk.index:06d}.json"
            checkpoint_exists = checkpoint.is_file()
            receipt_exists = receipt_path.is_file()
            if checkpoint_exists != receipt_exists:
                raise TranscriptionError(
                    f"adaptive chunk checkpoint pair is incomplete: {chunk.index:06d}"
                )
            if checkpoint_exists:
                try:
                    checkpoint_document = json.loads(checkpoint.read_text(encoding="utf-8"))
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    result = transcription_result_from_dict(checkpoint_document)
                except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise TranscriptionError(
                        f"malformed adaptive chunk checkpoint pair: {checkpoint.name}"
                    ) from exc
                expected_bindings = {
                    "schema_version": _CHUNK_RECEIPT_SCHEMA,
                    "coordinator_version": _COORDINATOR_VERSION,
                    "quality_policy_version": QUALITY_POLICY_VERSION,
                    "full_source_sha256": digest,
                    "manifest_sha256": manifest_sha256,
                    "chunk": chunk.to_dict(),
                }
                for name, expected_value in expected_bindings.items():
                    if receipt.get(name) != expected_value:
                        raise TranscriptionError(
                            f"adaptive chunk receipt binding mismatch: {chunk.index:06d}:{name}"
                        )
                if receipt.get("checkpoint_document_sha256") != _document_sha256(
                    checkpoint_document
                ):
                    raise TranscriptionError(
                        f"adaptive chunk checkpoint document hash mismatch: {chunk.index:06d}"
                    )
                extracted_audio_sha256 = receipt.get("extracted_audio_sha256")
                if not isinstance(extracted_audio_sha256, str) or not re.fullmatch(
                    r"[a-f0-9]{64}", extracted_audio_sha256
                ):
                    raise TranscriptionError(
                        f"adaptive chunk extracted audio hash is malformed: {chunk.index:06d}"
                    )
                if (
                    result.source_sha256 is not None
                    and result.source_sha256 != extracted_audio_sha256
                ):
                    raise TranscriptionError(
                        f"adaptive chunk backend source hash mismatch: {chunk.index:06d}"
                    )
                with tempfile.TemporaryDirectory(
                    prefix="dubbing-adaptive-replay-"
                ) as directory:
                    replay_audio = Path(directory) / f"{chunk.index:06d}.wav"
                    self._extract(source, streams, chunk, replay_audio)
                    if source_sha256(replay_audio) != extracted_audio_sha256:
                        raise TranscriptionError(
                            f"adaptive chunk extracted audio hash mismatch: {chunk.index:06d}"
                        )
            else:
                result = None
            if result is None:
                with tempfile.TemporaryDirectory(prefix="dubbing-adaptive-") as directory:
                    audio = Path(directory) / f"{chunk.index:06d}.wav"
                    self._extract(source, streams, chunk, audio)
                    extracted_audio_sha256 = source_sha256(audio)
                    duration_ms = chunk.extract_end_ms - chunk.extract_start_ms
                    energy_silence_coverage_ms = _interval_coverage_ms(
                        chunk.extract_start_ms,
                        chunk.extract_end_ms,
                        silence_intervals,
                    )
                    result, receipt = self._decode_chunk(
                        audio,
                        chunk,
                        options,
                        energy_silence_coverage_ms=(
                            energy_silence_coverage_ms
                            if energy_silence_coverage_ms == duration_ms
                            else 0
                        ),
                    )
                    if (
                        result.source_sha256 is not None
                        and result.source_sha256 != extracted_audio_sha256
                    ):
                        raise TranscriptionError(
                            f"adaptive chunk backend source hash mismatch: {chunk.index:06d}"
                        )
                checkpoint_document = result.to_dict()
                receipt = dict(receipt)
                receipt.update(
                    {
                        "schema_version": _CHUNK_RECEIPT_SCHEMA,
                        "coordinator_version": _COORDINATOR_VERSION,
                        "quality_policy_version": QUALITY_POLICY_VERSION,
                        "full_source_sha256": digest,
                        "manifest_sha256": manifest_sha256,
                        "chunk": chunk.to_dict(),
                        "extracted_audio_sha256": extracted_audio_sha256,
                        "checkpoint_document_sha256": _document_sha256(
                            checkpoint_document
                        ),
                    }
                )
                _atomic_json(checkpoint, checkpoint_document)
                _atomic_json(receipt_path, receipt)
            completed.append((chunk, result))
            _atomic_json(
                self.checkpoint_dir / "progress.json",
                {
                    "schema_version": "dubbing.adaptive-transcription-progress.v1",
                    "completed_chunks": chunk.index + 1,
                    "total_chunks": len(chunks),
                    "completed_through_ms": chunk.end_ms,
                },
            )
        segments = reconcile_chunks(completed)
        result = TranscriptionResult(
            segments=segments,
            text=" ".join(item.text for item in segments),
            backend=self.backend.identity.split(":", 1)[0],
            model=self.backend.identity,
            device="adaptive-local",
            language=options.language,
            duration_ms=effective_duration_ms,
            confidence_available=any(item.confidence is not None for item in segments),
            source_sha256=digest,
            provenance={
                "coordinator": _COORDINATOR_VERSION,
                "chunk_count": len(chunks),
                "processing_scope": {
                    "start_ms": 0,
                    "end_ms": effective_duration_ms,
                    "processed_duration_ms": effective_duration_ms,
                    "full_source_duration_ms": probe.duration_ms,
                    "full_source_sha256_bound": True,
                },
                "audio_streams": [stream.to_dict() for stream in streams],
                "channel_preservation": "all-streams-merged-with-discrete-channels-v1",
            },
        )
        quality = evaluate_transcript_quality(result, expected_duration_ms=effective_duration_ms)
        if any(issue.code == "large_unexplained_gaps" for issue in quality.issues):
            quality = evaluate_transcript_quality(
                result,
                expected_duration_ms=effective_duration_ms,
                known_silence_intervals=silence_intervals,
            )
        _atomic_json(self.checkpoint_dir / "result.json", result.to_dict())
        _atomic_json(self.checkpoint_dir / "quality-report.json", quality.to_dict())
        return result, quality.to_dict()
