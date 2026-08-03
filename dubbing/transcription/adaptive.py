from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from dubbing.media import ffmpeg_executable
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
    TranscriptQualityReport,
    TranscriptQualityStatus,
    evaluate_transcript_quality,
)


_FAILED_SPAN_TEXT = "[UNCERTAIN: LOCAL TRANSCRIPTION FAILED]"
_TARGET_RETRY_MIN_MS = 20_000
_TARGET_RETRY_TARGET_MS = 45_000
_TARGET_RETRY_MAX_MS = 60_000
_TARGET_RETRY_PADDING_MS = 2_000


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


def probe_media(path: str | Path) -> MediaProbe:
    source = Path(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise TranscriptionError("ffprobe is required for adaptive transcription")
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
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        raise TranscriptionError(f"could not probe media: {source.name}") from exc
    if duration_ms <= 0 or not streams:
        raise TranscriptionError("media must contain at least one positive-duration audio stream")
    return MediaProbe(duration_ms, size_bytes, streams)


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


def _detect_silence_seconds(
    path: str | Path,
    *,
    minimum_silence_seconds: float,
    noise_db: float,
) -> tuple[tuple[float, float], ...]:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise TranscriptionError("ffmpeg is required for silence-aware chunking")
    process = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-i",
            str(path),
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


class AdaptiveChunkPlanner:
    def __init__(
        self,
        *,
        target_seconds: int = 240,
        minimum_seconds: int = 60,
        maximum_seconds: int = 480,
        overlap_seconds: float = 2.0,
    ) -> None:
        if not 30 <= minimum_seconds <= target_seconds <= maximum_seconds <= 1800:
            raise ValueError("chunk bounds must satisfy 30 <= minimum <= target <= maximum <= 1800")
        if overlap_seconds < 0 or overlap_seconds >= minimum_seconds / 2:
            raise ValueError("invalid overlap_seconds")
        self.target_ms = target_seconds * 1000
        self.minimum_ms = minimum_seconds * 1000
        self.maximum_ms = maximum_seconds * 1000
        self.overlap_ms = round(overlap_seconds * 1000)

    def plan(self, duration_ms: int, silence_centres: tuple[int, ...]) -> tuple[AdaptiveChunk, ...]:
        boundaries = [0]
        cursor = 0
        silences = sorted(point for point in silence_centres if 0 < point < duration_ms)
        reasons: list[str] = []
        while duration_ms - cursor > self.maximum_ms:
            minimum = cursor + self.minimum_ms
            maximum = min(duration_ms, cursor + self.maximum_ms)
            target = cursor + self.target_ms
            candidates = [point for point in silences if minimum <= point <= maximum]
            if candidates:
                boundary = min(candidates, key=lambda point: (abs(point - target), point))
                reason = "silence"
            else:
                boundary = maximum
                reason = "maximum-duration"
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
        candidate_languages: tuple[str, ...] = ("en", "ru", "ro", "ko"),
    ) -> None:
        self.backend = backend
        self.retry_backend = retry_backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.planner = planner or AdaptiveChunkPlanner()
        self.candidate_languages = candidate_languages

    @staticmethod
    def _extract(source: Path, stream: AudioStream, chunk: AdaptiveChunk, output: Path) -> None:
        ffmpeg = ffmpeg_executable()
        if ffmpeg is None:
            raise TranscriptionError("ffmpeg is required for adaptive transcription")
        process = subprocess.run(
            [
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
                "-map",
                f"0:{stream.index}",
                "-vn",
                "-ac",
                str(stream.channels),
                "-ar",
                str(stream.sample_rate_hz),
                "-c:a",
                "pcm_s16le",
                "-y",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=max(300, (chunk.extract_end_ms - chunk.extract_start_ms) // 1000),
        )
        if process.returncode:
            raise TranscriptionError(f"chunk extraction failed: {process.stderr.strip()}")

    def _manifest(self, source: Path, digest: str, probe: MediaProbe, chunks: tuple[AdaptiveChunk, ...]) -> dict:
        return {
            "schema_version": "dubbing.adaptive-transcription-checkpoint.v1",
            "coordinator_version": "adaptive-long-form-v4",
            "source_name": source.name,
            "source_sha256": digest,
            "probe": probe.to_dict(),
            "backend_identity": self.backend.identity,
            "retry_backend_identity": self.retry_backend.identity if self.retry_backend else None,
            "candidate_languages": list(self.candidate_languages),
            "targeted_retry": {
                "minimum_ms": _TARGET_RETRY_MIN_MS,
                "target_ms": _TARGET_RETRY_TARGET_MS,
                "maximum_ms": _TARGET_RETRY_MAX_MS,
                "padding_ms": _TARGET_RETRY_PADDING_MS,
                "maximum_rounds": 2,
            },
            "planner": self.planner.to_dict(),
            "chunks": [item.to_dict() for item in chunks],
        }

    def _admit_manifest(self, expected: dict) -> None:
        path = self.checkpoint_dir / "manifest.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TranscriptionError("adaptive checkpoint manifest is malformed") from exc
            compatible_v3 = existing.get("coordinator_version") == "adaptive-long-form-v3"
            if compatible_v3:
                migrated = dict(existing)
                migrated["coordinator_version"] = expected["coordinator_version"]
                migrated["targeted_retry"] = expected["targeted_retry"]
                compatible_v3 = migrated == expected
            if existing != expected and not compatible_v3:
                raise TranscriptionError("adaptive checkpoint does not match source or configuration")
            if compatible_v3:
                _atomic_json(path, expected)
        else:
            _atomic_json(path, expected)

    @staticmethod
    def _has_repeated_tokens(text: str) -> bool:
        tokens = _normalize(text).split()
        run = 1
        for previous, current in zip(tokens, tokens[1:]):
            run = run + 1 if current == previous else 1
            if run >= 8:
                return True
        for width in range(1, min(6, len(tokens)) + 1):
            for index in range(0, len(tokens) - (2 * width) + 1):
                phrase = tokens[index : index + width]
                repeats = 1
                cursor = index + width
                while tokens[cursor : cursor + width] == phrase:
                    repeats += 1
                    cursor += width
                if repeats >= 6:
                    return True
        return False

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
        intervals: list[tuple[int, int]] = []
        previous: TranscriptSegment | None = None
        for segment in result.segments:
            if (
                AdaptiveLongFormCoordinator._has_repeated_tokens(segment.text)
                or segment.text == _FAILED_SPAN_TEXT
                or (segment.diagnostics and segment.diagnostics.fallback_exhausted)
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
                        (min(previous.start_ms, segment.start_ms), max(previous.end_ms, segment.end_ms))
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
            if (
                duration_ms >= _TARGET_RETRY_MIN_MS
                and end_ms - start_ms < _TARGET_RETRY_MIN_MS
            ):
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
                    if cursor + _TARGET_RETRY_MIN_MS
                    <= point
                    <= cursor + _TARGET_RETRY_MAX_MS
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
            if not any(segment.start_ms < end_ms and segment.end_ms > start_ms for start_ms, end_ms in spans)
        ]
        segments = tuple(
            sorted(retained + replacements, key=lambda item: (item.start_ms, item.end_ms))
        )
        return replace(original, segments=segments, text=" ".join(item.text for item in segments))

    def _target_languages(self, text: str) -> tuple[str, ...]:
        evidence = script_evidence(text)
        if evidence["cyrillic"] > evidence["latin"]:
            return ("ru",)
        if evidence["hangul"] > evidence["latin"]:
            return ("ko",)
        if evidence["latin"]:
            return ("en", "ro")
        return self.candidate_languages

    def _decode_target_span(
        self,
        audio: Path,
        span: tuple[int, int],
        original_text: str,
        options: TranscriptionOptions,
        attempts: list[dict],
    ) -> tuple[TranscriptSegment, ...] | None:
        start_ms, end_ms = span
        candidates: list[tuple[TranscriptionBackend, TranscriptionOptions]] = [
            (self.backend, options)
        ]
        if self.retry_backend is not None:
            candidates.append((self.retry_backend, options))
        retry_backend = self.retry_backend or self.backend
        for language in self._target_languages(original_text):
            candidate_options = replace(options, language=language)
            if not any(
                backend.identity == retry_backend.identity and existing == candidate_options
                for backend, existing in candidates
            ):
                candidates.append((retry_backend, candidate_options))

        eligible: list[tuple[tuple[float, ...], int, tuple[TranscriptSegment, ...]]] = []
        with tempfile.TemporaryDirectory(prefix="dubbing-targeted-retry-") as directory:
            retry_audio = Path(directory) / "span.wav"
            marker = TranscriptSegment(start_ms, end_ms, "targeted retry span")
            self._extract_language_span(audio, marker, retry_audio)
            for backend, retry_options in candidates:
                candidate = annotate_transcript_languages(
                    backend.transcribe(retry_audio, retry_options)
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
                        "quality": quality.to_dict(),
                    }
                )
                if not candidate.segments or quality.status in {
                    TranscriptQualityStatus.FAILED,
                    TranscriptQualityStatus.REPROCESS_REQUIRED,
                }:
                    continue
                uncertain_count = sum(item.uncertain for item in candidate.segments)
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
                score = (
                    2.0 if quality.status is TranscriptQualityStatus.PASS else 1.0,
                    -float(uncertain_count),
                    average_log_probability,
                )
                shifted = tuple(item.shifted(start_ms) for item in candidate.segments)
                eligible.append((score, len(attempts) - 1, shifted))
                if quality.status is TranscriptQualityStatus.PASS and not uncertain_count:
                    break
        if not eligible:
            return None
        _, selected_attempt, selected = max(eligible, key=lambda item: item[0])
        attempts[selected_attempt]["selected"] = True
        return selected

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
        silence_centres = detect_silence_centres(audio)
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
            report = evaluate_transcript_quality(
                repaired, expected_duration_ms=duration_ms
            )
            attempts.append(
                {
                    "kind": "targeted-repair-round",
                    "round": round_index + 1,
                    "spans": [
                        {"start_ms": start_ms, "end_ms": end_ms}
                        for start_ms, end_ms in spans
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
    ) -> tuple[TranscriptionResult, dict]:
        attempts: list[dict] = []
        duration_ms = chunk.extract_end_ms - chunk.extract_start_ms
        result = annotate_transcript_languages(self.backend.transcribe(audio, options))
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
    def _extract_language_span(
        source: Path,
        segment: TranscriptSegment,
        destination: Path,
    ) -> None:
        ffmpeg = ffmpeg_executable()
        if ffmpeg is None:
            raise TranscriptionError("ffmpeg is required for turn-level language retry")
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
            eligible_replacements: list[
                tuple[tuple[float, ...], str, tuple[TranscriptSegment, ...]]
            ] = []
            with tempfile.TemporaryDirectory(prefix="dubbing-language-retry-") as directory:
                span = Path(directory) / "span.wav"
                self._extract_language_span(audio, segment, span)
                for language in languages:
                    candidate = annotate_transcript_languages(
                        backend.transcribe(span, replace(base_options, language=language))
                    )
                    quality = evaluate_transcript_quality(
                        candidate,
                        expected_duration_ms=segment.end_ms - segment.start_ms,
                    )
                    attempts.append(
                        {
                            "kind": "turn-language-redecode",
                            "source_start_ms": segment.start_ms,
                            "source_end_ms": segment.end_ms,
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
                            if item.diagnostics
                            and item.diagnostics.avg_log_probability is not None
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
                        eligible_replacements.append(
                            (score, language, candidate_segments)
                        )
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
    ) -> tuple[TranscriptionResult, dict]:
        source = Path(media).resolve()
        if not source.is_file():
            raise TranscriptionError(f"media input not found: {source}")
        options = options or TranscriptionOptions()
        digest = source_sha256(source)
        probe = probe_media(source)
        silence_centres = detect_silence_centres(source)
        chunks = self.planner.plan(probe.duration_ms, silence_centres)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._admit_manifest(self._manifest(source, digest, probe, chunks))
        stream = probe.audio_streams[0]
        completed: list[tuple[AdaptiveChunk, TranscriptionResult]] = []
        for chunk in chunks:
            checkpoint = self.checkpoint_dir / "chunks" / f"{chunk.index:06d}.json"
            receipt_path = self.checkpoint_dir / "receipts" / f"{chunk.index:06d}.json"
            if checkpoint.is_file():
                try:
                    result = transcription_result_from_dict(json.loads(checkpoint.read_text(encoding="utf-8")))
                except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise TranscriptionError(f"malformed adaptive chunk: {checkpoint.name}") from exc
                if any(item.text == _FAILED_SPAN_TEXT for item in result.segments):
                    checkpoint.unlink()
                    receipt_path.unlink(missing_ok=True)
                    result = None
            else:
                result = None
            if result is None:
                with tempfile.TemporaryDirectory(prefix="dubbing-adaptive-") as directory:
                    audio = Path(directory) / f"{chunk.index:06d}.wav"
                    self._extract(source, stream, chunk, audio)
                    result, receipt = self._decode_chunk(audio, chunk, options)
                _atomic_json(checkpoint, result.to_dict())
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
            duration_ms=probe.duration_ms,
            confidence_available=any(item.confidence is not None for item in segments),
            source_sha256=digest,
            provenance={
                "coordinator": "adaptive-long-form-v4",
                "chunk_count": len(chunks),
                "audio_stream": stream.to_dict(),
            },
        )
        quality = evaluate_transcript_quality(result, expected_duration_ms=probe.duration_ms)
        if any(issue.code == "large_unexplained_gaps" for issue in quality.issues):
            quality = evaluate_transcript_quality(
                result,
                expected_duration_ms=probe.duration_ms,
                known_silence_intervals=detect_silence_intervals(source),
            )
        _atomic_json(self.checkpoint_dir / "result.json", result.to_dict())
        _atomic_json(self.checkpoint_dir / "quality-report.json", quality.to_dict())
        return result, quality.to_dict()
