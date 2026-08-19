from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dubbing.media import ffmpeg_executable
from dubbing.transcription.models import TranscriptionError, TranscriptionResult
from dubbing.transcription.speech_regions import SpeechRegionPlan


@dataclass(frozen=True)
class RetainedInterval:
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("retained interval must satisfy 0 <= start_ms < end_ms")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}


@dataclass(frozen=True)
class CompactionSlice:
    original_start_ms: int
    original_end_ms: int
    compact_start_ms: int
    compact_end_ms: int

    def __post_init__(self) -> None:
        if (
            self.original_start_ms < 0
            or self.original_end_ms <= self.original_start_ms
            or self.compact_start_ms < 0
            or self.compact_end_ms <= self.compact_start_ms
            or self.original_end_ms - self.original_start_ms
            != self.compact_end_ms - self.compact_start_ms
        ):
            raise ValueError("compaction slice must preserve a positive duration")

    def to_dict(self) -> dict:
        return {
            "original_start_ms": self.original_start_ms,
            "original_end_ms": self.original_end_ms,
            "compact_start_ms": self.compact_start_ms,
            "compact_end_ms": self.compact_end_ms,
        }


@dataclass(frozen=True)
class CompactionPacket:
    index: int
    slices: tuple[CompactionSlice, ...]
    compact_duration_ms: int
    separator_ms: int

    def __post_init__(self) -> None:
        if self.index < 0 or not self.slices or self.compact_duration_ms <= 0:
            raise ValueError("compaction packet must contain positive-duration slices")
        if self.separator_ms < 0:
            raise ValueError("separator_ms cannot be negative")
        expected = sum(
            item.compact_end_ms - item.compact_start_ms for item in self.slices
        ) + self.separator_ms * (len(self.slices) - 1)
        if expected != self.compact_duration_ms:
            raise ValueError("compaction packet duration does not match its map")

    @property
    def original_retained_ms(self) -> int:
        return sum(item.original_end_ms - item.original_start_ms for item in self.slices)

    @property
    def mapping_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "compact_duration_ms": self.compact_duration_ms,
            "separator_ms": self.separator_ms,
            "original_retained_ms": self.original_retained_ms,
            "slices": [item.to_dict() for item in self.slices],
        }

    def restore_interval(self, start_ms: int, end_ms: int) -> tuple[int, int] | None:
        if start_ms < 0 or end_ms <= start_ms or end_ms > self.compact_duration_ms:
            raise TranscriptionError("provider timestamp is outside compacted audio")
        midpoint = start_ms + (end_ms - start_ms) // 2
        target = next(
            (
                item
                for item in self.slices
                if item.compact_start_ms <= midpoint < item.compact_end_ms
            ),
            None,
        )
        if target is None:
            return None
        if start_ms < target.compact_start_ms or end_ms > target.compact_end_ms:
            return None
        original_start = target.original_start_ms + start_ms - target.compact_start_ms
        original_end = target.original_start_ms + end_ms - target.compact_start_ms
        return original_start, original_end


@dataclass(frozen=True)
class SpeechCompactionPlan:
    detector_identity: str
    source_duration_ms: int
    padding_ms: int
    merge_gap_ms: int
    separator_ms: int
    maximum_slices_per_packet: int
    preserve_diarization_context: bool
    retained_intervals: tuple[RetainedInterval, ...]
    packets: tuple[CompactionPacket, ...]
    fallback_reason: str | None = None

    @property
    def retained_ms(self) -> int:
        return sum(item.duration_ms for item in self.retained_intervals)

    @property
    def uploaded_ms(self) -> int:
        return sum(item.compact_duration_ms for item in self.packets)

    @property
    def removed_ms(self) -> int:
        return self.source_duration_ms - self.retained_ms

    @property
    def mapping_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema_version": "dubbing.speech-compaction-plan.v1",
            "detector_identity": self.detector_identity,
            "source_duration_ms": self.source_duration_ms,
            "padding_ms": self.padding_ms,
            "merge_gap_ms": self.merge_gap_ms,
            "separator_ms": self.separator_ms,
            "maximum_slices_per_packet": self.maximum_slices_per_packet,
            "preserve_diarization_context": self.preserve_diarization_context,
            "retained_ms": self.retained_ms,
            "removed_ms": self.removed_ms,
            "uploaded_ms": self.uploaded_ms,
            "estimated_reduction_fraction": (
                1 - self.uploaded_ms / self.source_duration_ms
                if self.source_duration_ms
                else 0
            ),
            "fallback_reason": self.fallback_reason,
            "retained_intervals": [item.to_dict() for item in self.retained_intervals],
            "packets": [item.to_dict() for item in self.packets],
            "known_limit": (
                "Sensitive VAD plus local transcript evidence reduces but cannot eliminate "
                "the risk that both systems miss quiet speech."
            ),
        }


def _merge_intervals(
    intervals: list[tuple[int, int]], *, duration_ms: int, padding_ms: int, merge_gap_ms: int
) -> tuple[RetainedInterval, ...]:
    padded = sorted(
        (max(0, start - padding_ms), min(duration_ms, end + padding_ms))
        for start, end in intervals
        if end > start
    )
    merged: list[list[int]] = []
    for start, end in padded:
        if merged and start - merged[-1][1] <= merge_gap_ms:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple(RetainedInterval(start, end) for start, end in merged if end > start)


def _split_long_intervals(
    intervals: tuple[RetainedInterval, ...], maximum_packet_ms: int
) -> tuple[RetainedInterval, ...]:
    split = []
    for interval in intervals:
        cursor = interval.start_ms
        while cursor < interval.end_ms:
            end = min(interval.end_ms, cursor + maximum_packet_ms)
            split.append(RetainedInterval(cursor, end))
            cursor = end
    return tuple(split)


def build_speech_compaction_plan(
    region_plan: SpeechRegionPlan,
    local_result: TranscriptionResult,
    *,
    padding_ms: int = 1_500,
    merge_gap_ms: int = 2_000,
    separator_ms: int = 500,
    maximum_packet_ms: int = 3_600_000,
    maximum_slices_per_packet: int = 64,
    preserve_diarization_context: bool = True,
) -> SpeechCompactionPlan:
    if padding_ms < 0 or merge_gap_ms < 0 or separator_ms < 0:
        raise ValueError("compaction timing values cannot be negative")
    if maximum_packet_ms < 60_000:
        raise ValueError("maximum compact packet must be at least 60 seconds")
    if not 1 <= maximum_slices_per_packet <= 200:
        raise ValueError("maximum_slices_per_packet must be between 1 and 200")
    if local_result.duration_ms is None or local_result.duration_ms <= 0:
        raise TranscriptionError("local transcript must preserve source duration")
    if abs(region_plan.duration_ms - local_result.duration_ms) > 1_000:
        raise TranscriptionError("VAD duration does not match the local transcript")

    candidates = [
        (item.start_ms, item.end_ms)
        for item in (*region_plan.strict_regions, *region_plan.sensitive_regions)
    ]
    candidates.extend((item.start_ms, item.end_ms) for item in local_result.segments)
    fallback_reason = None
    intervals = _merge_intervals(
        candidates,
        duration_ms=local_result.duration_ms,
        padding_ms=padding_ms,
        merge_gap_ms=merge_gap_ms,
    )
    if not intervals:
        intervals = (RetainedInterval(0, local_result.duration_ms),)
        fallback_reason = "NO_SPEECH_EVIDENCE_FULL_AUDIO_RETAINED"
    intervals = _split_long_intervals(intervals, maximum_packet_ms)

    packets = []
    packet_intervals: list[RetainedInterval] = []
    packet_duration = 0

    def finish_packet() -> None:
        nonlocal packet_duration, packet_intervals
        if not packet_intervals:
            return
        compact_cursor = 0
        slices = []
        for index, interval in enumerate(packet_intervals):
            if index:
                compact_cursor += separator_ms
            compact_end = compact_cursor + interval.duration_ms
            slices.append(
                CompactionSlice(
                    interval.start_ms,
                    interval.end_ms,
                    compact_cursor,
                    compact_end,
                )
            )
            compact_cursor = compact_end
        packets.append(
            CompactionPacket(len(packets), tuple(slices), compact_cursor, separator_ms)
        )
        packet_intervals = []
        packet_duration = 0

    for interval in intervals:
        addition = interval.duration_ms + (separator_ms if packet_intervals else 0)
        if packet_intervals and (
            packet_duration + addition > maximum_packet_ms
            or len(packet_intervals) >= maximum_slices_per_packet
            or preserve_diarization_context
        ):
            finish_packet()
            addition = interval.duration_ms
        packet_intervals.append(interval)
        packet_duration += addition
    finish_packet()

    return SpeechCompactionPlan(
        detector_identity=region_plan.detector_identity,
        source_duration_ms=local_result.duration_ms,
        padding_ms=padding_ms,
        merge_gap_ms=merge_gap_ms,
        separator_ms=separator_ms,
        maximum_slices_per_packet=maximum_slices_per_packet,
        preserve_diarization_context=preserve_diarization_context,
        retained_intervals=intervals,
        packets=tuple(packets),
        fallback_reason=fallback_reason,
    )


def extract_compacted_flac(
    source: str | Path, packet: CompactionPacket, destination: str | Path
) -> None:
    """Extract mapped source intervals and insert deterministic separator silence."""

    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise TranscriptionError("ffmpeg is required for cloud speech compaction")
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    for item in packet.slices:
        command.extend(
            [
                "-ss",
                f"{item.original_start_ms / 1000:.3f}",
                "-t",
                f"{(item.original_end_ms - item.original_start_ms) / 1000:.3f}",
                "-i",
                str(source_path),
            ]
        )
    filters = []
    labels = []
    for index, _ in enumerate(packet.slices):
        filters.append(
            f"[{index}:a]aresample=16000,aformat=sample_fmts=s16:"
            f"channel_layouts=mono,asetpts=PTS-STARTPTS[c{index}]"
        )
        labels.append(f"[c{index}]")
        if index < len(packet.slices) - 1 and packet.separator_ms:
            filters.append(
                f"anullsrc=r=16000:cl=mono,atrim=duration={packet.separator_ms / 1000:.3f},"
                f"asetpts=PTS-STARTPTS[s{index}]"
            )
            labels.append(f"[s{index}]")
    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[out]")
    else:
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:a",
            "flac",
            "-y",
            str(destination_path),
        ]
    )
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(180, packet.original_retained_ms // 1000 + 120),
    )
    if process.returncode:
        raise TranscriptionError(
            f"cloud speech compaction failed: {process.stderr.strip()}"
        )
    if not destination_path.is_file() or destination_path.stat().st_size == 0:
        raise TranscriptionError("cloud speech compaction produced no audio")
