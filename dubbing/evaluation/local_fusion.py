from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptionResult,
    transcription_result_from_dict,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _internal_repetition(segment: TranscriptSegment) -> bool:
    tokens = _tokens(segment.text)
    run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if previous == current else 1
        if run >= 8:
            return True
    for width in range(1, min(6, len(tokens)) + 1):
        for index in range(len(tokens) - 2 * width + 1):
            phrase = tokens[index : index + width]
            cursor = index + width
            repeats = 1
            while tokens[cursor : cursor + width] == phrase:
                repeats += 1
                cursor += width
            if repeats >= 6:
                return True
    return False


def pathological_indices(segments: tuple[TranscriptSegment, ...]) -> set[int]:
    marked = {index for index, segment in enumerate(segments) if _internal_repetition(segment)}
    group_start = 0
    for index in range(1, len(segments) + 1):
        same = (
            index < len(segments)
            and _tokens(segments[index].text) == _tokens(segments[group_start].text)
        )
        if same:
            continue
        if index - group_start >= 4:
            marked.update(range(group_start, index))
        group_start = index
    return marked


def _merge_intervals(
    segments: tuple[TranscriptSegment, ...], indices: set[int], padding_ms: int
) -> tuple[tuple[int, int], ...]:
    intervals = []
    for index in sorted(indices):
        segment = segments[index]
        start = max(0, segment.start_ms - padding_ms)
        end = segment.end_ms + padding_ms
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return tuple(intervals)


def fuse_transcripts(
    primary: TranscriptionResult,
    alternate: TranscriptionResult,
    *,
    padding_ms: int = 2000,
) -> tuple[TranscriptionResult, dict]:
    if primary.source_sha256 != alternate.source_sha256 or not primary.source_sha256:
        raise ValueError("transcripts must bind to the same source SHA-256")
    marked = pathological_indices(primary.segments)
    intervals = _merge_intervals(primary.segments, marked, padding_ms)
    alternate_bad = pathological_indices(alternate.segments)
    retained = [
        segment
        for segment in primary.segments
        if not any(
            segment.start_ms < end_ms and segment.end_ms > start_ms
            for start_ms, end_ms in intervals
        )
    ]
    replacements = []
    for index, segment in enumerate(alternate.segments):
        midpoint = segment.start_ms + (segment.end_ms - segment.start_ms) // 2
        if index in alternate_bad:
            continue
        if any(start_ms <= midpoint < end_ms for start_ms, end_ms in intervals):
            replacements.append(replace(segment, uncertain=True))
    ordered = sorted(retained + replacements, key=lambda item: (item.start_ms, item.end_ms))
    reconciled = []
    removed_duplicates = 0
    for segment in ordered:
        if reconciled and _tokens(reconciled[-1].text) == _tokens(segment.text):
            if segment.start_ms <= reconciled[-1].end_ms + padding_ms:
                removed_duplicates += 1
                continue
        reconciled.append(segment)
    provenance = dict(primary.provenance or {})
    provenance["automatic_local_fusion"] = {
        "policy_version": "dubbing.local-pathology-fusion.v1",
        "pathological_primary_segment_count": len(marked),
        "replacement_segment_count": len(replacements),
        "removed_duplicate_segment_count": removed_duplicates,
        "repair_intervals": [
            {"start_ms": start_ms, "end_ms": end_ms}
            for start_ms, end_ms in intervals
        ],
        "alternate_backend": alternate.backend,
        "alternate_model": alternate.model,
    }
    result = replace(
        primary,
        segments=tuple(reconciled),
        text=" ".join(segment.text for segment in reconciled),
        provenance=provenance,
    )
    return result, provenance["automatic_local_fusion"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse only pathological local ASR spans")
    parser.add_argument("--primary", required=True)
    parser.add_argument("--alternate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--padding-ms", type=int, default=2000)
    args = parser.parse_args()
    primary_path = Path(args.primary).resolve()
    alternate_path = Path(args.alternate).resolve()
    primary = transcription_result_from_dict(json.loads(primary_path.read_text(encoding="utf-8")))
    alternate = transcription_result_from_dict(
        json.loads(alternate_path.read_text(encoding="utf-8"))
    )
    result, receipt = fuse_transcripts(primary, alternate, padding_ms=args.padding_ms)
    receipt["primary_sha256"] = _sha256(primary_path)
    receipt["alternate_sha256"] = _sha256(alternate_path)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["output_sha256"] = _sha256(output)
    receipt_path = output.with_name(f"{output.stem}-fusion-receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
