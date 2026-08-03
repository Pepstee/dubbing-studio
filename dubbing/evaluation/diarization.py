from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Iterable

from dubbing.diarization.models import SpeakerTurn
from dubbing.evaluation.metrics import error_rate, word_tokens
from dubbing.transcription.models import TranscriptSegment


def _active(turns: Iterable[SpeakerTurn], midpoint: float) -> set[str]:
    return {turn.speaker for turn in turns if turn.start_ms <= midpoint < turn.end_ms}


def _overlap_matrix(
    reference: tuple[SpeakerTurn, ...], hypothesis: tuple[SpeakerTurn, ...]
) -> tuple[dict[tuple[str, str], int], tuple[int, ...]]:
    boundaries = tuple(
        sorted(
            {value for turn in reference + hypothesis for value in (turn.start_ms, turn.end_ms)}
        )
    )
    overlap: dict[tuple[str, str], int] = defaultdict(int)
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = start + (end - start) / 2
        for ref in _active(reference, midpoint):
            for hyp in _active(hypothesis, midpoint):
                overlap[(ref, hyp)] += end - start
    return dict(overlap), boundaries


def _speaker_mapping(
    reference: tuple[SpeakerTurn, ...], hypothesis: tuple[SpeakerTurn, ...]
) -> dict[str, str]:
    refs = sorted({turn.speaker for turn in reference})
    hyps = sorted({turn.speaker for turn in hypothesis})
    overlap, _ = _overlap_matrix(reference, hypothesis)
    if len(refs) <= 8 and len(hyps) <= 8:
        padded_refs = refs + [f"__NONE_{index}" for index in range(max(0, len(hyps) - len(refs)))]
        best_score = -1
        best: dict[str, str] = {}
        for assignment in itertools.permutations(padded_refs, len(hyps)):
            score = sum(overlap.get((ref, hyp), 0) for hyp, ref in zip(hyps, assignment))
            if score > best_score:
                best_score = score
                best = {hyp: ref for hyp, ref in zip(hyps, assignment) if not ref.startswith("__NONE_")}
        return best
    return {
        hyp: max(refs, key=lambda ref: overlap.get((ref, hyp), 0))
        for hyp in hyps
        if refs
    }


def diarization_error_report(
    reference: tuple[SpeakerTurn, ...], hypothesis: tuple[SpeakerTurn, ...]
) -> dict:
    mapping = _speaker_mapping(reference, hypothesis)
    _, boundaries = _overlap_matrix(reference, hypothesis)
    miss = false_alarm = confusion = reference_speech = 0
    speaker_intersection: dict[str, int] = defaultdict(int)
    speaker_union: dict[str, int] = defaultdict(int)
    reference_speakers = {turn.speaker for turn in reference}
    for start, end in zip(boundaries, boundaries[1:]):
        duration = end - start
        midpoint = start + duration / 2
        refs = _active(reference, midpoint)
        mapped_hyps = {mapping.get(speaker, f"UNMAPPED:{speaker}") for speaker in _active(hypothesis, midpoint)}
        reference_speech += duration * len(refs)
        miss += duration * max(0, len(refs) - len(mapped_hyps))
        false_alarm += duration * max(0, len(mapped_hyps) - len(refs))
        confusion += duration * min(len(refs), len(mapped_hyps)) - duration * len(refs & mapped_hyps)
        for speaker in reference_speakers:
            in_ref = speaker in refs
            in_hyp = speaker in mapped_hyps
            if in_ref and in_hyp:
                speaker_intersection[speaker] += duration
            if in_ref or in_hyp:
                speaker_union[speaker] += duration
    der = (miss + false_alarm + confusion) / reference_speech if reference_speech else 0.0
    per_speaker_jer = {
        speaker: 1 - speaker_intersection[speaker] / speaker_union[speaker]
        for speaker in reference_speakers
        if speaker_union[speaker]
    }
    return {
        "schema_version": "dubbing.diarization-evaluation.v1",
        "mapping": mapping,
        "der": der,
        "miss_ms": miss,
        "false_alarm_ms": false_alarm,
        "confusion_ms": confusion,
        "reference_speaker_ms": reference_speech,
        "jer": sum(per_speaker_jer.values()) / len(per_speaker_jer) if per_speaker_jer else 0.0,
        "per_speaker_jer": per_speaker_jer,
    }


def speaker_attributed_wer(
    reference: tuple[TranscriptSegment, ...], hypothesis: tuple[TranscriptSegment, ...]
) -> dict:
    def attributed(segments: tuple[TranscriptSegment, ...]) -> list[str]:
        units = []
        for segment in segments:
            speaker = segment.speaker or "UNKNOWN"
            units.extend(f"{speaker}|{token}" for token in word_tokens(segment.text))
        return units

    return error_rate(attributed(reference), attributed(hypothesis))
