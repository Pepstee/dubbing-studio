from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from dubbing.evaluation.accuracy_gate import build_accuracy_gate
from dubbing.evaluation.metrics import (
    character_tokens,
    error_rate,
    levenshtein_distance,
    word_tokens,
)
from dubbing.transcription.models import TranscriptionResult, transcription_result_from_dict
from dubbing.transcription.quality import evaluate_transcript_quality
from dubbing.media import ffmpeg_executable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusion_reason(decision: dict) -> str | None:
    if decision.get("status") == "no_speech":
        return None
    if decision.get("status") != "corrected":
        return "not_approved_correction"
    text = str(decision.get("text", "")).strip()
    if not word_tokens(text):
        return "punctuation_only"
    if "??" in text or "�" in text:
        return "explicitly_unclear_text"
    if re.search(r"\w\.{2,}(?:\s|$)", text, flags=re.UNICODE):
        return "truncated_word"
    return None


def build_fixture(package_dir: str | Path, output_path: str | Path) -> dict:
    package = Path(package_dir).resolve()
    manifest_path = package / "manifest.json"
    decisions_path = package / "decisions.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions_document = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions = decisions_document.get("decisions") or decisions_document.get("items") or {}
    included = []
    excluded = []
    language_counts: dict[str, int] = {}
    for item in manifest.get("items", []):
        item_id = item["id"]
        decision = decisions.get(item_id)
        if decision is None:
            excluded.append({"id": item_id, "reason": "missing_decision"})
            continue
        reason = _exclusion_reason(decision)
        if reason:
            excluded.append({"id": item_id, "reason": reason})
            continue
        clip = (package / item["clip_path"]).resolve()
        language = decision.get("language") or item.get("proposed_language") or "unknown"
        text = decision.get("text", "")
        segment_duration_seconds = max(0.001, (item["end_ms"] - item["start_ms"]) / 1000)
        implied_words_per_second = len(word_tokens(text)) / segment_duration_seconds
        language_counts[language] = language_counts.get(language, 0) + 1
        included.append(
            {
                "id": item_id,
                "start_ms": item["clip_start_ms"],
                "end_ms": item["clip_end_ms"],
                "segment_start_ms": item["start_ms"],
                "segment_end_ms": item["end_ms"],
                "text": text,
                "language": language,
                "no_speech": decision.get("status") == "no_speech",
                "clip_path": str(clip),
                "clip_sha256": _sha256(clip),
                "decision_updated_at": decision.get("updated_at"),
                "source_segment_sha256": item["segment_sha256"],
                "reference_text_human_verified": True,
                "timestamp_human_verified": False,
                "implied_words_per_second": round(implied_words_per_second, 6),
            }
        )
    source = manifest["source"]
    fixture = {
        "schema_version": "dubbing.operator-ground-truth.v1",
        "fixture_id": f"{package.name}-operator-corrections-v1",
        "source": source,
        "review_evidence": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "decisions_path": str(decisions_path),
            "decisions_sha256": _sha256(decisions_path),
        },
        "selection_policy": {
            "policy_version": "dubbing.operator-ground-truth-selection.v1",
            "timing_basis": "review_clip_with_bounded_context",
            "included_statuses": ["corrected", "no_speech"],
            "excluded": [
                "pending or unclear decisions",
                "punctuation-only corrections",
                "explicit ??/replacement-character uncertainty",
                "truncated words ending in ellipses",
            ],
            "reference_text_is_human_ground_truth": True,
            "reference_timestamps_are_human_ground_truth": False,
            "accuracy_certification_eligible": False,
            "scope_limitation": (
                "Human corrections cover only operator-reviewed uncertain clips. Their "
                "original ASR segment times were not independently corrected, so this "
                "fixture cannot certify timestamp-bounded or full-recording accuracy."
            ),
        },
        "language_counts": language_counts,
        "coverage_gaps": [
            language
            for language in ("en", "ru", "ro", "ko")
            if language_counts.get(language, 0) == 0
        ],
        "included_count": len(included),
        "excluded_count": len(excluded),
        "spans": included,
        "exclusions": excluded,
        "giga_admission_emitted": False,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fixture


def build_context_fixture(
    fixture_path: str | Path,
    output_path: str | Path,
    clips_dir: str | Path,
    *,
    context_seconds: int = 30,
) -> dict:
    if context_seconds < 10 or context_seconds > 60:
        raise ValueError("context_seconds must be between 10 and 60")
    fixture_file = Path(fixture_path).resolve()
    fixture = json.loads(fixture_file.read_text(encoding="utf-8"))
    source = Path(fixture["source"]["path"]).resolve()
    if _sha256(source) != fixture["source"]["sha256"]:
        raise ValueError("source SHA-256 mismatch")
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build context clips")
    destination_dir = Path(clips_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    context_ms = context_seconds * 1000
    duration_ms = int(fixture.get("source", {}).get("duration_ms", 0) or 0)
    spans = []
    for span in fixture["spans"]:
        centre = span["segment_start_ms"] + (span["segment_end_ms"] - span["segment_start_ms"]) // 2
        start_ms = max(0, centre - context_ms // 2)
        end_ms = centre + context_ms // 2
        if duration_ms:
            end_ms = min(duration_ms, end_ms)
            start_ms = max(0, end_ms - context_ms)
        clip = destination_dir / f"{span['id']}-context-{context_seconds}s.wav"
        temporary = clip.with_name(f".{clip.name}.{os.getpid()}.tmp.wav")
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
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if process.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"context extraction failed for {span['id']}: {process.stderr}")
        os.replace(temporary, clip)
        updated = dict(span)
        updated.update(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "clip_path": str(clip),
                "clip_sha256": _sha256(clip),
            }
        )
        spans.append(updated)
    derived = dict(fixture)
    derived["schema_version"] = "dubbing.operator-ground-truth-context.v1"
    derived["fixture_id"] = f"{fixture['fixture_id']}-context-{context_seconds}s"
    derived["base_fixture_sha256"] = _sha256(fixture_file)
    derived["context_seconds"] = context_seconds
    derived["spans"] = spans
    derived["selection_policy"] = dict(fixture["selection_policy"])
    derived["selection_policy"]["timing_basis"] = (
        "fixed context around reviewed segment; reference alignment remains bounded within clip"
    )
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return derived


def build_conditioned_fixture(
    fixture_path: str | Path,
    output_path: str | Path,
    clips_dir: str | Path,
    *,
    filter_graph: str,
) -> dict:
    if not filter_graph.strip():
        raise ValueError("filter graph must not be empty")
    fixture_file = Path(fixture_path).resolve()
    fixture = json.loads(fixture_file.read_text(encoding="utf-8"))
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to condition clips")
    destination_dir = Path(clips_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    spans = []
    for span in fixture["spans"]:
        source_clip = Path(span["clip_path"]).resolve()
        if not source_clip.is_file():
            raise ValueError(f"source clip does not exist: {source_clip}")
        if _sha256(source_clip) != span["clip_sha256"]:
            raise ValueError(f"source clip SHA-256 mismatch: {source_clip.name}")
        clip = destination_dir / source_clip.name
        temporary = clip.with_name(f".{clip.name}.{os.getpid()}.tmp.wav")
        process = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_clip),
                "-af",
                filter_graph,
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if process.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"conditioning failed for {span['id']}: {process.stderr}")
        os.replace(temporary, clip)
        updated = dict(span)
        updated.update(
            {
                "clip_path": str(clip),
                "clip_sha256": _sha256(clip),
                "unconditioned_clip_sha256": span["clip_sha256"],
            }
        )
        spans.append(updated)
    derived = dict(fixture)
    derived["schema_version"] = "dubbing.operator-ground-truth-conditioned.v1"
    derived["fixture_id"] = f"{fixture['fixture_id']}-conditioned"
    derived["base_fixture_sha256"] = _sha256(fixture_file)
    derived["audio_conditioning"] = {
        "tool": Path(ffmpeg).name,
        "filter_graph": filter_graph,
        "output_codec": "pcm_s16le",
        "sample_rate_hz": 16000,
        "channels": 1,
    }
    derived["spans"] = spans
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return derived


def build_trimmed_fixture(
    fixture_path: str | Path,
    output_path: str | Path,
    clips_dir: str | Path,
    *,
    padding_ms: int,
) -> dict:
    if padding_ms < 0 or padding_ms > 1500:
        raise ValueError("padding_ms must be between 0 and 1500")
    fixture_file = Path(fixture_path).resolve()
    fixture = json.loads(fixture_file.read_text(encoding="utf-8"))
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to trim clips")
    destination_dir = Path(clips_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    spans = []
    for span in fixture["spans"]:
        source_clip = Path(span["clip_path"]).resolve()
        if not source_clip.is_file():
            raise ValueError(f"source clip does not exist: {source_clip}")
        if _sha256(source_clip) != span["clip_sha256"]:
            raise ValueError(f"source clip SHA-256 mismatch: {source_clip.name}")
        start_ms = max(span["start_ms"], span["segment_start_ms"] - padding_ms)
        end_ms = min(span["end_ms"], span["segment_end_ms"] + padding_ms)
        relative_start = (start_ms - span["start_ms"]) / 1000
        duration = (end_ms - start_ms) / 1000
        clip = destination_dir / source_clip.name
        temporary = clip.with_name(f".{clip.name}.{os.getpid()}.tmp.wav")
        process = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{relative_start:.3f}",
                "-i",
                str(source_clip),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if process.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"trimming failed for {span['id']}: {process.stderr}")
        os.replace(temporary, clip)
        updated = dict(span)
        updated.update(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "clip_path": str(clip),
                "clip_sha256": _sha256(clip),
                "untrimmed_clip_sha256": span["clip_sha256"],
            }
        )
        spans.append(updated)
    derived = dict(fixture)
    derived["schema_version"] = "dubbing.operator-ground-truth-trimmed.v1"
    derived["fixture_id"] = f"{fixture['fixture_id']}-padding-{padding_ms}ms"
    derived["base_fixture_sha256"] = _sha256(fixture_file)
    derived["audio_trimming"] = {
        "tool": Path(ffmpeg).name,
        "padding_ms": padding_ms,
        "output_codec": "pcm_s16le",
        "sample_rate_hz": 16000,
        "channels": 1,
    }
    derived["spans"] = spans
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return derived


def _candidate_tokens(result: TranscriptionResult, start_ms: int, end_ms: int) -> list[str]:
    tokens = []
    for segment in result.segments:
        midpoint = segment.start_ms + (segment.end_ms - segment.start_ms) // 2
        selected_words = [
            word.text
            for word in segment.words
            if start_ms <= word.start_ms + (word.end_ms - word.start_ms) // 2 < end_ms
        ]
        if selected_words:
            tokens.extend(word_tokens("".join(selected_words)))
        elif start_ms <= midpoint < end_ms:
            tokens.extend(word_tokens(segment.text))
    return tokens


def _candidate_text(
    result: TranscriptionResult,
    start_ms: int,
    end_ms: int,
    reference_text: str,
    *,
    no_speech: bool,
    segment_start_ms: int,
    segment_end_ms: int,
) -> str:
    if no_speech:
        return " ".join(_candidate_tokens(result, segment_start_ms, segment_end_ms))
    tokens = _candidate_tokens(result, start_ms, end_ms)
    reference = word_tokens(reference_text)
    if not tokens or not reference:
        return " ".join(tokens)
    minimum = max(1, len(reference) // 2)
    maximum = min(len(tokens), len(reference) * 2 + 4)
    best: tuple[tuple[float, int, int], list[str]] | None = None
    for width in range(minimum, maximum + 1):
        for index in range(0, len(tokens) - width + 1):
            candidate = tokens[index : index + width]
            edits = levenshtein_distance(reference, candidate)
            score = (edits / len(reference), abs(width - len(reference)), index)
            if best is None or score < best[0]:
                best = (score, candidate)
    return " ".join(best[1] if best else tokens)


def evaluate_candidate(
    fixture_path: str | Path, candidate_path: str | Path, output_path: str | Path
) -> dict:
    fixture_file = Path(fixture_path).resolve()
    candidate_file = Path(candidate_path).resolve()
    fixture = json.loads(fixture_file.read_text(encoding="utf-8"))
    candidate = transcription_result_from_dict(
        json.loads(candidate_file.read_text(encoding="utf-8"))
    )
    expected_hash = fixture["source"]["sha256"]
    if candidate.source_sha256 != expected_hash:
        raise ValueError("candidate source SHA-256 does not match ground-truth fixture")
    rows = []
    aggregate: dict[str, dict[str, list[str]]] = {}
    for span in fixture["spans"]:
        reference_text = "" if span["no_speech"] else span["text"]
        candidate_text = _candidate_text(
            candidate,
            span["start_ms"],
            span["end_ms"],
            reference_text,
            no_speech=span["no_speech"],
            segment_start_ms=span["segment_start_ms"],
            segment_end_ms=span["segment_end_ms"],
        )
        reference_words = word_tokens(reference_text)
        candidate_words = word_tokens(candidate_text)
        reference_characters = character_tokens(reference_text)
        candidate_characters = character_tokens(candidate_text)
        language = span["language"]
        bucket = aggregate.setdefault(
            language,
            {
                "reference_words": [],
                "candidate_words": [],
                "reference_chars": [],
                "candidate_chars": [],
            },
        )
        bucket["reference_words"].extend(reference_words)
        bucket["candidate_words"].extend(candidate_words)
        bucket["reference_chars"].extend(reference_characters)
        bucket["candidate_chars"].extend(candidate_characters)
        rows.append(
            {
                "id": span["id"],
                "start_ms": span["start_ms"],
                "end_ms": span["end_ms"],
                "language": language,
                "reference": reference_text,
                "candidate": candidate_text,
                "wer": error_rate(reference_words, candidate_words),
                "cer": error_rate(reference_characters, candidate_characters),
            }
        )
    all_reference_words = [item for row in rows for item in word_tokens(row["reference"])]
    all_candidate_words = [item for row in rows for item in word_tokens(row["candidate"])]
    all_reference_chars = [item for row in rows for item in character_tokens(row["reference"])]
    all_candidate_chars = [item for row in rows for item in character_tokens(row["candidate"])]
    overall_wer = error_rate(all_reference_words, all_candidate_words)
    overall_cer = error_rate(all_reference_chars, all_candidate_chars)
    per_language = {}
    for language, values in sorted(aggregate.items()):
        language_wer = error_rate(
            values["reference_words"], values["candidate_words"]
        )
        language_cer = error_rate(
            values["reference_chars"], values["candidate_chars"]
        )
        per_language[language] = {
            "wer": language_wer,
            "cer": language_cer,
            "word_accuracy": max(0.0, 1.0 - language_wer["rate"]),
            "character_accuracy": max(0.0, 1.0 - language_cer["rate"]),
            "word_accuracy_threshold": 0.9,
            "word_accuracy_threshold_passed": language_wer["rate"] <= 0.1,
            "character_accuracy_threshold": 0.9,
            "character_accuracy_threshold_passed": language_cer["rate"] <= 0.1,
        }
    expected_duration_ms = candidate.duration_ms or max(
        (int(span["end_ms"]) for span in fixture["spans"]), default=1
    )
    quality = evaluate_transcript_quality(
        candidate, expected_duration_ms=expected_duration_ms
    ).to_dict()
    accuracy_gate = build_accuracy_gate(
        fixture,
        rows,
        deployable_selector=False,
        quality_gate_passed=quality["status"] == "PASS",
        word_timestamp_gate_passed=False,
    )
    report = {
        "schema_version": "dubbing.operator-ground-truth-evaluation.v2",
        "fixture_sha256": _sha256(fixture_file),
        "candidate_sha256": _sha256(candidate_file),
        "source_sha256": expected_hash,
        "candidate": {"backend": candidate.backend, "model": candidate.model},
        "metrics": {
            "wer": overall_wer,
            "cer": overall_cer,
            "word_accuracy": max(0.0, 1.0 - overall_wer["rate"]),
            "character_accuracy": max(0.0, 1.0 - overall_cer["rate"]),
            "word_accuracy_threshold": 0.9,
            "aggregate_word_accuracy_threshold_passed": overall_wer["rate"] <= 0.1,
            "character_accuracy_threshold": 0.9,
            "aggregate_character_accuracy_threshold_passed": (
                overall_cer["rate"] <= 0.1
            ),
        },
        "per_language": per_language,
        "coverage_gaps": fixture["coverage_gaps"],
        "span_count": len(rows),
        "spans": rows,
        "accuracy_certification_scope": "operator-reviewed uncertain clips only",
        "alignment_policy": "bounded minimum-edit token subsequence within review clip",
        "accuracy_certification_eligible": fixture.get("selection_policy", {}).get(
            "accuracy_certification_eligible", False
        ),
        "quality": quality,
        "accuracy_gate": accuracy_gate,
        "measurement_gate_passed": accuracy_gate["measurement_gate"]["passed"],
        "benchmark_claim_passed": accuracy_gate["benchmark_claim"]["passed"],
        "production_claim_passed": False,
        "full_recording_accuracy_certified": False,
        "giga_admission_emitted": False,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Operator-reviewed ground-truth tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--package", required=True)
    build.add_argument("--output", required=True)
    context = subparsers.add_parser("context")
    context.add_argument("--fixture", required=True)
    context.add_argument("--output", required=True)
    context.add_argument("--clips-dir", required=True)
    context.add_argument("--context-seconds", type=int, default=30)
    condition = subparsers.add_parser("condition")
    condition.add_argument("--fixture", required=True)
    condition.add_argument("--output", required=True)
    condition.add_argument("--clips-dir", required=True)
    condition.add_argument("--filter-graph", required=True)
    trim = subparsers.add_parser("trim")
    trim.add_argument("--fixture", required=True)
    trim.add_argument("--output", required=True)
    trim.add_argument("--clips-dir", required=True)
    trim.add_argument("--padding-ms", type=int, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--fixture", required=True)
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        report = build_fixture(args.package, args.output)
        print(
            json.dumps(
                {
                    "included": report["included_count"],
                    "excluded": report["excluded_count"],
                    "coverage_gaps": report["coverage_gaps"],
                },
                indent=2,
            )
        )
    elif args.command == "context":
        report = build_context_fixture(
            args.fixture,
            args.output,
            args.clips_dir,
            context_seconds=args.context_seconds,
        )
        print(
            json.dumps(
                {"spans": len(report["spans"]), "context_seconds": report["context_seconds"]},
                indent=2,
            )
        )
    elif args.command == "condition":
        report = build_conditioned_fixture(
            args.fixture,
            args.output,
            args.clips_dir,
            filter_graph=args.filter_graph,
        )
        print(
            json.dumps(
                {
                    "spans": len(report["spans"]),
                    "filter_graph": report["audio_conditioning"]["filter_graph"],
                },
                indent=2,
            )
        )
    elif args.command == "trim":
        report = build_trimmed_fixture(
            args.fixture,
            args.output,
            args.clips_dir,
            padding_ms=args.padding_ms,
        )
        print(
            json.dumps(
                {
                    "spans": len(report["spans"]),
                    "padding_ms": report["audio_trimming"]["padding_ms"],
                },
                indent=2,
            )
        )
    else:
        report = evaluate_candidate(args.fixture, args.candidate, args.output)
        print(json.dumps(report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
