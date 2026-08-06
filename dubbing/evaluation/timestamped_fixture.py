from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dubbing.evaluation.metrics import character_tokens, error_rate, word_tokens
from dubbing.transcription.models import transcription_result_from_dict
from dubbing.transcription.quality import evaluate_transcript_quality


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens_in_window(result, start_ms: int, end_ms: int) -> list[str]:
    tokens = []
    for segment in result.segments:
        for word in segment.words:
            midpoint = word.start_ms + (word.end_ms - word.start_ms) // 2
            if start_ms <= midpoint < end_ms:
                tokens.extend(word_tokens(word.text))
    return tokens


def evaluate_timestamped_fixture(
    fixture_path: str | Path, candidate_path: str | Path, output_path: str | Path
) -> dict:
    fixture_file = Path(fixture_path).resolve()
    candidate_file = Path(candidate_path).resolve()
    fixture = json.loads(fixture_file.read_text(encoding="utf-8"))
    result = transcription_result_from_dict(
        json.loads(candidate_file.read_text(encoding="utf-8"))
    )
    if result.source_sha256 != fixture["source"]["sha256"]:
        raise ValueError("candidate source SHA-256 does not match fixture")

    quality = evaluate_transcript_quality(
        result, expected_duration_ms=int(fixture["source"]["duration_ms"])
    ).to_dict()
    rows = []
    language_buckets: dict[str, dict[str, list[str]]] = {}
    missing_word_timestamp_segments = [
        {"start_ms": segment.start_ms, "end_ms": segment.end_ms}
        for segment in result.segments
        if not segment.words
    ]
    for span in fixture["spans"]:
        candidate_words = _tokens_in_window(
            result, int(span["start_ms"]), int(span["end_ms"])
        )
        reference_words = word_tokens(span["text"])
        reference_chars = character_tokens(span["text"])
        candidate_text = " ".join(candidate_words)
        candidate_chars = character_tokens(candidate_text)
        bucket = language_buckets.setdefault(
            span["language"], {"reference_words": [], "candidate_words": [], "reference_chars": [], "candidate_chars": []}
        )
        bucket["reference_words"].extend(reference_words)
        bucket["candidate_words"].extend(candidate_words)
        bucket["reference_chars"].extend(reference_chars)
        bucket["candidate_chars"].extend(candidate_chars)
        rows.append(
            {
                "id": span["id"],
                "language": span["language"],
                "start_ms": span["start_ms"],
                "end_ms": span["end_ms"],
                "reference": span["text"],
                "candidate": candidate_text,
                "wer": error_rate(reference_words, candidate_words),
                "cer": error_rate(reference_chars, candidate_chars),
            }
        )

    all_reference_words = [token for row in rows for token in word_tokens(row["reference"])]
    all_candidate_words = [token for row in rows for token in word_tokens(row["candidate"])]
    all_reference_chars = [token for row in rows for token in character_tokens(row["reference"])]
    all_candidate_chars = [token for row in rows for token in character_tokens(row["candidate"])]
    overall_wer = error_rate(all_reference_words, all_candidate_words)
    overall_cer = error_rate(all_reference_chars, all_candidate_chars)
    per_language = {}
    for language, bucket in sorted(language_buckets.items()):
        language_wer = error_rate(bucket["reference_words"], bucket["candidate_words"])
        language_cer = error_rate(bucket["reference_chars"], bucket["candidate_chars"])
        per_language[language] = {
            "wer": language_wer,
            "cer": language_cer,
            "word_accuracy": max(0.0, 1.0 - language_wer["rate"]),
            "target_passed": language_wer["rate"] <= 0.1,
        }
    gate_passed = (
        overall_wer["rate"] <= 0.1
        and all(values["target_passed"] for values in per_language.values())
        and not missing_word_timestamp_segments
        and quality["status"] == "PASS"
        and not fixture["coverage_gaps"]
    )
    report = {
        "schema_version": "dubbing.timestamped-fixture-evaluation.v1",
        "fixture_sha256": _sha256(fixture_file),
        "candidate_sha256": _sha256(candidate_file),
        "source_sha256": result.source_sha256,
        "candidate": {"backend": result.backend, "model": result.model},
        "metrics": {
            "wer": overall_wer,
            "cer": overall_cer,
            "word_accuracy": max(0.0, 1.0 - overall_wer["rate"]),
            "target_passed": overall_wer["rate"] <= 0.1,
        },
        "per_language": per_language,
        "word_timestamp_gate_passed": not missing_word_timestamp_segments,
        "missing_word_timestamp_segments": missing_word_timestamp_segments,
        "quality": quality,
        "fixture_gate_passed": gate_passed,
        "accuracy_certification_eligible": fixture["selection_policy"].get(
            "accuracy_certification_eligible", False
        ),
        "production_scope_certified": False,
        "giga_admission_emitted": False,
        "spans": rows,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact timestamped fixture")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate_timestamped_fixture(args.fixture, args.candidate, args.output)
    print(json.dumps({"metrics": report["metrics"], "per_language": report["per_language"], "fixture_gate_passed": report["fixture_gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
