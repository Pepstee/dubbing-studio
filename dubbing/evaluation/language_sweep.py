from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

from dubbing.evaluation.metrics import error_rate, word_tokens


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(reference: list[str], candidate: list[str]) -> dict:
    wer = error_rate(reference, candidate)
    return {"wer": wer, "word_accuracy": max(0.0, 1.0 - wer["rate"])}


def _candidate_tokens(candidate: dict, span: dict, timing_policy: str) -> list[str]:
    if timing_policy == "full_clip":
        return word_tokens(candidate["text"])
    if timing_policy != "target_midpoint":
        raise ValueError(f"unknown timing policy: {timing_policy}")
    target_start = span["segment_start_ms"] - span["start_ms"]
    target_end = span["segment_end_ms"] - span["start_ms"]
    selected: list[str] = []
    has_word_timestamps = False
    for segment in candidate.get("segments", []):
        for word in segment.get("words", []):
            has_word_timestamps = True
            midpoint = word["start_ms"] + (word["end_ms"] - word["start_ms"]) // 2
            if target_start <= midpoint < target_end:
                selected.extend(word_tokens(word["text"]))
    if has_word_timestamps:
        return selected
    return word_tokens(candidate["text"])


def evaluate_language_sweep(
    fixture_path: str | Path,
    sweep_path: str | Path,
    output_path: str | Path,
    *,
    timing_policy: str = "target_midpoint",
) -> dict:
    fixture_file = Path(fixture_path).resolve()
    sweep_file = Path(sweep_path).resolve()
    fixture = json.loads(fixture_file.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_file.read_text(encoding="utf-8"))
    if sweep["source_sha256"] != fixture["source"]["sha256"]:
        raise ValueError("sweep source SHA-256 does not match fixture")
    if sweep["fixture_sha256"] != _sha256(fixture_file):
        raise ValueError("sweep fixture SHA-256 does not match fixture")

    references = {span["id"]: span for span in fixture["spans"]}
    if {row["id"] for row in sweep["spans"]} != set(references):
        raise ValueError("sweep span identifiers do not exactly match fixture")

    minimum_beam = min(sweep["beam_sizes"])

    def auto(row: dict, _: list[str], __: dict) -> dict:
        return next(
            candidate
            for candidate in row["candidates"]
            if candidate["forced_language"] is None and candidate["beam_size"] == minimum_beam
        )

    def maximum_log_probability(row: dict, _: list[str], __: dict) -> dict:
        return max(
            row["candidates"],
            key=lambda candidate: (
                candidate["avg_log_probability"]
                if candidate["avg_log_probability"] is not None
                else float("-inf")
            ),
        )

    def declared_language(row: dict, _: list[str], span: dict) -> dict:
        language = row["declared_reference_language"]
        eligible = [
            candidate for candidate in row["candidates"] if candidate["forced_language"] == language
        ]
        if not eligible:
            return auto(row, _, span)
        return max(
            eligible,
            key=lambda candidate: (
                candidate["beam_size"],
                candidate["avg_log_probability"]
                if candidate["avg_log_probability"] is not None
                else float("-inf"),
            ),
        )

    def oracle(row: dict, reference: list[str], span: dict) -> dict:
        return min(
            row["candidates"],
            key=lambda candidate: (
                error_rate(reference, _candidate_tokens(candidate, span, timing_policy))["edits"],
                abs(len(reference) - len(_candidate_tokens(candidate, span, timing_policy))),
            ),
        )

    selectors: dict[str, tuple[Callable[[dict, list[str], dict], dict], bool, str]] = {
        "automatic_minimum_beam": (
            auto,
            True,
            "Automatic language detection at the minimum tested beam size.",
        ),
        "maximum_average_log_probability": (
            maximum_log_probability,
            True,
            "Highest decoder average log probability; does not inspect reference text.",
        ),
        "declared_reference_language": (
            declared_language,
            False,
            "Diagnostic upper bound using the human reference language label.",
        ),
        "oracle_minimum_edits": (
            oracle,
            False,
            "Non-deployable capacity ceiling selected using human reference text.",
        ),
    }
    methods = {}
    for method, (selector, deployable, description) in selectors.items():
        reference_words: list[str] = []
        candidate_words: list[str] = []
        per_language_tokens: dict[str, tuple[list[str], list[str]]] = {}
        rows = []
        for row in sweep["spans"]:
            span = references[row["id"]]
            reference = word_tokens("" if span["no_speech"] else span["text"])
            selected = selector(row, reference, span)
            candidate = _candidate_tokens(selected, span, timing_policy)
            reference_words.extend(reference)
            candidate_words.extend(candidate)
            language_reference, language_candidate = per_language_tokens.setdefault(
                span["language"], ([], [])
            )
            language_reference.extend(reference)
            language_candidate.extend(candidate)
            rows.append(
                {
                    "id": row["id"],
                    "language": span["language"],
                    "reference": span["text"] if not span["no_speech"] else "",
                    "candidate": " ".join(candidate),
                    "forced_language": selected["forced_language"],
                    "detected_language": selected["detected_language"],
                    "beam_size": selected["beam_size"],
                    **_score(reference, candidate),
                }
            )
        overall = _score(reference_words, candidate_words)
        methods[method] = {
            "deployable": deployable,
            "description": description,
            "metrics": {
                **overall,
                "target_word_accuracy": 0.9,
                "target_passed": overall["wer"]["rate"] <= 0.1,
            },
            "per_language": {
                language: _score(*tokens)
                for language, tokens in sorted(per_language_tokens.items())
            },
            "spans": rows,
        }

    report = {
        "schema_version": "dubbing.faster-whisper-language-sweep-evaluation.v1",
        "fixture_sha256": _sha256(fixture_file),
        "sweep_sha256": _sha256(sweep_file),
        "source_sha256": fixture["source"]["sha256"],
        "model": sweep["model"],
        "model_bin_sha256": sweep["model_bin_sha256"],
        "compute_type": sweep["compute_type"],
        "runtime_seconds": sweep["runtime_seconds"],
        "coverage_gaps": fixture["coverage_gaps"],
        "accuracy_certification_scope": "operator-reviewed uncertain clips only",
        "timing_policy": timing_policy,
        "accuracy_certification_eligible": fixture.get("selection_policy", {}).get(
            "accuracy_certification_eligible", False
        ),
        "methods": methods,
        "promotion_passed": False,
        "promotion_reason": (
            "No tested deployable selector reached 90% word accuracy; oracle/reference-"
            "informed methods cannot be promoted."
        ),
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
    parser = argparse.ArgumentParser(description="Evaluate a faster-whisper language sweep")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--timing-policy", choices=("target_midpoint", "full_clip"), default="target_midpoint"
    )
    args = parser.parse_args()
    report = evaluate_language_sweep(
        args.fixture, args.sweep, args.output, timing_policy=args.timing_policy
    )
    summary = {method: values["metrics"] for method, values in report["methods"].items()}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
