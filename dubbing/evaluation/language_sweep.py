from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

from dubbing.evaluation.metrics import character_tokens, error_rate, word_tokens


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(
    reference: list[str],
    candidate: list[str],
    reference_characters: list[str] | None = None,
    candidate_characters: list[str] | None = None,
) -> dict:
    wer = error_rate(reference, candidate)
    result = {"wer": wer, "word_accuracy": max(0.0, 1.0 - wer["rate"])}
    if reference_characters is not None and candidate_characters is not None:
        cer = error_rate(reference_characters, candidate_characters)
        result.update(
            {"cer": cer, "character_accuracy": max(0.0, 1.0 - cer["rate"])}
        )
    return result


def _aggregate_error_rates(rows: list[dict]) -> dict:
    reference_units = sum(row["reference_units"] for row in rows)
    candidate_units = sum(row["candidate_units"] for row in rows)
    edits = sum(row["edits"] for row in rows)
    return {
        "reference_units": reference_units,
        "candidate_units": candidate_units,
        "edits": edits,
        "rate": edits / reference_units if reference_units else (0.0 if not edits else 1.0),
    }


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


def _quality_diagnostics(selections: list[tuple[dict, dict]], temperatures: list[float]) -> dict:
    blank_ids: list[str] = []
    malformed_timestamp_ids: list[str] = []
    adjacent_duplicate_segment_ids: list[str] = []
    repeated_token_run_ids: list[str] = []
    fallback_ids: list[str] = []
    exhausted_fallback_ids: list[str] = []
    maximum_compression_ratio: float | None = None
    maximum_used_temperature = 0.0
    configured_maximum_temperature = max(temperatures, default=0.0)

    for span, candidate in selections:
        identifier = span["id"]
        tokens = word_tokens(candidate.get("text", ""))
        if not span["no_speech"] and not tokens:
            blank_ids.append(identifier)

        previous_segment_end = -1
        previous_segment_tokens: list[str] | None = None
        malformed = False
        adjacent_duplicate = False
        used_temperatures: list[float] = []
        for segment in candidate.get("segments", []):
            start = segment.get("start_ms")
            end = segment.get("end_ms")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or start < previous_segment_end
            ):
                malformed = True
            if isinstance(end, int):
                previous_segment_end = max(previous_segment_end, end)
            segment_tokens = word_tokens(segment.get("text", ""))
            if segment_tokens and segment_tokens == previous_segment_tokens:
                adjacent_duplicate = True
            previous_segment_tokens = segment_tokens
            compression_ratio = segment.get("compression_ratio")
            if isinstance(compression_ratio, (int, float)):
                maximum_compression_ratio = max(
                    float(compression_ratio), maximum_compression_ratio or 0.0
                )
            temperature = segment.get("temperature")
            if isinstance(temperature, (int, float)):
                used_temperatures.append(float(temperature))

            previous_word_end = start if isinstance(start, int) else -1
            for word in segment.get("words", []):
                word_start = word.get("start_ms")
                word_end = word.get("end_ms")
                if (
                    not isinstance(word_start, int)
                    or not isinstance(word_end, int)
                    or word_start < 0
                    or word_end < word_start
                    or word_start < previous_word_end
                ):
                    malformed = True
                if isinstance(word_end, int):
                    previous_word_end = max(previous_word_end, word_end)

        if malformed:
            malformed_timestamp_ids.append(identifier)
        if adjacent_duplicate:
            adjacent_duplicate_segment_ids.append(identifier)

        longest_run = 1
        current_run = 1
        for index in range(1, len(tokens)):
            if tokens[index] == tokens[index - 1]:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        if longest_run >= 4:
            repeated_token_run_ids.append(identifier)

        candidate_temperatures = candidate.get("decode_temperatures", used_temperatures)
        if candidate_temperatures:
            used_maximum = max(float(value) for value in candidate_temperatures)
            maximum_used_temperature = max(maximum_used_temperature, used_maximum)
            if used_maximum > 0:
                fallback_ids.append(identifier)
            if configured_maximum_temperature > 0 and used_maximum >= configured_maximum_temperature:
                exhausted_fallback_ids.append(identifier)

    fail_closed_issues = {
        "blank_hypotheses": sorted(blank_ids),
        "malformed_timestamps": sorted(malformed_timestamp_ids),
        "adjacent_duplicate_segments": sorted(adjacent_duplicate_segment_ids),
        "repeated_token_runs": sorted(repeated_token_run_ids),
        "fallback_exhausted": sorted(exhausted_fallback_ids),
    }
    return {
        "selected_candidate_count": len(selections),
        "maximum_compression_ratio": maximum_compression_ratio,
        "configured_maximum_temperature": configured_maximum_temperature,
        "maximum_used_temperature": maximum_used_temperature,
        "fallback_used_count": len(fallback_ids),
        "fallback_used_ids": sorted(fallback_ids),
        "issues": fail_closed_issues,
        "quality_gate_passed": not any(fail_closed_issues.values()),
    }


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

    def detected_language_retry(row: dict, reference: list[str], span: dict) -> dict:
        automatic = auto(row, reference, span)
        eligible = [
            candidate
            for candidate in row["candidates"]
            if candidate["forced_language"] == automatic["detected_language"]
        ]
        if not eligible:
            return automatic
        return max(
            eligible,
            key=lambda candidate: (
                candidate["beam_size"],
                candidate["avg_log_probability"]
                if candidate["avg_log_probability"] is not None
                else float("-inf"),
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
        "detected_language_retry": (
            detected_language_retry,
            True,
            "Retry forced under the language detected by the automatic first pass.",
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
        character_error_rows: list[dict] = []
        per_language_character_error_rows: dict[str, list[dict]] = {}
        rows = []
        selections: list[tuple[dict, dict]] = []
        for row in sweep["spans"]:
            span = references[row["id"]]
            reference = word_tokens("" if span["no_speech"] else span["text"])
            selected = selector(row, reference, span)
            selections.append((span, selected))
            candidate = _candidate_tokens(selected, span, timing_policy)
            reference_text = "" if span["no_speech"] else span["text"]
            candidate_text = " ".join(candidate)
            reference_chars = character_tokens(reference_text)
            candidate_chars = character_tokens(candidate_text)
            row_score = _score(reference, candidate, reference_chars, candidate_chars)
            reference_words.extend(reference)
            candidate_words.extend(candidate)
            language_reference, language_candidate = per_language_tokens.setdefault(
                span["language"], ([], [])
            )
            language_reference.extend(reference)
            language_candidate.extend(candidate)
            character_error_rows.append(row_score["cer"])
            per_language_character_error_rows.setdefault(span["language"], []).append(
                row_score["cer"]
            )
            rows.append(
                {
                    "id": row["id"],
                    "language": span["language"],
                    "reference": reference_text,
                    "candidate": candidate_text,
                    "forced_language": selected["forced_language"],
                    "detected_language": selected["detected_language"],
                    "beam_size": selected["beam_size"],
                    **row_score,
                }
            )
        overall = _score(reference_words, candidate_words)
        overall_cer = _aggregate_error_rates(character_error_rows)
        overall.update(
            {
                "cer": overall_cer,
                "character_accuracy": max(0.0, 1.0 - overall_cer["rate"]),
            }
        )
        per_language = {}
        for language, tokens in sorted(per_language_tokens.items()):
            language_score = _score(*tokens)
            language_cer = _aggregate_error_rates(
                per_language_character_error_rows[language]
            )
            per_language[language] = {
                **language_score,
                "cer": language_cer,
                "character_accuracy": max(0.0, 1.0 - language_cer["rate"]),
                "target_word_accuracy": 0.9,
                "target_passed": language_score["wer"]["rate"] <= 0.1,
            }
        methods[method] = {
            "deployable": deployable,
            "description": description,
            "metrics": {
                **overall,
                "target_word_accuracy": 0.9,
                "target_passed": overall["wer"]["rate"] <= 0.1,
                "all_language_targets_passed": all(
                    values["target_passed"] for values in per_language.values()
                ),
            },
            "per_language": per_language,
            "quality_diagnostics": _quality_diagnostics(
                selections, [float(value) for value in sweep.get("temperatures", [0.0])]
            ),
            "spans": rows,
        }

    fixture_gate_passed_methods = [
        method
        for method, values in methods.items()
        if not fixture["coverage_gaps"]
        and sweep.get("word_timestamps", True)
        and values["deployable"]
        and values["metrics"]["target_passed"]
        and values["metrics"]["all_language_targets_passed"]
        and values["quality_diagnostics"]["quality_gate_passed"]
    ]
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
        "accuracy_certification_scope": fixture.get("selection_policy", {}).get(
            "scope_limitation", "fixture scope is not declared"
        ),
        "timing_policy": timing_policy,
        "word_timestamps": sweep.get("word_timestamps", True),
        "word_timestamp_gate_passed": sweep.get("word_timestamps", True),
        "accuracy_certification_eligible": fixture.get("selection_policy", {}).get(
            "accuracy_certification_eligible", False
        ),
        "methods": methods,
        "fixture_gate_passed": bool(fixture_gate_passed_methods),
        "fixture_gate_passed_methods": fixture_gate_passed_methods,
        "promotion_passed": False,
        "promotion_reason": (
            "A clean fixture pass does not certify noisy code-switched long-form production."
            if fixture_gate_passed_methods
            else "No tested deployable selector reached 90% overall and in every covered "
            "language; oracle/reference-informed methods cannot be promoted."
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
