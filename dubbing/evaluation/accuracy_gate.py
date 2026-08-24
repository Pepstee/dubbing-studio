from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class AccuracyGateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class AccuracyGatePolicy:
    target_accuracy: float = 0.90
    required_languages: tuple[str, ...] = ("en", "ru", "ro", "ko")
    minimum_utterances_per_language: int = 25
    minimum_reference_words_per_language: int = 100
    minimum_reference_characters_per_language: int = 500


DEFAULT_ACCURACY_GATE_POLICY = AccuracyGatePolicy()


def _reason(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _aggregate(rows: Iterable[dict], metric: str) -> dict[str, int | float]:
    values = [row[metric] for row in rows]
    reference_units = sum(int(value["reference_units"]) for value in values)
    candidate_units = sum(int(value["candidate_units"]) for value in values)
    edits = sum(int(value["edits"]) for value in values)
    rate = edits / reference_units if reference_units else (0.0 if not edits else 1.0)
    return {
        "reference_units": reference_units,
        "candidate_units": candidate_units,
        "edits": edits,
        "rate": rate,
    }


def _one_sided_wilson_upper_bound(edits: int, reference_units: int) -> float | None:
    """Conservative 95% error bound used as a finite-sample promotion guard.

    WER is not binomial because insertions can make edits exceed reference words. The gate
    therefore reports this as a Wilson *guard*, not a confidence interval for WER. Any edit
    count above the reference count is assigned the maximum bound instead of being clipped
    into a favourable result.
    """

    if reference_units <= 0:
        return None
    if edits > reference_units:
        return 1.0
    z = 1.6448536269514722
    proportion = edits / reference_units
    z_squared = z * z
    denominator = 1.0 + z_squared / reference_units
    centre = proportion + z_squared / (2.0 * reference_units)
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * reference_units))
        / reference_units
    )
    return min(1.0, (centre + radius) / denominator)


def _metric_summary(rows: list[dict], metric: str, target_accuracy: float) -> dict:
    aggregate = _aggregate(rows, metric)
    upper_bound = _one_sided_wilson_upper_bound(
        int(aggregate["edits"]), int(aggregate["reference_units"])
    )
    observed_accuracy = max(0.0, 1.0 - float(aggregate["rate"]))
    lower_guard_accuracy = (
        max(0.0, 1.0 - upper_bound) if upper_bound is not None else None
    )
    return {
        metric: aggregate,
        "observed_accuracy": observed_accuracy,
        "observed_threshold_passed": observed_accuracy >= target_accuracy,
        "one_sided_wilson_error_guard_95": upper_bound,
        "guarded_lower_accuracy_95": lower_guard_accuracy,
        "guarded_threshold_passed": (
            lower_guard_accuracy is not None
            and lower_guard_accuracy >= target_accuracy
        ),
    }


def infer_evidence_profile(fixture: dict) -> dict:
    schema = str(fixture.get("schema_version", ""))
    policy = fixture.get("selection_policy", {})
    source = fixture.get("source", {})
    explicit = fixture.get("evidence_profile", {})

    if explicit.get("scope"):
        scope = explicit["scope"]
    elif schema.startswith("dubbing.fleurs-ground-truth"):
        scope = "clean_read_speech"
    elif schema.startswith("dubbing.derived-noisy-codeswitch"):
        scope = "synthetic_noisy_code_switch"
    elif schema.startswith("dubbing.operator-ground-truth"):
        scope = "natural_reviewed_spans"
    else:
        scope = "unspecified"

    split = source.get("split")
    held_out = explicit.get("held_out")
    if held_out is None:
        held_out = split == "test"

    natural_audio = explicit.get("natural_audio")
    if natural_audio is None:
        natural_audio = scope in {"natural_reviewed_spans", "natural_long_form"}

    reference_timestamps = bool(
        policy.get("reference_timestamps_are_human_ground_truth", False)
    )
    if scope == "clean_read_speech":
        reference_timestamps = bool(
            policy.get("reference_timestamps_are_human_ground_truth", True)
        )

    return {
        "scope": scope,
        "held_out": bool(held_out),
        "natural_audio": bool(natural_audio),
        "long_form": bool(explicit.get("long_form", False)),
        "noisy": bool(
            explicit.get("noisy", scope == "synthetic_noisy_code_switch")
        ),
        "code_switched": bool(
            explicit.get(
                "code_switched", scope == "synthetic_noisy_code_switch"
            )
        ),
        "overlapping_speech": bool(explicit.get("overlapping_speech", False)),
        "speaker_attributed": bool(explicit.get("speaker_attributed", False)),
        "reference_text_human_ground_truth": bool(
            policy.get("reference_text_is_human_ground_truth", False)
        ),
        "reference_timestamps_human_ground_truth": reference_timestamps,
        "complete_recordings": bool(explicit.get("complete_recordings", False)),
        "recording_count": int(explicit.get("recording_count", 0)),
        "duration_ms": int(explicit.get("duration_ms", 0)),
        "speaker_count": int(explicit.get("speaker_count", 0)),
        "speech_coverage_measured": bool(
            explicit.get("speech_coverage_measured", False)
        ),
        "uncertainty_rate": explicit.get("uncertainty_rate"),
        "speaker_attributed_wer_measured": bool(
            explicit.get("speaker_attributed_wer_measured", False)
        ),
        "der_jer_measured": bool(explicit.get("der_jer_measured", False)),
        "accuracy_certification_eligible": bool(
            policy.get("accuracy_certification_eligible", False)
        ),
        "scope_limitation": policy.get(
            "scope_limitation", "fixture scope is not declared"
        ),
    }


def build_accuracy_gate(
    fixture: dict,
    rows: list[dict],
    *,
    deployable_selector: bool,
    quality_gate_passed: bool,
    word_timestamp_gate_passed: bool,
    policy: AccuracyGatePolicy = DEFAULT_ACCURACY_GATE_POLICY,
) -> dict:
    profile = infer_evidence_profile(fixture)
    by_language: dict[str, list[dict]] = {}
    for row in rows:
        by_language.setdefault(str(row["language"]), []).append(row)

    overall_word = _metric_summary(rows, "wer", policy.target_accuracy)
    overall_character = _metric_summary(rows, "cer", policy.target_accuracy)
    per_language = {}
    for language in sorted(set(policy.required_languages) | set(by_language)):
        language_rows = by_language.get(language, [])
        per_language[language] = {
            "utterance_count": sum(
                int(row["wer"]["reference_units"]) > 0 for row in language_rows
            ),
            "word": _metric_summary(
                language_rows, "wer", policy.target_accuracy
            ),
            "character": _metric_summary(
                language_rows, "cer", policy.target_accuracy
            ),
        }

    performance_failures: list[dict] = []
    evidence_gaps: list[dict] = []
    if not overall_word["observed_threshold_passed"]:
        performance_failures.append(
            _reason("AGGREGATE_WORD_ACCURACY_BELOW_TARGET", "Aggregate WER exceeds 10%.")
        )
    if not overall_character["observed_threshold_passed"]:
        performance_failures.append(
            _reason(
                "AGGREGATE_CHARACTER_ACCURACY_BELOW_TARGET",
                "Aggregate CER exceeds 10%.",
            )
        )
    if not overall_word["guarded_threshold_passed"]:
        evidence_gaps.append(
            _reason(
                "AGGREGATE_WORD_ACCURACY_MARGIN_NOT_ESTABLISHED",
                "Aggregate evidence does not establish a guarded 90% lower word-accuracy bound.",
            )
        )
    if not overall_character["guarded_threshold_passed"]:
        evidence_gaps.append(
            _reason(
                "AGGREGATE_CHARACTER_ACCURACY_MARGIN_NOT_ESTABLISHED",
                "Aggregate evidence does not establish a guarded 90% lower character-accuracy bound.",
            )
        )
    if not quality_gate_passed:
        performance_failures.append(
            _reason("SEMANTIC_QUALITY_GATE_FAILED", "Selected output failed semantic quality checks.")
        )
    for language in policy.required_languages:
        evidence = per_language[language]
        word = evidence["word"]
        character = evidence["character"]
        if evidence["utterance_count"] < policy.minimum_utterances_per_language:
            evidence_gaps.append(
                _reason(
                    f"INSUFFICIENT_{language.upper()}_UTTERANCES",
                    f"{language} has {evidence['utterance_count']} reference utterances; "
                    f"{policy.minimum_utterances_per_language} required.",
                )
            )
        if word["wer"]["reference_units"] < policy.minimum_reference_words_per_language:
            evidence_gaps.append(
                _reason(
                    f"INSUFFICIENT_{language.upper()}_REFERENCE_WORDS",
                    f"{language} has {word['wer']['reference_units']} reference words; "
                    f"{policy.minimum_reference_words_per_language} required.",
                )
            )
        if (
            character["cer"]["reference_units"]
            < policy.minimum_reference_characters_per_language
        ):
            evidence_gaps.append(
                _reason(
                    f"INSUFFICIENT_{language.upper()}_REFERENCE_CHARACTERS",
                    f"{language} has {character['cer']['reference_units']} reference characters; "
                    f"{policy.minimum_reference_characters_per_language} required.",
                )
            )
        if word["wer"]["reference_units"] and not word["observed_threshold_passed"]:
            performance_failures.append(
                _reason(
                    f"{language.upper()}_WORD_ACCURACY_BELOW_TARGET",
                    f"{language} observed word accuracy is below 90%.",
                )
            )
        if character["cer"]["reference_units"] and not character[
            "observed_threshold_passed"
        ]:
            performance_failures.append(
                _reason(
                    f"{language.upper()}_CHARACTER_ACCURACY_BELOW_TARGET",
                    f"{language} observed character accuracy is below 90%.",
                )
            )
        if word["wer"]["reference_units"] and not word["guarded_threshold_passed"]:
            evidence_gaps.append(
                _reason(
                    f"{language.upper()}_WORD_ACCURACY_MARGIN_NOT_ESTABLISHED",
                    f"{language} does not establish a guarded 90% lower word-accuracy bound.",
                )
            )
        if character["cer"]["reference_units"] and not character[
            "guarded_threshold_passed"
        ]:
            evidence_gaps.append(
                _reason(
                    f"{language.upper()}_CHARACTER_ACCURACY_MARGIN_NOT_ESTABLISHED",
                    f"{language} does not establish a guarded 90% lower character-accuracy bound.",
                )
            )

    if fixture.get("coverage_gaps"):
        evidence_gaps.append(
            _reason(
                "DECLARED_COVERAGE_GAPS",
                f"Fixture declares coverage gaps: {fixture['coverage_gaps']!r}.",
            )
        )
    missing_languages = [
        language
        for language in policy.required_languages
        if not per_language[language]["word"]["wer"]["reference_units"]
    ]
    if missing_languages:
        evidence_gaps.append(
            _reason(
                "REQUIRED_LANGUAGES_MISSING",
                f"No scored reference words for: {', '.join(missing_languages)}.",
            )
        )
    if not deployable_selector:
        evidence_gaps.append(
            _reason(
                "REFERENCE_INFORMED_SELECTOR",
                "Candidate selection uses reference evidence and is not deployable.",
            )
        )
    if not word_timestamp_gate_passed:
        evidence_gaps.append(
            _reason(
                "WORD_TIMESTAMPS_MISSING",
                "Required word-level timing evidence is incomplete.",
            )
        )

    measurement_status = (
        AccuracyGateStatus.FAIL
        if performance_failures
        else AccuracyGateStatus.INSUFFICIENT_EVIDENCE
        if evidence_gaps
        else AccuracyGateStatus.PASS
    )
    benchmark_gaps = list(evidence_gaps)
    if not profile["accuracy_certification_eligible"]:
        benchmark_gaps.append(
            _reason(
                "FIXTURE_NOT_CERTIFICATION_ELIGIBLE",
                "Fixture policy does not permit an accuracy certificate.",
            )
        )
    if not profile["reference_text_human_ground_truth"]:
        benchmark_gaps.append(
            _reason("TEXT_NOT_HUMAN_GROUND_TRUTH", "Reference text is not human ground truth.")
        )
    if not profile["reference_timestamps_human_ground_truth"]:
        benchmark_gaps.append(
            _reason(
                "TIMESTAMPS_NOT_HUMAN_GROUND_TRUTH",
                "Reference timestamps are not human-verified ground truth.",
            )
        )
    if not profile["held_out"]:
        benchmark_gaps.append(
            _reason("NOT_HELD_OUT", "Fixture is not an untouched holdout split.")
        )
    benchmark_status = (
        AccuracyGateStatus.FAIL
        if performance_failures
        else AccuracyGateStatus.INSUFFICIENT_EVIDENCE
        if benchmark_gaps
        else AccuracyGateStatus.PASS
    )

    production_reasons = [
        _reason(
            "MULTI_SUITE_PORTFOLIO_REQUIRED",
            "A single fixture cannot certify production transcription accuracy.",
        ),
        _reason(
            "NATURAL_LONG_FORM_EVIDENCE_REQUIRED",
            "Production needs complete human-ground-truthed natural long recordings.",
        ),
        _reason(
            "NOISY_CODE_SWITCH_EVIDENCE_REQUIRED",
            "Production needs held-out noisy code-switched evidence.",
        ),
        _reason(
            "SPEAKER_ATTRIBUTED_EVIDENCE_REQUIRED",
            "Production needs speaker-attributed WER plus DER/JER evidence.",
        ),
        _reason(
            "UNCERTAINTY_AND_COVERAGE_EVIDENCE_REQUIRED",
            "Production needs measured speech coverage and unresolved-span burden.",
        ),
    ]
    production_status = (
        AccuracyGateStatus.FAIL
        if performance_failures
        else AccuracyGateStatus.INSUFFICIENT_EVIDENCE
    )
    return {
        "schema_version": "dubbing.accuracy-claim-gate.v1",
        "policy": {
            "target_accuracy": policy.target_accuracy,
            "required_languages": list(policy.required_languages),
            "minimum_utterances_per_language": policy.minimum_utterances_per_language,
            "minimum_reference_words_per_language": (
                policy.minimum_reference_words_per_language
            ),
            "minimum_reference_characters_per_language": (
                policy.minimum_reference_characters_per_language
            ),
            "finite_sample_guard": (
                "one-sided Wilson error guard at 95%; WER insertions above reference "
                "units force the maximum bound"
            ),
        },
        "evidence_profile": profile,
        "observations": {
            "aggregate_word": overall_word,
            "aggregate_character": overall_character,
            "per_language": per_language,
        },
        "measurement_gate": {
            "status": measurement_status,
            "passed": measurement_status is AccuracyGateStatus.PASS,
            "performance_failures": performance_failures,
            "evidence_gaps": evidence_gaps,
        },
        "benchmark_claim": {
            "status": benchmark_status,
            "passed": benchmark_status is AccuracyGateStatus.PASS,
            "performance_failures": performance_failures,
            "evidence_gaps": benchmark_gaps,
            "claim_scope": profile["scope"],
        },
        "production_claim": {
            "status": production_status,
            "passed": False,
            "reasons": [*performance_failures, *production_reasons],
        },
        "giga_admission_emitted": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report_gates(report: dict) -> list[dict]:
    if "accuracy_gate" in report:
        return [report["accuracy_gate"]]
    return [
        values["accuracy_gate"]
        for values in report.get("methods", {}).values()
        if values.get("deployable", False) and "accuracy_gate" in values
    ]


def build_accuracy_portfolio(reports: list[dict]) -> dict:
    grouped_gates = [
        {
            "source_sha256": report.get("source_sha256"),
            "gates": _report_gates(report),
        }
        for report in reports
    ]
    gates = [
        {"source_sha256": group["source_sha256"], "gate": gate}
        for group in grouped_gates
        for gate in group["gates"]
    ]
    if not gates:
        raise ValueError("accuracy portfolio requires at least one v2 accuracy gate")

    requirements = {
        "clean_read_speech_holdout": any(
            item["gate"]["evidence_profile"]["scope"] == "clean_read_speech"
            and item["gate"]["benchmark_claim"]["passed"]
            for item in gates
        ),
        "noisy_code_switch_stress": any(
            item["gate"]["evidence_profile"]["scope"]
            == "synthetic_noisy_code_switch"
            and item["gate"]["measurement_gate"]["passed"]
            for item in gates
        ),
        "natural_long_form_holdout": any(
            item["gate"]["evidence_profile"]["scope"] == "natural_long_form"
            and item["gate"]["evidence_profile"]["complete_recordings"]
            and item["gate"]["benchmark_claim"]["passed"]
            for item in gates
        ),
        "speaker_attributed_natural_audio": any(
            item["gate"]["evidence_profile"]["natural_audio"]
            and item["gate"]["evidence_profile"]["speaker_attributed"]
            and item["gate"]["evidence_profile"]["speaker_attributed_wer_measured"]
            and item["gate"]["evidence_profile"]["der_jer_measured"]
            and item["gate"]["benchmark_claim"]["passed"]
            for item in gates
        ),
        "overlapping_speech_evaluated": any(
            item["gate"]["evidence_profile"]["natural_audio"]
            and item["gate"]["evidence_profile"]["overlapping_speech"]
            and item["gate"]["evidence_profile"]["der_jer_measured"]
            and item["gate"]["benchmark_claim"]["passed"]
            for item in gates
        ),
    }
    natural_by_source: dict[str, dict] = {}
    for item in gates:
        profile = item["gate"]["evidence_profile"]
        if profile["natural_audio"]:
            key = str(item["source_sha256"] or id(profile))
            natural_by_source.setdefault(key, profile)
    natural_profiles = list(natural_by_source.values())
    natural_recordings = sum(profile["recording_count"] for profile in natural_profiles)
    natural_duration_ms = sum(profile["duration_ms"] for profile in natural_profiles)
    natural_speakers = max(
        (profile["speaker_count"] for profile in natural_profiles), default=0
    )
    requirements.update(
        {
            "natural_recording_diversity": natural_recordings >= 3,
            "natural_duration_at_least_three_hours": natural_duration_ms
            >= 3 * 60 * 60 * 1000,
            "natural_speaker_diversity": natural_speakers >= 2,
            "speech_coverage_measured": bool(natural_profiles)
            and all(
                profile["speech_coverage_measured"] for profile in natural_profiles
            ),
            "uncertainty_burden_at_most_five_percent": bool(natural_profiles)
            and all(
                isinstance(profile["uncertainty_rate"], (int, float))
                and 0 <= profile["uncertainty_rate"] <= 0.05
                for profile in natural_profiles
            ),
        }
    )
    failed_measurements = [
        group
        for group in grouped_gates
        if group["gates"]
        and all(
            gate["measurement_gate"]["status"] == AccuracyGateStatus.FAIL
            for gate in group["gates"]
        )
    ]
    missing = [name for name, passed in requirements.items() if not passed]
    status = (
        AccuracyGateStatus.FAIL
        if failed_measurements
        else AccuracyGateStatus.INSUFFICIENT_EVIDENCE
        if missing
        else AccuracyGateStatus.PASS
    )
    return {
        "schema_version": "dubbing.accuracy-evidence-portfolio.v1",
        "status": status,
        "production_claim_passed": status is AccuracyGateStatus.PASS,
        "requirements": requirements,
        "missing_requirements": missing,
        "failed_measurement_count": len(failed_measurements),
        "natural_evidence": {
            "unique_source_count": len(natural_by_source),
            "recording_count": natural_recordings,
            "duration_ms": natural_duration_ms,
            "maximum_speaker_count": natural_speakers,
        },
        "report_count": len(reports),
        "gate_count": len(gates),
        "giga_admission_emitted": False,
    }


def evaluate_accuracy_portfolio(
    report_paths: list[str | Path], output_path: str | Path
) -> dict:
    paths = [Path(path).resolve() for path in report_paths]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    portfolio = build_accuracy_portfolio(reports)
    portfolio["reports"] = [
        {"path": str(path), "sha256": _sha256(path)} for path in paths
    ]
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(portfolio, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return portfolio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a claim-scoped transcription accuracy evidence portfolio"
    )
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    portfolio = evaluate_accuracy_portfolio(args.report, args.output)
    print(
        json.dumps(
            {
                "status": portfolio["status"],
                "production_claim_passed": portfolio["production_claim_passed"],
                "missing_requirements": portfolio["missing_requirements"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
