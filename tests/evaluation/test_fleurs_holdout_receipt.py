from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "fleurs-test-25x4"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fleurs_holdout_failure_is_hash_bound_and_fail_closed() -> None:
    verdict = _load(FIXTURE / "verdict.json")
    report = _load(FIXTURE / "frozen-policy-report.json")
    manifest = FIXTURE / "manifest.json"
    amendment = FIXTURE / "holdout-selection-amendment.json"
    policy = ROOT / verdict["frozen_policy"]["path"]

    assert _sha256(manifest) == verdict["fixture"]["manifest_sha256"]
    assert _sha256(amendment) == verdict["selection_amendment"]["sha256"]
    assert _sha256(policy) == verdict["frozen_policy"]["sha256"]
    assert _sha256(FIXTURE / "frozen-policy-report.json") == verdict["evidence"][
        "report_sha256"
    ]
    assert report["source_sha256"] == verdict["fixture"]["source_sha256"]
    assert report["sweep_sha256"] == verdict["evidence"]["merged_sweep_sha256"]
    assert report["fixture_gate_passed"] is False
    assert report["promotion_passed"] is False
    assert report["giga_admission_emitted"] is False
    assert verdict["gates"]["aggregate_accuracy_at_least_90_percent"] is True
    assert verdict["gates"]["every_language_accuracy_at_least_90_percent"] is False
    assert verdict["gates"]["untouched_holdout_passed"] is False
    assert verdict["gates"]["giga_admission_emitted"] is False
