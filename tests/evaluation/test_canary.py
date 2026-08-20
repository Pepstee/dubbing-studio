import hashlib
import json
from pathlib import Path

import pytest

from dubbing.evaluation.canary import (
    evaluate_canary_result,
    load_canary_manifest,
    validate_canary_bindings,
)
from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    entries = []
    for index, role in enumerate(("development", "holdout", "stress")):
        source = tmp_path / f"source-{index}.wav"
        source.write_bytes(f"source-{index}".encode())
        reference = tmp_path / f"reference-{index}.txt"
        reference.write_text("hello привет", encoding="utf-8")
        entries.append(
            {
                "id": f"lesson-{index}",
                "role": role,
                "source": {
                    "path": str(source),
                    "sha256": _sha256(source),
                    "duration_ms": 1000,
                },
                "reference": {
                    "path": str(reference),
                    "sha256": _sha256(reference),
                    "kind": "provisional",
                },
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "dubbing.historical-canary.v1",
                "canary_id": "fixture",
                "giga_admission_allowed": False,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_binds_every_source_and_reference(tmp_path):
    path = _manifest(tmp_path)
    manifest_path, manifest = load_canary_manifest(path)
    bound = validate_canary_bindings(manifest_path, manifest)
    assert [item["role"] for item in bound] == ["development", "holdout", "stress"]

    Path(bound[1]["reference"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="reference SHA-256 mismatch"):
        validate_canary_bindings(manifest_path, manifest)


def test_operational_pass_does_not_claim_provisional_reference_is_ground_truth(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    result = TranscriptionResult(
        segments=(TranscriptSegment(0, 1000, "hello привет", language="mixed"),),
        text="hello привет",
        backend="fixture",
        model="fixture",
        device="test",
        language="mixed",
        duration_ms=1000,
        confidence_available=False,
        source_sha256=entry["source"]["sha256"],
    )
    report = evaluate_canary_result(
        entry,
        result.to_dict(),
        {"status": "PASS", "metrics": {"repetition_finding_count": 0}},
        runtime_seconds=0.1,
        policy={"maximum_realtime_factor": 0.25},
        execution_fingerprint="frozen",
    )
    assert report["operational_gate"]["status"] == "PASS"
    assert report["agreement_observation"]["accuracy_certified"] is False
    assert report["diarization_claim"]["status"] == "NOT_MEASURED"
    assert report["giga_admission_allowed"] is False


def test_hallucination_quality_failure_fails_canary(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    result = TranscriptionResult(
        segments=(TranscriptSegment(0, 1000, "loops loops loops"),),
        text="loops loops loops",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=1000,
        confidence_available=False,
        source_sha256=entry["source"]["sha256"],
    )
    report = evaluate_canary_result(
        entry,
        result.to_dict(),
        {
            "status": "REPROCESS_REQUIRED",
            "metrics": {"repetition_finding_count": 1},
        },
        runtime_seconds=0.1,
        policy={},
        execution_fingerprint="frozen",
    )
    assert report["operational_gate"]["status"] == "FAIL"
    assert "PATHOLOGICAL_REPETITION" in report["operational_gate"]["failure_reasons"]


def test_uncertain_spans_fail_the_default_canary_policy(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    result = TranscriptionResult(
        segments=(TranscriptSegment(0, 1000, "[UNCERTAIN]", uncertain=True),),
        text="[UNCERTAIN]",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=1000,
        confidence_available=False,
        source_sha256=entry["source"]["sha256"],
    )
    report = evaluate_canary_result(
        entry,
        result.to_dict(),
        {"status": "PASS_WITH_UNCERTAIN_SPANS", "metrics": {}},
        runtime_seconds=0.1,
        policy={},
        execution_fingerprint="frozen",
    )
    assert report["operational_gate"]["status"] == "FAIL"
    assert "QUALITY_PASS_WITH_UNCERTAIN_SPANS" in report["operational_gate"][
        "failure_reasons"
    ]
