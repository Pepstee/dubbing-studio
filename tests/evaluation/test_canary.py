import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import dubbing.evaluation.canary as canary_module
import dubbing.transcription.job as transcription_job_module
from dubbing.evaluation.canary import (
    evaluate_canary_result,
    load_canary_manifest,
    run_canary,
    validate_canary_bindings,
)
from dubbing.transcription.mlx_whisper import MLXWhisperTranscriptionBackend
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)
from dubbing.transcription.speech_regions import FasterWhisperSileroSpeechRegionDetector
from dubbing.transcription.whisperkit import WhisperKitTranscriptionBackend


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


class _LocalBackend:
    identity = "mlx-whisper:synthetic-test"


class _IndependentLocalBackend:
    identity = "faster-whisper:synthetic-independent"


class _LocalSpeechDetector:
    identity = "faster-whisper-silero:synthetic-test"


def _install_fake_coordinator(
    monkeypatch,
    *,
    quality_status="PASS",
    mutate=False,
    after_run=None,
    verify_source=False,
):
    class FakeCoordinator:
        calls = 0
        init_kwargs = []
        processing_ends = []
        source_bindings = []

        def __init__(self, backend, output_dir, **kwargs):
            self.output_dir = Path(output_dir)
            type(self).init_kwargs.append(kwargs)

        def run(self, source, options, *, source_binding=None, processing_end_ms=None):
            type(self).calls += 1
            type(self).processing_ends.append(processing_end_ms)
            type(self).source_bindings.append(source_binding)
            if verify_source:
                transcription_job_module.verify_source_binding(source_binding)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            source_hash = source_binding.sha256
            text = "changed" if mutate and type(self).calls > 1 else "hello привет"
            duration_ms = processing_end_ms or 1000
            result = TranscriptionResult(
                segments=(TranscriptSegment(0, duration_ms, text, language="mixed"),),
                text=text,
                backend="mlx-whisper",
                model="synthetic-test",
                device="test",
                language="mixed",
                duration_ms=duration_ms,
                confidence_available=False,
                source_sha256=source_hash,
            )
            quality = {
                "status": quality_status,
                "metrics": {"repetition_finding_count": 0},
            }
            (self.output_dir / "result.json").write_text(
                json.dumps(result.to_dict()), encoding="utf-8"
            )
            (self.output_dir / "quality-report.json").write_text(
                json.dumps(quality), encoding="utf-8"
            )
            if after_run is not None:
                after_run()
            return result, quality

    monkeypatch.setattr(canary_module, "AdaptiveLongFormCoordinator", FakeCoordinator)
    return FakeCoordinator


def test_manifest_binds_every_source_and_reference(tmp_path):
    path = _manifest(tmp_path)
    manifest_path, manifest = load_canary_manifest(path)
    bound = validate_canary_bindings(manifest_path, manifest)
    assert [item["role"] for item in bound] == ["development", "holdout", "stress"]

    Path(bound[1]["reference"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="reference SHA-256 mismatch"):
        validate_canary_bindings(manifest_path, manifest)


def test_reference_free_entry_reports_operational_evidence_without_invented_accuracy(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["entries"][1]["reference"] = None
    manifest.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch)

    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    summary = run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-1"})
    report = json.loads((output / "lesson-1" / "canary-report.json").read_text())
    frozen = json.loads((output / "frozen-execution.json").read_text())

    assert summary["completed_entries"] == ["lesson-0", "lesson-1"]
    assert report["operational_gate"]["status"] == "PASS"
    assert report["reference_sha256"] is None
    assert report["reference_kind"] is None
    assert report["provisional_reference_metrics"] is None
    assert report["provisional_reference_comparison_scope"]["reference_scope"] == "NOT_AVAILABLE"
    assert report["agreement_observation"]["status"] == "NOT_MEASURED"
    assert report["agreement_observation"]["accuracy_certified"] is False
    frozen_entry = next(item for item in frozen["corpus"]["entries"] if item["id"] == "lesson-1")
    assert frozen_entry["reference_sha256"] is None
    assert frozen_entry["reference_kind"] is None


def test_run_passes_exact_validated_source_binding_to_coordinator(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    created = []
    create_source_binding = canary_module.create_source_binding

    def observe_binding(path, *, expected_sha256=None):
        binding = create_source_binding(path, expected_sha256=expected_sha256)
        created.append(binding)
        return binding

    monkeypatch.setattr(canary_module, "create_source_binding", observe_binding)
    coordinator = _install_fake_coordinator(monkeypatch)

    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})

    assert len(created) == 1
    assert coordinator.source_bindings == [created[0]]
    assert coordinator.source_bindings[0] is created[0]
    assert created[0].sha256 == json.loads(manifest.read_text())["entries"][0]["source"][
        "sha256"
    ]


def test_wrong_manifest_source_digest_fails_before_attempt_or_backend(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["entries"][0]["source"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "output"
    coordinator = _install_fake_coordinator(monkeypatch)
    execution_fingerprint_calls = []

    def unexpected_execution_fingerprint(*args, **kwargs):
        execution_fingerprint_calls.append((args, kwargs))
        raise AssertionError("backend execution evidence must not run")

    monkeypatch.setattr(
        canary_module,
        "_execution_fingerprint",
        unexpected_execution_fingerprint,
    )

    with pytest.raises(TranscriptionError, match="does not match expected SHA-256"):
        run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})

    assert coordinator.calls == 0
    assert execution_fingerprint_calls == []
    assert not output.exists()


def test_source_content_is_read_once_for_binding_and_once_for_coordinator_verifier(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    actual_create_source_binding = transcription_job_module.create_source_binding
    content_reads = []

    def counted_binding(path, *, expected_sha256=None):
        content_reads.append(Path(path).resolve())
        return actual_create_source_binding(path, expected_sha256=expected_sha256)

    monkeypatch.setattr(canary_module, "create_source_binding", counted_binding)
    monkeypatch.setattr(
        transcription_job_module,
        "create_source_binding",
        counted_binding,
    )
    coordinator = _install_fake_coordinator(monkeypatch, verify_source=True)

    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})

    source = Path(json.loads(manifest.read_text())["entries"][0]["source"]["path"]).resolve()
    assert content_reads == [source, source]
    assert coordinator.source_bindings[0].path == source


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
    assert report["evaluation_scope"]["kind"] == "FULL_SOURCE"
    assert report["evaluation_scope"]["operational_quality_recomputed"] is False
    assert report["quality"]["status"] == "PASS"


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
    assert "QUALITY_PASS_WITH_UNCERTAIN_SPANS" in report["operational_gate"]["failure_reasons"]


def test_source_boundary_recomputes_gate_quality_and_observes_tail(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    entry = {
        **entry,
        "source": {
            **entry["source"],
            "duration_ms": 2000,
            "evaluation_end_ms": 1000,
        },
    }
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(0, 1000, "hello привет", language="mixed"),
            TranscriptSegment(1000, 2000, "loops loops loops loops", language="en"),
        ),
        text="hello привет loops loops loops loops",
        backend="fixture",
        model="fixture",
        device="test",
        language="mixed",
        duration_ms=2000,
        confidence_available=False,
        source_sha256=entry["source"]["sha256"],
    )
    full_quality = {
        "status": "REPROCESS_REQUIRED",
        "metrics": {"repetition_finding_count": 1},
    }

    report = evaluate_canary_result(
        entry,
        result.to_dict(),
        full_quality,
        runtime_seconds=0.1,
        policy={"maximum_realtime_factor": 0.25},
        execution_fingerprint="frozen",
    )

    assert report["operational_gate"]["status"] == "PASS"
    assert report["quality"]["status"] == "PASS"
    assert report["full_source_quality_observation"]["quality"] == full_quality
    assert report["full_source_quality_observation"]["used_for_operational_gate"] is False
    assert report["out_of_scope_observation"]["segment_count"] == 1
    assert report["out_of_scope_observation"]["used_for_operational_gate"] is False
    assert report["provisional_reference_comparison_scope"] == {
        "reference_scope": "WHOLE_UNTIMED_REFERENCE",
        "candidate_scope": "SOURCE_TIME_BOUNDARY",
        "observation_only": True,
        "reference_text_was_trimmed": False,
    }
    assert report["full_source_evidence"]["artifacts_modified_for_scope"] is False


def test_source_boundary_clips_crossing_segment_for_quality(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    entry = {
        **entry,
        "source": {**entry["source"], "evaluation_end_ms": 500},
    }
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
        {"status": "PASS", "metrics": {}},
        runtime_seconds=0.1,
        policy={},
        execution_fingerprint="frozen",
    )
    assert report["out_of_scope_observation"]["boundary_crossing_segment_count"] == 1
    assert report["quality"]["metrics"]["duration_ms"] == 500


def test_source_boundary_preserves_hash_bound_explained_silence(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    entry = {
        **entry,
        "source": {
            **entry["source"],
            "duration_ms": 150_000,
            "evaluation_end_ms": 140_000,
            "processing_end_ms": 140_000,
        },
    }
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(0, 10_000, "hello привет", language="mixed"),
            TranscriptSegment(130_000, 140_000, "goodbye пока", language="mixed"),
        ),
        text="hello привет goodbye пока",
        backend="fixture",
        model="fixture",
        device="test",
        language="mixed",
        duration_ms=140_000,
        confidence_available=False,
        source_sha256=entry["source"]["sha256"],
    )
    full_quality = {
        "status": "PASS",
        "metrics": {
            "explained_silence_gaps": [
                    {"start_ms": 10_000, "end_ms": 130_000, "silence_ratio": 1.0}
            ]
        },
    }

    report = evaluate_canary_result(
        entry,
        result.to_dict(),
        full_quality,
        runtime_seconds=1.0,
        policy={"maximum_realtime_factor": 0.25},
        execution_fingerprint="frozen",
    )

    assert report["operational_gate"]["status"] == "PASS"
    assert report["quality"]["status"] == "PASS"
    assert report["quality"]["metrics"]["explained_silence_gap_count"] == 1
    assert report["evaluation_scope"]["known_silence_interval_count"] == 1
    assert (
        report["evaluation_scope"]["known_silence_source"]
        == "FULL_QUALITY_EXPLAINED_SILENCE_GAPS"
    )


def test_source_boundary_rejects_malformed_explained_silence(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    entry = {
        **entry,
        "source": {**entry["source"], "evaluation_end_ms": 500},
    }
    result = TranscriptionResult(
        segments=(TranscriptSegment(0, 500, "hello", language="en"),),
        text="hello",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=1000,
        confidence_available=False,
        source_sha256=entry["source"]["sha256"],
    )

    with pytest.raises(ValueError, match="invalid timestamps"):
        evaluate_canary_result(
            entry,
            result.to_dict(),
            {
                "status": "PASS",
                "metrics": {
                    "explained_silence_gaps": [
                        {"start_ms": True, "end_ms": 400, "silence_ratio": 1.0}
                    ]
                },
            },
            runtime_seconds=0.1,
            policy={},
            execution_fingerprint="frozen",
        )


def test_processing_cap_binds_result_scope_and_realtime_factor(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    entry = {
        **entry,
        "source": {
            **entry["source"],
            "duration_ms": 2000,
            "evaluation_end_ms": 1000,
            "processing_end_ms": 1000,
        },
    }
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
        {"status": "PASS", "metrics": {}},
        runtime_seconds=0.2,
        policy={"maximum_realtime_factor": 0.25},
        execution_fingerprint="frozen",
    )

    assert report["operational_gate"]["status"] == "PASS"
    assert report["realtime_factor"] == pytest.approx(0.2)
    assert report["processing_scope"] == {
        "kind": "SOURCE_TIME_CAP",
        "processing_end_ms": 1000,
        "processed_duration_ms": 1000,
        "full_source_duration_ms": 2000,
        "unprocessed_tail_duration_ms": 1000,
        "full_source_sha256_bound": True,
        "realtime_factor_gate_denominator": "processed_duration_ms",
        "full_source_realtime_factor_observation": pytest.approx(0.1),
    }


def test_processing_cap_rejects_result_that_contains_source_tail(tmp_path):
    manifest_path, manifest = load_canary_manifest(_manifest(tmp_path))
    entry = validate_canary_bindings(manifest_path, manifest)[0]
    entry = {
        **entry,
        "source": {
            **entry["source"],
            "duration_ms": 2000,
            "evaluation_end_ms": 1000,
            "processing_end_ms": 1000,
        },
    }
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(0, 1000, "hello привет", language="mixed"),
            TranscriptSegment(1000, 2000, "unrelated tail", language="en"),
        ),
        text="hello привет unrelated tail",
        backend="fixture",
        model="fixture",
        device="test",
        language="mixed",
        duration_ms=2000,
        confidence_available=False,
        source_sha256=entry["source"]["sha256"],
    )

    report = evaluate_canary_result(
        entry,
        result.to_dict(),
        {"status": "PASS", "metrics": {}},
        runtime_seconds=0.1,
        policy={},
        execution_fingerprint="frozen",
    )

    assert report["operational_gate"]["status"] == "FAIL"
    assert "RESULT_PROCESSING_SCOPE_MISMATCH" in report["operational_gate"]["failure_reasons"]


def test_holdout_requires_valid_development_measurement_receipt(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch)

    with pytest.raises(ValueError, match="development canary before"):
        run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-1"})

    development = run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    assert development["status"] == "PARTIAL"
    holdout = run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-1"})
    assert holdout["completed_entries"] == ["lesson-0", "lesson-1"]
    assert holdout["missing_entries"] == ["lesson-2"]


def test_uncertain_but_structurally_safe_development_unlocks_holdout(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch, quality_status="PASS_WITH_UNCERTAIN_SPANS")

    development = run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    assert development["status"] == "FAIL"
    holdout = run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-1"})

    assert holdout["completed_entries"] == ["lesson-0", "lesson-1"]
    assert holdout["operational_failures"] == ["lesson-0", "lesson-1"]
    assert holdout["giga_admission_allowed"] is False


def test_full_selection_runs_development_before_later_roles(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    coordinator = _install_fake_coordinator(monkeypatch)

    summary = run_canary(manifest, output, _LocalBackend())

    assert summary["status"] == "PASS"
    assert summary["completed_entries"] == ["lesson-0", "lesson-1", "lesson-2"]
    assert coordinator.calls == 3


def test_failed_development_blocks_later_roles(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch, quality_status="REPROCESS_REQUIRED")
    summary = run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    assert summary["status"] == "FAIL"

    with pytest.raises(ValueError, match="not measurement-safe"):
        run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-1"})


def test_execution_freeze_binds_corpus_and_implementation(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"fixture ffmpeg")
    monkeypatch.setattr(canary_module, "ffmpeg_executable", lambda: str(ffmpeg))
    _install_fake_coordinator(monkeypatch)
    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})

    frozen = json.loads((output / "frozen-execution.json").read_text())
    assert len(frozen["corpus_sha256"]) == 64
    assert len(frozen["implementation_sha256"]) == 64
    assert {item["id"] for item in frozen["corpus"]["entries"]} == {
        "lesson-0",
        "lesson-1",
        "lesson-2",
    }
    assert any(
        item["path"] == "dubbing/evaluation/canary.py" for item in frozen["implementation"]["files"]
    )
    assert any(item["path"] == "dubbing/media.py" for item in frozen["implementation"]["files"])
    assert frozen["media_runtime"]["ffmpeg"] == {
        "path": str(ffmpeg.resolve()),
        "sha256": _sha256(ffmpeg),
        "size_bytes": ffmpeg.stat().st_size,
    }


def test_ffmpeg_binary_change_invalidates_frozen_execution(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"first ffmpeg")
    monkeypatch.setattr(canary_module, "ffmpeg_executable", lambda: str(ffmpeg))
    _install_fake_coordinator(monkeypatch)
    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})

    ffmpeg.write_bytes(b"changed ffmpeg")
    with pytest.raises(ValueError, match="execution changed after it was frozen"):
        run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})

    assert sorted(path.name for path in (output / "lesson-0/attempts").glob("*.json")) == [
        "000001-completed.json",
        "000001-started.json",
    ]


def test_execution_drift_during_transcription_writes_only_failed_terminal(
    tmp_path, monkeypatch
):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"first ffmpeg")
    monkeypatch.setattr(canary_module, "ffmpeg_executable", lambda: str(ffmpeg))
    _install_fake_coordinator(
        monkeypatch,
        after_run=lambda: ffmpeg.write_bytes(b"changed during transcription"),
    )

    with pytest.raises(ValueError, match="execution changed after it was frozen"):
        run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})

    attempts = output / "lesson-0/attempts"
    assert sorted(path.name for path in attempts.glob("*.json")) == [
        "000001-failed.json",
        "000001-started.json",
    ]
    assert not (output / "lesson-0/canary-run-receipt.json").exists()


def test_whisperkit_executable_change_invalidates_frozen_execution(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    executable = tmp_path / "whisperkit-cli"
    executable.write_bytes(b"first whisperkit")
    server = SimpleNamespace(
        executable=str(executable),
        model_sha256="a" * 64,
        model_size_bytes=123,
        version="v1.0.0",
    )
    backend = WhisperKitTranscriptionBackend(model="fixture", server=server)
    _install_fake_coordinator(monkeypatch)
    run_canary(manifest, output, backend, selected_ids={"lesson-0"})

    frozen = json.loads((output / "frozen-execution.json").read_text())
    assert frozen["backend"]["runtime_executable"]["sha256"] == _sha256(executable)
    executable.write_bytes(b"changed whisperkit")
    with pytest.raises(ValueError, match="execution changed after it was frozen"):
        run_canary(manifest, output, backend, selected_ids={"lesson-0"})


def test_unowned_whisperkit_server_is_rejected_before_attempt(tmp_path):
    output = tmp_path / "output"
    backend = WhisperKitTranscriptionBackend(model="fixture")

    with pytest.raises(ValueError, match="owned local WhisperKit server"):
        run_canary(_manifest(tmp_path), output, backend, selected_ids={"lesson-0"})

    assert not (output / "lesson-0/attempts").exists()


def test_local_retry_model_tree_and_runtime_are_frozen(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    model = tmp_path / "mlx-model"
    model.mkdir()
    weights = model / "weights.safetensors"
    weights.write_bytes(b"first weights")
    runtime_version = {"value": "0.4.3"}
    monkeypatch.setattr(
        canary_module,
        "_installed_distribution_version",
        lambda package: runtime_version["value"],
    )
    retry = MLXWhisperTranscriptionBackend(model=str(model), temperature=0)
    _install_fake_coordinator(monkeypatch)
    run_canary(
        manifest,
        output,
        _LocalBackend(),
        retry_backend=retry,
        selected_ids={"lesson-0"},
    )

    frozen = json.loads((output / "frozen-execution.json").read_text())
    assert frozen["retry_backend"]["runtime_version"] == "0.4.3"
    assert frozen["retry_backend"]["local_model"]["path"] == str(model.resolve())
    first_tree = frozen["retry_backend"]["local_model"]["tree_sha256"]
    weights.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="execution changed after it was frozen"):
        run_canary(
            manifest,
            output,
            _LocalBackend(),
            retry_backend=retry,
            selected_ids={"lesson-0"},
        )
    assert first_tree != canary_module._model_tree_evidence(model)["tree_sha256"]


def test_local_retry_runtime_change_invalidates_frozen_execution(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    model = tmp_path / "mlx-model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"weights")
    runtime_version = {"value": "0.4.3"}
    monkeypatch.setattr(
        canary_module,
        "_installed_distribution_version",
        lambda package: runtime_version["value"],
    )
    retry = MLXWhisperTranscriptionBackend(model=str(model), temperature=0)
    _install_fake_coordinator(monkeypatch)
    run_canary(
        manifest,
        output,
        _LocalBackend(),
        retry_backend=retry,
        selected_ids={"lesson-0"},
    )

    runtime_version["value"] = "0.4.4"
    with pytest.raises(ValueError, match="execution changed after it was frozen"):
        run_canary(
            manifest,
            output,
            _LocalBackend(),
            retry_backend=retry,
            selected_ids={"lesson-0"},
        )


def test_nonlocal_mlx_canary_model_is_rejected_before_attempt(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr(
        canary_module,
        "_installed_distribution_version",
        lambda package: "0.4.3",
    )
    backend = MLXWhisperTranscriptionBackend(model="publisher/remote-model", temperature=0)

    with pytest.raises(ValueError, match="existing local transcription model directory"):
        run_canary(_manifest(tmp_path), output, backend, selected_ids={"lesson-0"})

    assert not (output / "lesson-0/attempts").exists()


def test_retry_backend_and_candidate_languages_are_frozen_and_forwarded(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["execution"] = {"candidate_languages": ["en", "ru"]}
    document["entries"][0]["source"]["evaluation_end_ms"] = 800
    document["entries"][0]["source"]["processing_end_ms"] = 800
    manifest.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "output"
    coordinator = _install_fake_coordinator(monkeypatch)

    run_canary(
        manifest,
        output,
        _LocalBackend(),
        retry_backend=_IndependentLocalBackend(),
        silence_verification_detector=_LocalSpeechDetector(),
        targeted_retry_region_detector=None,
        selected_ids={"lesson-0"},
    )

    frozen = json.loads((output / "frozen-execution.json").read_text())
    assert frozen["retry_backend"]["identity"] == _IndependentLocalBackend.identity
    assert frozen["silence_verification_detector"]["identity"] == _LocalSpeechDetector.identity
    assert frozen["targeted_retry_region_detector"] is None
    assert len(frozen["silence_verification_detector"]["implementation"]["source_sha256"]) == 64
    assert len(frozen["retry_backend"]["implementation"]["source_sha256"]) == 64
    assert frozen["execution"]["candidate_languages"] == ["en", "ru"]
    assert frozen["corpus"]["entries"][0]["source_evaluation_end_ms"] == 800
    assert frozen["corpus"]["entries"][0]["source_processing_end_ms"] == 800
    assert coordinator.init_kwargs[0]["retry_backend"].identity == (
        _IndependentLocalBackend.identity
    )
    assert coordinator.init_kwargs[0]["silence_verification_detector"].identity == (
        _LocalSpeechDetector.identity
    )
    assert coordinator.init_kwargs[0]["targeted_retry_region_detector"] is None
    assert coordinator.init_kwargs[0]["candidate_languages"] == ("en", "ru")
    assert coordinator.processing_ends == [800]


def test_legacy_speech_detector_populates_both_frozen_roles(tmp_path, monkeypatch):
    output = tmp_path / "output"
    coordinator = _install_fake_coordinator(monkeypatch)

    run_canary(
        _manifest(tmp_path),
        output,
        _LocalBackend(),
        speech_region_detector=_LocalSpeechDetector(),
        selected_ids={"lesson-0"},
    )

    frozen = json.loads((output / "frozen-execution.json").read_text())
    assert frozen["silence_verification_detector"]["identity"] == _LocalSpeechDetector.identity
    assert frozen["targeted_retry_region_detector"]["identity"] == _LocalSpeechDetector.identity
    assert (
        coordinator.init_kwargs[0]["silence_verification_detector"].identity
        == _LocalSpeechDetector.identity
    )
    assert (
        coordinator.init_kwargs[0]["targeted_retry_region_detector"].identity
        == _LocalSpeechDetector.identity
    )


def test_detector_role_change_is_rejected_after_execution_freeze(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch)
    detector = _LocalSpeechDetector()
    run_canary(
        manifest,
        output,
        _LocalBackend(),
        silence_verification_detector=detector,
        targeted_retry_region_detector=None,
        selected_ids={"lesson-0"},
    )

    with pytest.raises(ValueError, match="execution changed after it was frozen"):
        run_canary(
            manifest,
            output,
            _LocalBackend(),
            silence_verification_detector=detector,
            targeted_retry_region_detector=detector,
            selected_ids={"lesson-0"},
        )


def test_retry_backend_must_be_independent(tmp_path):
    with pytest.raises(ValueError, match="primary backend identity"):
        run_canary(
            _manifest(tmp_path),
            tmp_path / "output",
            _LocalBackend(),
            retry_backend=_LocalBackend(),
            selected_ids={"lesson-0"},
        )


def test_retry_backend_must_be_local(tmp_path):
    class CloudRetryBackend:
        identity = "elevenlabs:scribe-v2"

    with pytest.raises(ValueError, match="local retry backend"):
        run_canary(
            _manifest(tmp_path),
            tmp_path / "output",
            _LocalBackend(),
            retry_backend=CloudRetryBackend(),
            selected_ids={"lesson-0"},
        )


def test_evaluate_only_requires_and_preserves_transcription_provenance(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch)
    with pytest.raises(ValueError, match="validated transcription receipt"):
        run_canary(
            manifest,
            output,
            _LocalBackend(),
            selected_ids={"lesson-0"},
            evaluate_only=True,
        )

    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    receipt_path = output / "lesson-0" / "canary-run-receipt.json"
    origin_hash = _sha256(receipt_path)
    run_canary(
        manifest,
        output,
        _LocalBackend(),
        selected_ids={"lesson-0"},
        evaluate_only=True,
    )
    receipt = json.loads(receipt_path.read_text())
    report = json.loads((output / "lesson-0" / "canary-report.json").read_text())
    assert receipt["origin"] == "LOCAL_TRANSCRIPTION"
    assert receipt["evaluation_count"] == 1
    assert report["evaluation_provenance"]["mode"] == "EVALUATE_ONLY"
    assert report["evaluation_provenance"]["origin_receipt_sha256"] == origin_hash


def test_nonfinite_or_ledger_inconsistent_receipt_is_rejected(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch)
    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    receipt_path = output / "lesson-0" / "canary-run-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["initial_runtime_seconds"] = float("nan")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-1"})
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert "lesson-0" in summary["invalid_evidence"]


def test_replay_result_mismatch_fails_closed(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch, mutate=True)
    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    summary = run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    report = json.loads((output / "lesson-0" / "canary-report.json").read_text())
    assert summary["status"] == "FAIL"
    assert "CHECKPOINT_REPLAY_RESULT_MISMATCH" in report["operational_gate"]["failure_reasons"]
    assert report["operational_gate"]["passed"] is False

    third = run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    receipt = json.loads((output / "lesson-0" / "canary-run-receipt.json").read_text())
    assert third["status"] == "FAIL"
    assert receipt["replay_mismatch_count"] == 1


def test_attempt_ledger_is_immutable_and_runtime_does_not_use_mtime(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    _install_fake_coordinator(monkeypatch)
    entry_output = output / "lesson-0"
    entry_output.mkdir(parents=True)
    (entry_output / "manifest.json").write_text("{}", encoding="utf-8")
    (entry_output / "progress.json").write_text("{}", encoding="utf-8")
    (entry_output / "manifest.json").touch()
    (entry_output / "progress.json").touch()

    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    receipt = json.loads((entry_output / "canary-run-receipt.json").read_text())
    events = sorted((entry_output / "attempts").glob("*.json"))
    assert [item.name for item in events] == [
        "000001-completed.json",
        "000001-started.json",
    ]
    assert receipt["runtime_measurement"] == "append-only-monotonic-attempt-ledger-v1"
    assert "recovered_checkpoint_seconds" not in receipt
    assert receipt["runtime_accounting_complete"] is True


def test_failed_replay_invalidates_stale_receipt_and_summary(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    coordinator = _install_fake_coordinator(monkeypatch)
    run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})
    receipt_path = output / "lesson-0" / "canary-run-receipt.json"
    original_receipt = receipt_path.read_bytes()

    def fail_replay(self, source, options, *, source_binding=None, processing_end_ms=None):
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(coordinator, "run", fail_replay)
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        run_canary(manifest, output, _LocalBackend(), selected_ids={"lesson-0"})

    assert receipt_path.read_bytes() == original_receipt
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert summary["completed_entries"] == []
    assert "latest attempt did not complete" in summary["invalid_evidence"]["lesson-0"]
    assert sorted(item.name for item in (receipt_path.parent / "attempts").glob("*.json")) == [
        "000001-completed.json",
        "000001-started.json",
        "000002-failed.json",
        "000002-started.json",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document["entries"][0].update(id="Bad ID"), "kebab-case"),
        (
            lambda document: document.update(policy={"allowed_quality_statuses": ["MADE_UP"]}),
            "invalid status",
        ),
        (
            lambda document: document.update(policy={"maximum_realtime_factor": float("inf")}),
            "finite",
        ),
        (
            lambda document: document.update(execution={"candidate_languages": ["en", "en"]}),
            "candidate_languages",
        ),
        (
            lambda document: document.update(execution={"candidate_languages": []}),
            "candidate_languages",
        ),
        (
            lambda document: document["entries"][0]["source"].update(evaluation_end_ms=0),
            "evaluation_end_ms",
        ),
        (
            lambda document: document["entries"][0]["source"].update(evaluation_end_ms=1001),
            "evaluation_end_ms",
        ),
        (
            lambda document: document["entries"][0]["source"].update(processing_end_ms=0),
            "processing_end_ms",
        ),
        (
            lambda document: document["entries"][0]["source"].update(
                evaluation_end_ms=900, processing_end_ms=800
            ),
            "must not exceed processing_end_ms",
        ),
    ],
)
def test_manifest_strictly_validates_slugs_and_policy(tmp_path, mutation, message):
    path = _manifest(tmp_path)
    document = json.loads(path.read_text())
    mutation(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_canary_manifest(path)


def test_canary_rejects_nonlocal_backend(tmp_path):
    class CloudBackend:
        identity = "elevenlabs:scribe-v2"

    with pytest.raises(ValueError, match="local transcription backend"):
        run_canary(
            _manifest(tmp_path),
            tmp_path / "output",
            CloudBackend(),
            selected_ids={"lesson-0"},
        )


def test_cli_returns_nonzero_for_partial_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(canary_module, "build_backend", lambda args: _LocalBackend())
    monkeypatch.setattr(canary_module, "build_retry_backend", lambda args: None)
    monkeypatch.setattr(
        canary_module,
        "run_canary",
        lambda *args, **kwargs: {"status": "PARTIAL"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dubbing-canary",
            "--manifest",
            str(_manifest(tmp_path)),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        canary_module.main()
    assert exc.value.code == 2
    assert '"status": "PARTIAL"' in capsys.readouterr().out


def test_cli_builds_and_forwards_matching_retry_arguments(tmp_path, monkeypatch):
    observed = {}
    monkeypatch.setattr(canary_module, "build_backend", lambda args: _LocalBackend())

    def fake_build_retry(args):
        observed["retry_backend"] = args.retry_backend
        observed["retry_model"] = args.retry_model
        observed["retry_device"] = args.retry_device
        observed["retry_compute_type"] = args.retry_compute_type
        observed["retry_mlx_temperature"] = args.retry_mlx_temperature
        return _IndependentLocalBackend()

    def fake_run(*args, **kwargs):
        observed["forwarded"] = kwargs["retry_backend"]
        observed["silence_verification_detector"] = kwargs["silence_verification_detector"]
        observed["targeted_retry_region_detector"] = kwargs["targeted_retry_region_detector"]
        return {"status": "PASS"}

    monkeypatch.setattr(canary_module, "build_retry_backend", fake_build_retry)
    monkeypatch.setattr(canary_module, "run_canary", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dubbing-canary",
            "--manifest",
            str(_manifest(tmp_path)),
            "--output",
            str(tmp_path / "output"),
            "--retry-backend",
            "faster-whisper",
            "--retry-model",
            "/tmp/retry-model",
            "--retry-device",
            "cpu",
            "--retry-compute-type",
            "int8",
            "--retry-mlx-temperature",
            "0",
        ],
    )

    canary_module.main()

    assert observed["retry_backend"] == "faster-whisper"
    assert observed["retry_model"] == "/tmp/retry-model"
    assert observed["retry_device"] == "cpu"
    assert observed["retry_compute_type"] == "int8"
    assert observed["retry_mlx_temperature"] == [0.0]
    assert observed["forwarded"].identity == _IndependentLocalBackend.identity
    assert (
        observed["silence_verification_detector"].identity
        == FasterWhisperSileroSpeechRegionDetector().identity
    )
    assert observed["targeted_retry_region_detector"] is None
