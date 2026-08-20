from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from dubbing.evaluation.cloud_teacher_canary import prepare_cloud_teacher_canary
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionResult,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wave(path: Path, duration_seconds: int = 45) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 16_000 * duration_seconds)


def _parent(source: Path, path: Path) -> None:
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(
                1_000,
                4_000,
                "clean English control",
                words=(
                    TranscriptWord(1_000, 1_700, " clean"),
                    TranscriptWord(1_800, 2_700, " English"),
                    TranscriptWord(2_800, 4_000, " control"),
                ),
                language="en",
            ),
            TranscriptSegment(
                24_000,
                28_000,
                "[UNCERTAIN: INDEPENDENT TRANSCRIPTIONS DISAGREE]",
                uncertain=True,
            ),
        ),
        text=(
            "clean English control "
            "[UNCERTAIN: INDEPENDENT TRANSCRIPTIONS DISAGREE]"
        ),
        backend="fixture",
        model="fixture",
        device="cpu",
        language=None,
        duration_ms=45_000,
        confidence_available=False,
        source_sha256=_sha256(source),
        provenance={"fixture": True},
    )
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest(source: Path, parent: Path, **updates) -> dict:
    document = {
        "schema_version": "dubbing.cloud-teacher-canary-spans.v1",
        "canary_id": "fixture-canary",
        "parent_source_sha256": _sha256(source),
        "parent_local_result_sha256": _sha256(parent),
        "network_allowed": False,
        "giga_admission_allowed": False,
        "spans": [
            {"id": "control", "role": "control", "start_ms": 0, "end_ms": 20_000},
            {"id": "hard", "role": "hard", "start_ms": 20_000, "end_ms": 40_000},
        ],
    }
    document.update(updates)
    return document


def _inputs(tmp_path: Path):
    source = tmp_path / "source.wav"
    parent = tmp_path / "parent.json"
    manifest = tmp_path / "spans.json"
    _wave(source)
    _parent(source, parent)
    manifest.write_text(json.dumps(_manifest(source, parent)), encoding="utf-8")
    return source, parent, manifest


def test_preparer_emits_atomic_lossless_source_bound_inputs(tmp_path, monkeypatch) -> None:
    source, parent, manifest = _inputs(tmp_path)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "must-not-be-read-or-serialized")
    output = tmp_path / "prepared"

    report = prepare_cloud_teacher_canary(source, parent, manifest, output)

    assert report["network_used"] is False
    assert report["credentials_accessed"] is False
    assert report["cloud_allowed"] is False
    assert report["giga_admission_allowed"] is False
    assert [item["duration_ms"] for item in report["spans"]] == [20_000, 20_000]
    assert not any(path.is_symlink() for path in output.rglob("*"))
    assert "must-not-be-read" not in json.dumps(report)
    for row in report["spans"]:
        clip = output / row["clip"]
        local_path = output / row["local_result"]
        local = json.loads(local_path.read_text(encoding="utf-8"))
        assert clip.read_bytes()[:4] == b"fLaC"
        assert _sha256(clip) == row["clip_sha256"] == local["source_sha256"]
        assert _sha256(local_path) == row["local_result_sha256"]
        assert local["duration_ms"] == row["duration_ms"]
        assert local["provenance"]["parent_local_result_sha256"] == _sha256(parent)
    control = json.loads((output / "local-results/control.json").read_text())
    hard = json.loads((output / "local-results/hard.json").read_text())
    assert control["segments"][0]["start_ms"] == 1_000
    assert control["segments"][0]["words"][0]["start_ms"] == 1_000
    assert hard["segments"][0]["start_ms"] == 4_000
    assert hard["segments"][0]["uncertain"] is True
    assert json.loads((output / "preparation-manifest.json").read_text()) == report


def test_preparer_rejects_boundary_crossing_without_partial_output(tmp_path) -> None:
    source, parent, manifest = _inputs(tmp_path)
    document = _manifest(source, parent)
    document["spans"] = [
        {"id": "crossing", "role": "hard", "start_ms": 2_000, "end_ms": 22_000}
    ]
    manifest.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="crosses .* segment boundary"):
        prepare_cloud_teacher_canary(source, parent, manifest, output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".prepared.*"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda doc: doc.update(parent_source_sha256="0" * 64), "source SHA-256"),
        (lambda doc: doc.update(network_allowed=True), "network_allowed=false"),
        (
            lambda doc: doc.update(
                spans=[
                    {
                        "id": "too-short",
                        "role": "hard",
                        "start_ms": 0,
                        "end_ms": 19_999,
                    }
                ]
            ),
            "20-60 seconds",
        ),
        (
            lambda doc: doc.update(
                spans=[
                    {"id": "a", "role": "hard", "start_ms": 0, "end_ms": 25_000},
                    {
                        "id": "b",
                        "role": "control",
                        "start_ms": 20_000,
                        "end_ms": 40_000,
                    },
                ]
            ),
            "must not overlap",
        ),
    ],
)
def test_preparer_rejects_malformed_or_drifted_manifests(
    tmp_path, mutation, message
) -> None:
    source, parent, manifest = _inputs(tmp_path)
    document = _manifest(source, parent)
    mutation(document)
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_cloud_teacher_canary(source, parent, manifest, tmp_path / "prepared")


def test_preparer_rejects_symlink_inputs_and_existing_outputs(tmp_path) -> None:
    source, parent, manifest = _inputs(tmp_path)
    source_link = tmp_path / "source-link.wav"
    source_link.symlink_to(source)
    with pytest.raises(ValueError, match="must not be a symlink"):
        prepare_cloud_teacher_canary(
            source_link, parent, manifest, tmp_path / "prepared-link"
        )

    output = tmp_path / "prepared-existing"
    output.mkdir()
    marker = output / "operator-data"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        prepare_cloud_teacher_canary(source, parent, manifest, output)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_preparer_rejects_expected_derived_hash_drift_atomically(tmp_path) -> None:
    source, parent, manifest = _inputs(tmp_path)
    document = _manifest(source, parent)
    document["spans"][0]["expected_clip_sha256"] = "f" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="derived clip SHA-256 drift"):
        prepare_cloud_teacher_canary(source, parent, manifest, output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".prepared.*"))
