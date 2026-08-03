import hashlib
import json
from pathlib import Path

from dubbing.evaluation.cli import evaluate_fixture
from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_quality_uses_hash_bound_known_silence(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    reference = tmp_path / "reference.txt"
    reference.write_text("before after", encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(0, 1000, "before"),
            TranscriptSegment(70_000, 71_000, "after"),
        ),
        text="before after",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=71_000,
        confidence_available=False,
        source_sha256=_sha256(source),
    )
    candidate.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "fixture_id": "known-silence",
                "source": {
                    "path": str(source),
                    "sha256": _sha256(source),
                    "duration_ms": 71_000,
                },
                "reference": {
                    "path": str(reference),
                    "sha256": _sha256(reference),
                    "kind": "fixture",
                },
                "candidate": {
                    "path": str(candidate),
                    "sha256": _sha256(candidate),
                },
                "backend": {"provider": "fixture"},
                "evaluation": {
                    "known_silence_intervals": [
                        {"start_ms": 1000, "end_ms": 70_000}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_fixture(manifest)

    assert report["quality"]["status"] == "PASS"
    assert report["quality"]["metrics"]["large_gap_count"] == 0
    assert report["quality"]["metrics"]["explained_silence_gap_count"] == 1
