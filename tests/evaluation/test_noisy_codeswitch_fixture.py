from __future__ import annotations

import hashlib
import json
import wave
from array import array
from pathlib import Path

from scripts.build_noisy_codeswitch_fixture import build_noisy_codeswitch_fixture


REPOSITORY = Path(__file__).resolve().parents[2]


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path, value: int) -> None:
    samples = array("h", [value, -value] * 1600)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16000)
        destination.writeframes(samples.tobytes())


def test_noisy_codeswitch_fixture_is_deterministic_and_exactly_bound(tmp_path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    spans = []
    for index, language in enumerate(("en", "ru", "ro", "ko"), start=1):
        clip = clips / f"{language}.wav"
        _write_wav(clip, index * 1000)
        spans.append(
            {
                "id": language,
                "language": language,
                "text": f"text {language}",
                "clip_relative_path": clip.name,
                "clip_sha256": _sha256(clip),
            }
        )
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps({"source": {"sha256": "a" * 64}, "spans": spans}),
        encoding="utf-8",
    )

    first = build_noisy_codeswitch_fixture(
        base,
        clips,
        tmp_path / "first.wav",
        tmp_path / "first.json",
        clips_per_language=1,
    )
    second = build_noisy_codeswitch_fixture(
        base,
        clips,
        tmp_path / "second.wav",
        tmp_path / "second.json",
        clips_per_language=1,
    )

    assert first["source"]["sha256"] == second["source"]["sha256"]
    assert first["language_counts"] == {"en": 1, "ru": 1, "ro": 1, "ko": 1}
    assert [span["language"] for span in first["spans"]] == ["en", "ru", "ro", "ko"]
    assert all(
        earlier["end_ms"] <= later["start_ms"]
        for earlier, later in zip(first["spans"], first["spans"][1:])
    )
    assert first["selection_policy"]["reference_text_is_human_ground_truth"]
    assert first["selection_policy"]["reference_timestamps_are_exact_derived_boundaries"]
    assert not first["selection_policy"]["accuracy_certification_eligible"]
    assert not first["giga_admission_emitted"]
    assert first["derivation"]["speech_target_rms_dbfs"] == -20.0


def test_committed_noisy_codeswitch_verdicts_are_hash_bound_and_fail_closed() -> None:
    for fixture_name in (
        "fleurs-noisy-codeswitch-rms20-snr20-25x4",
        "fleurs-noisy-codeswitch-rms20-snr30-25x4",
    ):
        fixture_dir = REPOSITORY / "benchmarks" / "fixtures" / fixture_name
        verdict = json.loads((fixture_dir / "verdict.json").read_text(encoding="utf-8"))
        manifest = fixture_dir / "manifest.json"
        candidate = verdict.get("selected_candidate", verdict.get("candidate"))
        report = REPOSITORY / candidate["report_path"]

        assert _sha256(manifest) == verdict["fixture"]["manifest_sha256"]
        assert _sha256(report) == candidate["report_sha256"]
        assert _sha256(REPOSITORY / "scripts" / "build_noisy_codeswitch_fixture.py") == (
            verdict["fixture"]["derivation_builder_sha256"]
        )
        assert verdict["verdict"] == "FAIL_CLOSED"
        assert verdict["scope"] == "DEVELOPMENT_ONLY"
        assert not verdict["gates"]["every_language_accuracy_at_least_90_percent"]
        assert not verdict["gates"]["accuracy_certification_eligible"]
        assert not verdict["gates"]["production_scope_certified"]
        assert not verdict["gates"]["giga_admission_emitted"]
