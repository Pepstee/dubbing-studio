import json
from pathlib import Path

import pytest

from dubbing.transcription import korean_spacing
from dubbing.transcription.korean_spacing import (
    KIWIPIEPY_MODEL_ARCHIVE,
    KIWIPIEPY_MODEL_VERSION,
    KIWIPIEPY_WINDOWS_ARCHIVE,
    KIWIPIEPY_VERSION,
    kiwi_runtime_receipt,
    normalize_candidate_spacing,
)


def _candidate(text: str = "한국어띄어 쓰기") -> dict:
    return {
        "text": text,
        "segments": [
            {
                "start_ms": 0,
                "end_ms": 1000,
                "text": text,
                "words": [
                    {"start_ms": 0, "end_ms": 600, "text": "한국어띄어", "probability": 0.9},
                    {"start_ms": 600, "end_ms": 1000, "text": " 쓰기", "probability": 0.8},
                ],
            }
        ],
    }


def test_spacing_only_normalization_retimes_words_monotonically() -> None:
    result = normalize_candidate_spacing(
        _candidate(), lambda _: "한국어 띄어쓰기", provider="test"
    )

    assert result["text"] == "한국어 띄어쓰기"
    assert result["spacing_normalization"]["status"] == "APPLIED"
    assert [word["text"] for word in result["segments"][0]["words"]] == [
        "한국어",
        "띄어쓰기",
    ]
    words = result["segments"][0]["words"]
    assert words[0]["start_ms"] == 0
    assert words[0]["end_ms"] <= words[1]["start_ms"]
    assert words[1]["end_ms"] == 1000
    assert words[0]["probability"] == 0.9
    assert words[1]["probability"] == pytest.approx(0.85)


def test_normalization_rejects_any_non_whitespace_change() -> None:
    original = _candidate()
    result = normalize_candidate_spacing(
        original, lambda _: "한국어 띄어쓰기요", provider="test"
    )

    assert result["text"] == original["text"]
    assert result["spacing_normalization"]["status"] == "REJECTED_CHARACTER_CHANGE"


def test_normalization_rejects_candidate_segment_misalignment() -> None:
    original = _candidate()
    original["text"] = "다른 내용"
    result = normalize_candidate_spacing(original, lambda text: text, provider="test")

    assert result["spacing_normalization"]["status"] == "REJECTED_SOURCE_ALIGNMENT"


def test_normalization_rejects_missing_word_timestamps() -> None:
    original = _candidate()
    original["segments"][0]["words"] = []
    result = normalize_candidate_spacing(original, lambda text: text, provider="test")

    assert result["spacing_normalization"]["status"] == "REJECTED_TIMESTAMP_EVIDENCE"


def test_unchanged_spacing_is_explicit() -> None:
    original = _candidate()
    result = normalize_candidate_spacing(original, lambda text: text, provider="test")

    assert result["spacing_normalization"]["status"] == "UNCHANGED"


def test_kiwi_download_receipt_is_exact_and_version_locked(monkeypatch) -> None:
    assert KIWIPIEPY_WINDOWS_ARCHIVE["bytes"] == 3613895
    assert KIWIPIEPY_MODEL_ARCHIVE["bytes"] == 87976912

    def receipt(package: str, _: str) -> dict:
        return {
            "package": package,
            "version": "0.23.2" if package == "kiwipiepy" else "0.23.0",
            "tree_sha256": "a" * 64,
            "file_count": 1,
            "installed_bytes": 1,
        }

    monkeypatch.setattr(korean_spacing, "_distribution_tree_receipt", receipt)
    result = kiwi_runtime_receipt()

    assert result["expected_download_bytes"] == 91590807
    assert result["runtime"]["version"] == "0.23.2"
    assert result["model"]["version"] == "0.23.0"


def test_download_gate_matches_runtime_constants_and_remains_unauthorized() -> None:
    repository = Path(__file__).resolve().parents[2]
    gate = json.loads(
        (
            repository
            / "benchmarks"
            / "fixtures"
            / "fleurs-validation-25x4"
            / "kiwi-spacing-download-gate.json"
        ).read_text(encoding="utf-8")
    )

    assert gate["candidate"]["packages"] == [
        {"name": "kiwipiepy", "version": KIWIPIEPY_VERSION, **KIWIPIEPY_WINDOWS_ARCHIVE},
        {
            "name": "kiwipiepy_model",
            "version": KIWIPIEPY_MODEL_VERSION,
            **KIWIPIEPY_MODEL_ARCHIVE,
        },
    ]
    assert gate["candidate"]["total_download_bytes"] == 91590807
    assert not gate["authorization"]["download_authorized"]
    assert not gate["authorization"]["download_started"]
    assert not gate["giga_admission_emitted"]


def test_kiwi_runtime_rejects_unpinned_version(monkeypatch) -> None:
    def receipt(package: str, _: str) -> dict:
        return {
            "package": package,
            "version": "99.0" if package == "kiwipiepy" else "0.23.0",
            "tree_sha256": "a" * 64,
            "file_count": 1,
            "installed_bytes": 1,
        }

    monkeypatch.setattr(korean_spacing, "_distribution_tree_receipt", receipt)
    with pytest.raises(RuntimeError, match="version mismatch"):
        kiwi_runtime_receipt()
