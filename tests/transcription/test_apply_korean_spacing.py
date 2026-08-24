from __future__ import annotations

import json

import pytest

from scripts import apply_korean_spacing as command


def _sweep() -> dict:
    return {
        "spans": [
            {
                "id": "ko-1",
                "candidates": [
                    {
                        "forced_language": "ko",
                        "text": "한국어띄어 쓰기",
                        "segments": [
                            {
                                "start_ms": 0,
                                "end_ms": 1000,
                                "text": "한국어띄어 쓰기",
                                "words": [
                                    {
                                        "start_ms": 0,
                                        "end_ms": 600,
                                        "text": "한국어띄어",
                                    },
                                    {
                                        "start_ms": 600,
                                        "end_ms": 1000,
                                        "text": " 쓰기",
                                    },
                                ],
                            }
                        ],
                    },
                    {"forced_language": None, "text": "untouched", "segments": []},
                ],
            }
        ]
    }


def test_apply_korean_spacing_binds_runtime_and_source(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    source.write_text(json.dumps(_sweep(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        command,
        "kiwi_runtime_receipt",
        lambda: {
            "runtime": {"version": "0.23.2", "tree_sha256": "a" * 64},
            "model": {"version": "0.23.0", "tree_sha256": "b" * 64},
            "expected_download_bytes": 99490262,
        },
    )
    monkeypatch.setattr(
        command,
        "kiwi_space",
        lambda _: "한국어 띄어쓰기",
    )

    receipt = command.apply_korean_spacing(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert receipt["applied_candidates"] == 1
    assert receipt["rejected_candidates"] == 0
    assert receipt["source_sweep_sha256"] == command._sha256(source)
    assert receipt["runtime_receipt"]["model"]["tree_sha256"] == "b" * 64
    assert result["spans"][0]["candidates"][0]["text"] == "한국어 띄어쓰기"
    assert result["spans"][0]["candidates"][1]["text"] == "untouched"


def test_apply_korean_spacing_rejects_ambiguous_double_processing(
    tmp_path, monkeypatch
) -> None:
    document = _sweep()
    document["postprocessing"] = {"kind": "existing"}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="already contains postprocessing"):
        command.apply_korean_spacing(source, tmp_path / "output.json")
