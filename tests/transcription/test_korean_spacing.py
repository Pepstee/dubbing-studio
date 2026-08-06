from dubbing.transcription.korean_spacing import normalize_candidate_spacing


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


def test_normalization_rejects_any_non_whitespace_change() -> None:
    original = _candidate()
    result = normalize_candidate_spacing(
        original, lambda _: "한국어 띄어쓰기요", provider="test"
    )

    assert result["text"] == original["text"]
    assert result["spacing_normalization"]["status"] == "REJECTED_CHARACTER_CHANGE"
