from __future__ import annotations

import hashlib
from collections.abc import Callable


def _non_whitespace(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _interpolated_character_boundaries(words: list[dict]) -> list[float]:
    boundaries: list[float] = []
    for word in words:
        characters = _non_whitespace(str(word.get("text", "")))
        if not characters:
            continue
        start = float(word["start_ms"])
        end = float(word["end_ms"])
        width = (end - start) / len(characters)
        if not boundaries:
            boundaries.append(start)
        elif start > boundaries[-1]:
            boundaries[-1] = start
        boundaries.extend(start + width * index for index in range(1, len(characters) + 1))
    return boundaries


def _retime_words(original_words: list[dict], normalized: str) -> list[dict]:
    tokens = normalized.split()
    boundaries = _interpolated_character_boundaries(original_words)
    if not tokens:
        return []
    if len(boundaries) != len(_non_whitespace(normalized)) + 1:
        raise ValueError("word timestamps do not cover the normalized character sequence")

    result = []
    offset = 0
    probabilities = [
        float(word["probability"])
        for word in original_words
        if isinstance(word.get("probability"), (int, float))
    ]
    probability = sum(probabilities) / len(probabilities) if probabilities else None
    for token in tokens:
        end_offset = offset + len(token)
        word = {
            "start_ms": round(boundaries[offset]),
            "end_ms": round(boundaries[end_offset]),
            "text": token,
        }
        if probability is not None:
            word["probability"] = probability
        result.append(word)
        offset = end_offset
    return result


def normalize_candidate_spacing(
    candidate: dict,
    normalizer: Callable[[str], str],
    *,
    provider: str,
) -> dict:
    original = str(candidate.get("text", ""))
    normalized = normalizer(original).strip()
    provenance = {
        "kind": "korean_spacing",
        "provider": provider,
        "input_sha256": _sha256_text(original),
        "output_sha256": _sha256_text(normalized),
        "non_whitespace_guard_passed": _non_whitespace(original)
        == _non_whitespace(normalized),
    }
    if not provenance["non_whitespace_guard_passed"]:
        return {
            **candidate,
            "spacing_normalization": {**provenance, "status": "REJECTED_CHARACTER_CHANGE"},
        }

    segments = []
    for segment in candidate.get("segments", []):
        segment_text = str(segment.get("text", ""))
        normalized_segment = normalizer(segment_text).strip()
        if _non_whitespace(segment_text) != _non_whitespace(normalized_segment):
            return {
                **candidate,
                "spacing_normalization": {
                    **provenance,
                    "status": "REJECTED_SEGMENT_CHARACTER_CHANGE",
                },
            }
        segments.append(
            {
                **segment,
                "text": normalized_segment,
                "words": _retime_words(segment.get("words", []), normalized_segment),
            }
        )
    return {
        **candidate,
        "text": normalized,
        "segments": segments,
        "spacing_normalization": {**provenance, "status": "APPLIED"},
    }


def kiwi_space(text: str) -> str:
    try:
        from kiwipiepy import Kiwi
    except ImportError as error:  # pragma: no cover - exercised only with optional runtime.
        raise RuntimeError("kiwipiepy is required for the Kiwi spacing provider") from error
    if not hasattr(kiwi_space, "_kiwi"):
        kiwi_space._kiwi = Kiwi()  # type: ignore[attr-defined]
    return kiwi_space._kiwi.space(text, reset_whitespace=True)  # type: ignore[attr-defined]
