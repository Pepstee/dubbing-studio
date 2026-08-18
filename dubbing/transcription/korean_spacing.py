from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


KIWIPIEPY_VERSION = "0.23.2"
KIWIPIEPY_MODEL_VERSION = "0.23.0"
KIWIPIEPY_GIGABYTE_ARCHIVE = {
    "filename": (
        "kiwipiepy-0.23.2-cp39-abi3-manylinux2014_x86_64."
        "manylinux_2_17_x86_64.whl"
    ),
    "bytes": 11513350,
    "sha256": "46a0a9fd36727736e8010ff54c655639f5df1c2ec34b92679cd3a94e8734d81f",
}
KIWIPIEPY_MODEL_ARCHIVE = {
    "filename": "kiwipiepy_model-0.23.0.tar.gz",
    "bytes": 87976912,
    "sha256": "498a22f5585e6c4a162423d7557eb3ee3f71cddc6e0aeb2650c50467e85933e2",
}


def _non_whitespace(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _distribution_tree_receipt(package: str, import_root: str) -> dict:
    try:
        installed = distribution(package)
    except PackageNotFoundError as error:
        raise RuntimeError(f"required local package is not installed: {package}") from error
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for relative in sorted(installed.files or [], key=str):
        relative_path = Path(str(relative))
        if not relative_path.parts or relative_path.parts[0] != import_root:
            continue
        path = Path(installed.locate_file(relative))
        if not path.is_file():
            continue
        content = path.read_bytes()
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        file_count += 1
        total_bytes += len(content)
    if not file_count:
        raise RuntimeError(f"installed package has no hashable files: {package}")
    return {
        "package": package,
        "version": installed.version,
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "installed_bytes": total_bytes,
    }


def kiwi_runtime_receipt() -> dict:
    runtime = _distribution_tree_receipt("kiwipiepy", "kiwipiepy")
    model = _distribution_tree_receipt("kiwipiepy_model", "kiwipiepy_model")
    if runtime["version"] != KIWIPIEPY_VERSION:
        raise RuntimeError(
            f"kiwipiepy version mismatch: {runtime['version']} != {KIWIPIEPY_VERSION}"
        )
    if model["version"] != KIWIPIEPY_MODEL_VERSION:
        raise RuntimeError(
            "kiwipiepy_model version mismatch: "
            f"{model['version']} != {KIWIPIEPY_MODEL_VERSION}"
        )
    return {
        "runtime": runtime,
        "model": model,
        "expected_downloads": [KIWIPIEPY_GIGABYTE_ARCHIVE, KIWIPIEPY_MODEL_ARCHIVE],
        "expected_download_bytes": (
            KIWIPIEPY_GIGABYTE_ARCHIVE["bytes"] + KIWIPIEPY_MODEL_ARCHIVE["bytes"]
        ),
    }


def _interpolated_character_evidence(
    words: list[dict],
) -> tuple[list[float], list[float | None]]:
    boundaries: list[float] = []
    character_probabilities: list[float | None] = []
    for word in words:
        characters = _non_whitespace(str(word.get("text", "")))
        if not characters:
            continue
        try:
            start = float(word["start_ms"])
            end = float(word["end_ms"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("word timestamp is missing or non-numeric") from error
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("word timestamp is malformed")
        if boundaries and start < boundaries[-1]:
            raise ValueError("word timestamps overlap or are unordered")
        width = (end - start) / len(characters)
        if not boundaries:
            boundaries.append(start)
        else:
            boundaries[-1] = start
        boundaries.extend(start + width * index for index in range(1, len(characters) + 1))
        raw_probability = word.get("probability")
        probability = (
            float(raw_probability)
            if isinstance(raw_probability, (int, float))
            and math.isfinite(float(raw_probability))
            else None
        )
        character_probabilities.extend([probability] * len(characters))
    return boundaries, character_probabilities


def _retime_words(original_words: list[dict], normalized: str) -> list[dict]:
    tokens = normalized.split()
    boundaries, character_probabilities = _interpolated_character_evidence(original_words)
    if not tokens:
        return []
    if len(boundaries) != len(_non_whitespace(normalized)) + 1:
        raise ValueError("word timestamps do not cover the normalized character sequence")

    result = []
    offset = 0
    for token in tokens:
        end_offset = offset + len(token)
        word = {
            "start_ms": round(boundaries[offset]),
            "end_ms": round(boundaries[end_offset]),
            "text": token,
        }
        token_probabilities = [
            value
            for value in character_probabilities[offset:end_offset]
            if value is not None
        ]
        if token_probabilities:
            word["probability"] = sum(token_probabilities) / len(token_probabilities)
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
    source_segments = candidate.get("segments", [])
    source_joined = " ".join(str(segment.get("text", "")).strip() for segment in source_segments)
    if _non_whitespace(original) != _non_whitespace(source_joined):
        return {
            **candidate,
            "spacing_normalization": {
                "kind": "korean_spacing",
                "provider": provider,
                "input_sha256": _sha256_text(original),
                "output_sha256": None,
                "non_whitespace_guard_passed": False,
                "status": "REJECTED_SOURCE_ALIGNMENT",
            },
        }

    normalized_segments: list[tuple[dict, str]] = []
    for segment in source_segments:
        segment_text = str(segment.get("text", ""))
        normalized_segment = normalizer(segment_text)
        if not isinstance(normalized_segment, str):
            raise TypeError("spacing provider must return text")
        normalized_segment = normalized_segment.strip()
        if _non_whitespace(segment_text) != _non_whitespace(normalized_segment):
            return {
                **candidate,
                "spacing_normalization": {
                    "kind": "korean_spacing",
                    "provider": provider,
                    "input_sha256": _sha256_text(original),
                    "output_sha256": _sha256_text(normalized_segment),
                    "non_whitespace_guard_passed": False,
                    "status": "REJECTED_CHARACTER_CHANGE",
                },
            }
        normalized_segments.append((segment, normalized_segment))
    normalized = " ".join(text for _, text in normalized_segments if text)
    provenance = {
        "kind": "korean_spacing",
        "provider": provider,
        "input_sha256": _sha256_text(original),
        "output_sha256": _sha256_text(normalized),
        "non_whitespace_guard_passed": _non_whitespace(original)
        == _non_whitespace(normalized),
    }
    segments = []
    for segment, normalized_segment in normalized_segments:
        try:
            words = _retime_words(segment.get("words", []), normalized_segment)
        except ValueError:
            return {
                **candidate,
                "spacing_normalization": {
                    **provenance,
                    "status": "REJECTED_TIMESTAMP_EVIDENCE",
                },
            }
        segments.append(
            {
                **segment,
                "text": normalized_segment,
                "words": words,
            }
        )
    status = "UNCHANGED" if normalized == original.strip() else "APPLIED"
    return {
        **candidate,
        "text": normalized,
        "segments": segments,
        "spacing_normalization": {**provenance, "status": status},
    }


def kiwi_space(text: str) -> str:
    try:
        from kiwipiepy import Kiwi
    except ImportError as error:  # pragma: no cover - exercised only with optional runtime.
        raise RuntimeError("kiwipiepy is required for the Kiwi spacing provider") from error
    if not hasattr(kiwi_space, "_kiwi"):
        kiwi_space._kiwi = Kiwi()  # type: ignore[attr-defined]
    return kiwi_space._kiwi.space(text, reset_whitespace=True)  # type: ignore[attr-defined]
