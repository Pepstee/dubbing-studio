from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def character_tokens(text: str) -> list[str]:
    return [character for character in normalize_text(text) if not character.isspace()]


def levenshtein_distance(reference: Sequence[str], candidate: Sequence[str]) -> int:
    """Exact insertion/deletion/substitution edit distance with bounded memory."""
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, candidate_item in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != candidate_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[str], candidate: Sequence[str]) -> dict:
    edits = levenshtein_distance(reference, candidate)
    return {
        "reference_units": len(reference),
        "candidate_units": len(candidate),
        "edits": edits,
        "rate": edits / len(reference) if reference else (0.0 if not candidate else 1.0),
    }


def token_agreement(left: str, right: str) -> float:
    """Symmetric token agreement where 1 is identical and 0 is total disagreement."""
    left_tokens = word_tokens(left)
    right_tokens = word_tokens(right)
    denominator = max(len(left_tokens), len(right_tokens))
    if denominator == 0:
        return 1.0
    edits = levenshtein_distance(left_tokens, right_tokens)
    return max(0.0, 1.0 - edits / denominator)


def script_of_token(token: str) -> str:
    counts = defaultdict(int)
    for character in token:
        point = ord(character)
        if 0x0400 <= point <= 0x052F:
            counts["cyrillic"] += 1
        elif 0xAC00 <= point <= 0xD7AF or 0x1100 <= point <= 0x11FF:
            counts["hangul"] += 1
        elif "LATIN" in unicodedata.name(character, ""):
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return max(counts, key=counts.get, default="other")


def _by_script(tokens: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for token in tokens:
        grouped[script_of_token(token)].append(token)
    return dict(grouped)


@dataclass(frozen=True)
class TimedText:
    start_ms: int
    end_ms: int
    text: str
    language: str | None = None
    speaker: str | None = None


def _window_rows(
    reference_segments: Sequence[TimedText],
    candidate_segments: Sequence[TimedText],
    duration_ms: int,
    window_ms: int,
) -> list[dict]:
    rows = []
    for start_ms in range(0, duration_ms, window_ms):
        end_ms = min(duration_ms, start_ms + window_ms)
        reference = " ".join(
            item.text
            for item in reference_segments
            if start_ms <= item.start_ms + (item.end_ms - item.start_ms) // 2 < end_ms
        )
        candidate = " ".join(
            item.text
            for item in candidate_segments
            if start_ms <= item.start_ms + (item.end_ms - item.start_ms) // 2 < end_ms
        )
        rows.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "wer": error_rate(word_tokens(reference), word_tokens(candidate)),
                "cer": error_rate(character_tokens(reference), character_tokens(candidate)),
            }
        )
    return rows


def evaluate_documents(
    reference_text: str,
    candidate_text: str,
    *,
    reference_segments: Sequence[TimedText] = (),
    candidate_segments: Sequence[TimedText] = (),
    duration_ms: int | None = None,
    window_ms: int = 5 * 60 * 1000,
) -> dict:
    reference_words = word_tokens(reference_text)
    candidate_words = word_tokens(candidate_text)
    reference_characters = character_tokens(reference_text)
    candidate_characters = character_tokens(candidate_text)
    reference_scripts = _by_script(reference_words)
    candidate_scripts = _by_script(candidate_words)
    scripts = sorted(set(reference_scripts) | set(candidate_scripts))
    windows_available = bool(reference_segments and candidate_segments and duration_ms)
    return {
        "schema_version": "dubbing.transcript-evaluation.v1",
        "wer": error_rate(reference_words, candidate_words),
        "cer": error_rate(reference_characters, candidate_characters),
        "per_script": {
            script: {
                "wer": error_rate(
                    reference_scripts.get(script, []), candidate_scripts.get(script, [])
                )
            }
            for script in scripts
        },
        "per_time_window": (
            _window_rows(
                reference_segments,
                candidate_segments,
                int(duration_ms),
                window_ms,
            )
            if windows_available
            else []
        ),
        "time_window_evaluation_available": windows_available,
        "time_window_limitation": (
            None
            if windows_available
            else "A timestamped reference is required for defensible window-level WER/CER."
        ),
        "language_note": (
            "Script buckets are reported for this untimed mixed-language reference; "
            "language-labelled reference turns are required for true per-language WER."
        ),
    }
