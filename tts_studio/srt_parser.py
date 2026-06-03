from __future__ import annotations

import re
from pathlib import Path

from tts_studio.models import ProsodicHint, Segment

_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_INLINE_TAG_RE = re.compile(r"<(emotion|rate|pitch):([^>]+)>", re.IGNORECASE)


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def _extract_hints(text: str) -> tuple[str, list[ProsodicHint]]:
    hints: list[ProsodicHint] = []

    def _sub(match: re.Match) -> str:
        hints.append(ProsodicHint(tag=match.group(1).lower(), value=match.group(2)))
        return ""

    clean = _INLINE_TAG_RE.sub(_sub, text).strip()
    return clean, hints


def parse_text(raw: str, default_language: str = "en") -> list[Segment]:
    segments: list[Segment] = []
    blocks = re.split(r"\n{2,}", raw.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        ts_match = _TIMESTAMP_RE.match(lines[1].strip())
        if not ts_match:
            continue

        start_ms = _ts_to_ms(*ts_match.group(1, 2, 3, 4))
        end_ms = _ts_to_ms(*ts_match.group(5, 6, 7, 8))

        raw_text = " ".join(lines[2:])
        text, hints = _extract_hints(raw_text)

        segments.append(
            Segment(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                language=default_language,
                hints=hints,
            )
        )

    return segments


def parse_srt(path: str | Path, default_language: str = "en") -> list[Segment]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    return parse_text(raw, default_language=default_language)
