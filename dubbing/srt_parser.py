from __future__ import annotations

import re
from pathlib import Path

from dubbing.models import SRTEntry

_TIMECODE_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def _ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def parse_srt_string(text: str) -> list[SRTEntry]:
    entries: list[SRTEntry] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        m = _TIMECODE_RE.match(lines[1].strip())
        if not m:
            continue
        start_ms = _ms(*m.group(1, 2, 3, 4))
        end_ms = _ms(*m.group(5, 6, 7, 8))
        subtitle_text = "\n".join(line for line in lines[2:])
        entries.append(SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=subtitle_text))
    return entries


def parse_srt(path: str | Path) -> list[SRTEntry]:
    return parse_srt_string(Path(path).read_text(encoding="utf-8"))
