from __future__ import annotations

import re

from dubbing.models import ProsodyTag

_TAG_RE = re.compile(r"<(\w+):(\w+)>")


def parse_prosody(text: str) -> tuple[str, list[ProsodyTag]]:
    tags: list[ProsodyTag] = []
    for m in _TAG_RE.finditer(text):
        tags.append(ProsodyTag(name=m.group(1), value=m.group(2)))
    clean = _TAG_RE.sub("", text)
    return clean, tags
