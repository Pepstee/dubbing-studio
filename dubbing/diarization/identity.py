from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


UNKNOWN = "UNKNOWN"


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("voiceprint embeddings must have the same non-zero dimension")
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


@dataclass(frozen=True)
class VoiceReference:
    clip_sha256: str
    embedding: tuple[float, ...]
    consent_record: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "clip_sha256": self.clip_sha256,
            "embedding": list(self.embedding),
            "consent_record": self.consent_record,
            "created_at": self.created_at,
        }


class VoiceprintRegistry:
    """Consent-bound reference registry; diarization labels remain separate inputs."""

    def __init__(self, path: str | Path, *, threshold: float = 0.78, margin: float = 0.05) -> None:
        if not 0 < threshold <= 1 or not 0 <= margin < 1:
            raise ValueError("invalid voiceprint thresholds")
        self.path = Path(path)
        self.threshold = threshold
        self.margin = margin

    def _load(self) -> dict:
        if not self.path.exists():
            return {
                "schema_version": "dubbing.voiceprint-registry.v1",
                "people": {},
                "corrections": [],
            }
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if document.get("schema_version") != "dubbing.voiceprint-registry.v1":
            raise ValueError("unsupported voiceprint registry schema")
        return document

    def _save(self, document: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def add_reference(
        self,
        person_id: str,
        clip: str | Path,
        embedding: tuple[float, ...],
        *,
        consent_record: str,
    ) -> None:
        if not person_id.strip() or not consent_record.strip():
            raise ValueError("person_id and consent_record are required")
        digest = hashlib.sha256(Path(clip).read_bytes()).hexdigest()
        reference = VoiceReference(
            digest,
            embedding,
            consent_record,
            datetime.now(timezone.utc).isoformat(),
        )
        document = self._load()
        references = document["people"].setdefault(person_id, {"references": []})["references"]
        if not any(item["clip_sha256"] == digest for item in references):
            references.append(reference.to_dict())
        self._save(document)

    def match(self, embedding: tuple[float, ...]) -> dict:
        scores = {}
        for person_id, record in self._load()["people"].items():
            references = record.get("references", [])
            if references:
                scores[person_id] = max(
                    _cosine(embedding, tuple(item["embedding"])) for item in references
                )
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            return {"identity": UNKNOWN, "score": None, "reason": "no_references"}
        best_id, best_score = ordered[0]
        next_score = ordered[1][1] if len(ordered) > 1 else -1.0
        if best_score < self.threshold or best_score - next_score < self.margin:
            return {
                "identity": UNKNOWN,
                "score": best_score,
                "reason": "below_threshold_or_margin",
                "candidates": ordered[:3],
            }
        return {"identity": best_id, "score": best_score, "reason": "calibrated_match"}

    def record_manual_correction(
        self,
        *,
        recording_sha256: str,
        start_ms: int,
        end_ms: int,
        previous_identity: str,
        corrected_identity: str,
        operator: str,
    ) -> None:
        document = self._load()
        document["corrections"].append(
            {
                "recording_sha256": recording_sha256,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "previous_identity": previous_identity,
                "corrected_identity": corrected_identity,
                "operator": operator,
                "corrected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save(document)
