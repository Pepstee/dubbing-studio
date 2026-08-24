from __future__ import annotations

import hashlib
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from dubbing.media import ffmpeg_executable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AudioCandidate:
    identifier: str
    path: Path
    source_sha256: str
    candidate_sha256: str
    processing: dict

    def to_dict(self) -> dict:
        return {
            "id": self.identifier,
            "source_sha256": self.source_sha256,
            "candidate_sha256": self.candidate_sha256,
            "processing": self.processing,
        }


@dataclass(frozen=True)
class AudioCandidateSet:
    candidates: tuple[AudioCandidate, ...]
    source_channels: int | None
    failures: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            "source_channels": self.source_channels,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "failures": list(self.failures),
        }


def _pcm_channels(source: Path) -> int | None:
    try:
        with wave.open(str(source), "rb") as audio:
            if audio.getcomptype() != "NONE":
                return None
            return audio.getnchannels()
    except (EOFError, wave.Error):
        return None


def build_audio_candidates(
    source: str | Path,
    output_dir: str | Path,
    *,
    policies: tuple[str, ...] = (
        "raw",
        "downmix",
        "channels",
    ),
    maximum_channels: int = 4,
) -> AudioCandidateSet:
    """Build bounded, deterministic retry candidates without changing the source."""

    source = Path(source)
    output_dir = Path(output_dir)
    allowed = {"raw", "downmix", "channels", "speech-band-normalized"}
    if not policies or policies[0] != "raw" or len(set(policies)) != len(policies):
        raise ValueError("audio candidate policies must start with unique raw")
    if set(policies) - allowed:
        raise ValueError("unsupported audio candidate policy")
    if maximum_channels < 1 or maximum_channels > 8:
        raise ValueError("maximum_channels must be between 1 and 8")
    if not source.is_file():
        raise FileNotFoundError(source)

    source_digest = _sha256(source)
    channels = _pcm_channels(source)
    raw = AudioCandidate(
        "raw",
        source,
        source_digest,
        source_digest,
        {"kind": "source-preserved", "filter": None},
    )
    candidates = [raw]
    failures: list[dict] = []
    if channels is None:
        return AudioCandidateSet(
            tuple(candidates),
            None,
            ({"id": "probe", "reason": "not-readable-pcm-wave"},),
        )

    ffmpeg = ffmpeg_executable()
    requested: list[tuple[str, str | None]] = []
    if "downmix" in policies and channels > 1:
        requested.append(("downmix", None))
    if "channels" in policies and channels > 1:
        requested.extend(
            (f"channel-{index}", f"pan=mono|c0=c{index}")
            for index in range(min(channels, maximum_channels))
        )
    if "speech-band-normalized" in policies:
        requested.append(
            (
                "speech-band-normalized",
                "highpass=f=80,lowpass=f=7600,dynaudnorm=f=150:g=6:p=0.9",
            )
        )
    if not ffmpeg:
        return AudioCandidateSet(
            tuple(candidates),
            channels,
            tuple({"id": identifier, "reason": "ffmpeg-unavailable"} for identifier, _ in requested),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for identifier, filter_graph in requested:
        destination = output_dir / f"{identifier}.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
        ]
        if filter_graph:
            command.extend(("-af", filter_graph))
        command.extend(("-ac", "1", "-c:a", "pcm_s16le", "-y", str(destination)))
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if process.returncode or not destination.is_file():
            failures.append(
                {
                    "id": identifier,
                    "reason": "ffmpeg-failed",
                    "detail": process.stderr.strip()[-500:],
                }
            )
            continue
        candidates.append(
            AudioCandidate(
                identifier,
                destination,
                source_digest,
                _sha256(destination),
                {
                    "kind": identifier,
                    "filter": filter_graph,
                    "output_channels": 1,
                },
            )
        )
    return AudioCandidateSet(tuple(candidates), channels, tuple(failures))
