from __future__ import annotations

import argparse
import os

from dubbing.backends.base import TTSBackend
from dubbing.diarization import SherpaOnnxDiarizationBackend, SpeakerConstraints
from dubbing.transcription import (
    DEFAULT_FASTER_WHISPER_MODEL,
    DEFAULT_MLX_MODEL,
    FasterWhisperTranscriptionBackend,
    MLXWhisperTranscriptionBackend,
)


def make_tts_backend(name: str) -> TTSBackend:
    if name in ("auto", ""):
        from dubbing.backends import select_backend

        return select_backend()
    if name == "say":
        from dubbing.backends.say import SayTTSBackend

        return SayTTSBackend()
    if name == "piper":
        from dubbing.backends.piper import PiperTTSBackend

        return PiperTTSBackend()
    if name == "espeak":
        from dubbing.backends.espeak import EspeakTTSBackend

        return EspeakTTSBackend()
    raise SystemExit(f"Unknown backend: {name!r}. Available: auto, say, piper, espeak")


def make_diarizer(args: argparse.Namespace) -> SherpaOnnxDiarizationBackend:
    segmentation_model = args.segmentation_model or os.environ.get(
        "DUBBING_DIARIZATION_SEGMENTATION_MODEL"
    )
    embedding_model = args.embedding_model or os.environ.get(
        "DUBBING_DIARIZATION_EMBEDDING_MODEL"
    )
    if not segmentation_model or not embedding_model:
        raise ValueError(
            "Sherpa model paths are required. Pass --segmentation-model and "
            "--embedding-model, or set DUBBING_DIARIZATION_SEGMENTATION_MODEL "
            "and DUBBING_DIARIZATION_EMBEDDING_MODEL."
        )
    return SherpaOnnxDiarizationBackend(
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        device=args.diarization_device,
        num_threads=args.diarization_threads,
        cluster_threshold=args.cluster_threshold,
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
    )


def speaker_constraints(args: argparse.Namespace) -> SpeakerConstraints:
    return SpeakerConstraints(
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )


def make_transcriber(args: argparse.Namespace):
    if args.asr_backend == "mlx-whisper":
        return MLXWhisperTranscriptionBackend(
            model=args.asr_model or DEFAULT_MLX_MODEL,
            temperature=args.asr_temperature,
        )
    if args.asr_backend == "faster-whisper":
        return FasterWhisperTranscriptionBackend(
            model=args.asr_model or DEFAULT_FASTER_WHISPER_MODEL,
            device=args.asr_device,
            compute_type=args.asr_compute_type,
            temperature=args.asr_temperature,
            cpu_threads=args.asr_cpu_threads,
            num_workers=args.asr_workers,
        )
    raise ValueError(
        f"Unknown ASR backend: {args.asr_backend!r}. "
        "Available: mlx-whisper, faster-whisper"
    )


def speaker_aliases(values: list[str]) -> dict[str, str]:
    aliases = {}
    for value in values:
        key, separator, name = value.partition("=")
        if not separator or not key.strip() or not name.strip():
            raise ValueError("--speaker must use LABEL=Name")
        aliases[key.strip()] = name.strip()
    return aliases
