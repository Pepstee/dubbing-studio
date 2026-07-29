from __future__ import annotations

import json
from pathlib import Path

from dubbing.capture.service import CaptureService
from dubbing.diarization import SherpaOnnxDiarizationBackend
from dubbing.transcription import FasterWhisperTranscriptionBackend
from dubbing.translation import LinguaLanguageDetector, NLLBTranslationBackend


def load_config(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("landing_inbox", "workspace", "defaults"):
        if key not in document:
            raise ValueError(f"deployment config is missing {key}")
    return document


def build_service(config: dict) -> CaptureService:
    defaults = config["defaults"]
    diarizer = None
    if defaults.get("segmentation_model") and defaults.get("embedding_model"):
        diarizer = SherpaOnnxDiarizationBackend(
            defaults["segmentation_model"],
            defaults["embedding_model"],
            device=defaults.get("diarization_device", "cpu"),
        )
    return CaptureService(
        config["workspace"]["wsl_path"],
        FasterWhisperTranscriptionBackend(
            defaults.get("asr_model", "large-v3-turbo"),
            device=defaults.get("asr_device", "cuda"),
            compute_type=defaults.get("asr_compute_type", "float16"),
        ),
        diarizer=diarizer,
        language_detector=LinguaLanguageDetector(),
        translation_backend=NLLBTranslationBackend(
            defaults.get("translation_model", "facebook/nllb-200-distilled-600M"),
            device=defaults.get("translation_device", "cuda"),
        ),
        translation_target=defaults.get("translation_target", "en"),
    )
