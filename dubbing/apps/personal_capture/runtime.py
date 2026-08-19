from __future__ import annotations

from dubbing.apps.personal_capture.service import CaptureService
from dubbing.apps.personal_capture.model_manifest import verify_model_manifest
from dubbing.diarization import SherpaOnnxDiarizationBackend
from dubbing.transcription import (
    FasterWhisperSileroSpeechRegionDetector,
    FasterWhisperTranscriptionBackend,
)
from dubbing.translation import LinguaLanguageDetector, NLLBTranslationBackend


def build_control_service(config: dict) -> CaptureService:
    """Build the ledger/review/outbox surface without loading any ML runtime."""
    paths = config.get("paths", {})
    return CaptureService(
        config["workspace"]["wsl_path"],
        packages_dir=paths.get("packages", "outputs/packages"),
        state_dir=paths.get("state", "state"),
        processing_dir=paths.get("processing", "processing"),
        outbox_dir=config.get("giga_outbox", {}).get("path"),
        inbox_dir=config["landing_inbox"]["wsl_path"],
    )


def build_service(config: dict) -> CaptureService:
    defaults = config["defaults"]
    model_manifest = verify_model_manifest(config)
    paths = config.get("paths", {})
    diarizer = None
    if defaults.get("segmentation_model") and defaults.get("embedding_model"):
        diarizer = SherpaOnnxDiarizationBackend(
            defaults["segmentation_model"],
            defaults["embedding_model"],
            device=defaults.get("diarization_device", "cpu"),
        )
    primary_asr = FasterWhisperTranscriptionBackend(
        defaults["asr_model"],
        device=defaults.get("asr_device", "cuda"),
        compute_type=defaults.get("asr_compute_type", "float16"),
        local_files_only=defaults.get("offline_models_required", False),
        model_revision=model_manifest["models"]["asr"]["revision"],
    )
    retry_asr = FasterWhisperTranscriptionBackend(
        defaults["asr_retry_model"],
        device=defaults.get("asr_retry_device", "cuda"),
        compute_type=defaults.get("asr_retry_compute_type", "int8_float16"),
        local_files_only=defaults.get("offline_models_required", False),
        model_revision=model_manifest["models"]["asr_retry"]["revision"],
    )
    if retry_asr.identity == primary_asr.identity:
        raise ValueError("production retry ASR must be independent from the primary ASR")
    return CaptureService(
        config["workspace"]["wsl_path"],
        primary_asr,
        transcription_retry_backend=retry_asr,
        speech_region_detector=FasterWhisperSileroSpeechRegionDetector(
            strict_threshold=defaults["vad_strict_threshold"],
            sensitive_threshold=defaults["vad_sensitive_threshold"],
            minimum_speech_ms=defaults["vad_minimum_speech_ms"],
            minimum_silence_ms=defaults["vad_minimum_silence_ms"],
            speech_pad_ms=defaults["vad_speech_pad_ms"],
            maximum_region_seconds=defaults["vad_maximum_region_seconds"],
        ),
        diarizer=diarizer,
        language_detector=LinguaLanguageDetector(),
        translation_backend=NLLBTranslationBackend(
            defaults["translation_model"],
            device=defaults.get("translation_device", "cuda"),
            local_files_only=defaults.get("offline_models_required", False),
            model_revision=model_manifest["models"]["translation"]["revision"],
        ),
        translation_target=defaults.get("translation_target", "en"),
        packages_dir=paths.get("packages", "outputs/packages"),
        state_dir=paths.get("state", "state"),
        processing_dir=paths.get("processing", "processing"),
        resumable=defaults.get("resumable", True),
        transcription_strategy=defaults.get("transcription_strategy", "adaptive"),
        transcription_chunk_seconds=defaults.get(
            "transcription_chunk_seconds", 4 * 60
        ),
        transcription_minimum_chunk_seconds=defaults.get(
            "transcription_minimum_chunk_seconds", 60
        ),
        transcription_maximum_chunk_seconds=defaults.get(
            "transcription_maximum_chunk_seconds", 8 * 60
        ),
        transcription_overlap_seconds=defaults.get(
            "transcription_overlap_seconds", 2
        ),
        transcription_minimum_silence_seconds=defaults.get(
            "transcription_minimum_silence_seconds", 0.7
        ),
        diarization_chunk_seconds=defaults.get(
            "diarization_chunk_seconds", 2 * 60 * 60
        ),
        diarization_global_speaker_threshold=defaults.get(
            "diarization_global_speaker_threshold", 0.65
        ),
        diarization_global_speaker_margin=defaults.get(
            "diarization_global_speaker_margin", 0.05
        ),
        minimum_free_bytes=defaults.get("minimum_free_bytes", 0),
        maximum_audio_seconds=defaults.get("maximum_audio_seconds", 24 * 60 * 60),
        outbox_dir=config.get("giga_outbox", {}).get("path"),
        inbox_dir=config["landing_inbox"]["wsl_path"],
    )
