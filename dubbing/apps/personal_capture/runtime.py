from __future__ import annotations

from dubbing.apps.personal_capture.service import CaptureService
from dubbing.apps.personal_capture.model_manifest import verify_model_manifest
from dubbing.diarization import PyannoteCommunityBackend, SherpaOnnxDiarizationBackend
from dubbing.transcription import (
    FasterWhisperSileroSpeechRegionDetector,
    FasterWhisperTranscriptionBackend,
    MLXWhisperTranscriptionBackend,
)
from dubbing.transcription.whisperkit import (
    WhisperKitServerProcess,
    WhisperKitTranscriptionBackend,
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


def build_transcription_backend(
    defaults: dict,
    model_manifest: dict,
    *,
    retry: bool = False,
):
    """Build one manifest-bound local ASR provider without changing the pipeline."""
    prefix = "asr_retry" if retry else "asr"
    backend = defaults[f"{prefix}_backend"]
    model = defaults[f"{prefix}_model"]
    manifest_name = "asr_retry" if retry else "asr"
    revision = model_manifest["models"][manifest_name]["revision"]
    if backend == "faster-whisper":
        return FasterWhisperTranscriptionBackend(
            model,
            device=defaults[f"{prefix}_device"],
            compute_type=defaults[f"{prefix}_compute_type"],
            local_files_only=True,
            model_revision=revision,
        )
    if backend == "mlx":
        return MLXWhisperTranscriptionBackend(
            model,
            temperature=tuple(defaults.get(f"{prefix}_temperatures", (0.0,))),
            condition_on_previous_text=False,
        )
    server = WhisperKitServerProcess(
        executable=defaults[f"{prefix}_executable"],
        model_path=model,
        port=int(defaults[f"{prefix}_server_port"]),
    )
    return WhisperKitTranscriptionBackend(
        model=defaults[f"{prefix}_model_name"],
        endpoint=server.endpoint,
        server=server,
    )


def build_service(config: dict) -> CaptureService:
    defaults = config["defaults"]
    model_manifest = verify_model_manifest(config)
    paths = config.get("paths", {})
    diarizer = None
    diarization_embedding_backend = None
    if defaults.get("segmentation_model") and defaults.get("embedding_model"):
        sherpa = SherpaOnnxDiarizationBackend(
            defaults["segmentation_model"],
            defaults["embedding_model"],
            device=defaults.get("diarization_device", "cpu"),
            cluster_threshold=defaults.get("diarization_cluster_threshold", 0.85),
        )
        diarization_embedding_backend = sherpa
        if defaults.get("diarization_backend", "sherpa-onnx") == "pyannote-community-1":
            diarizer = PyannoteCommunityBackend(
                defaults["pyannote_model"],
                device=defaults.get("diarization_device", "auto"),
            )
        else:
            diarizer = sherpa
    primary_asr = build_transcription_backend(defaults, model_manifest)
    retry_asr = build_transcription_backend(defaults, model_manifest, retry=True)
    if retry_asr.identity == primary_asr.identity:
        raise ValueError("production retry ASR must be independent from the primary ASR")
    translation_backend = None
    language_detector = None
    if defaults.get("translation_backend", "nllb") == "nllb":
        language_detector = LinguaLanguageDetector()
        translation_backend = NLLBTranslationBackend(
            defaults["translation_model"],
            device=defaults.get("translation_device", "cuda"),
            local_files_only=defaults.get("offline_models_required", False),
            model_revision=model_manifest["models"]["translation"]["revision"],
        )
    speech_region_detector = FasterWhisperSileroSpeechRegionDetector(
        strict_threshold=defaults["vad_strict_threshold"],
        sensitive_threshold=defaults["vad_sensitive_threshold"],
        minimum_speech_ms=defaults["vad_minimum_speech_ms"],
        minimum_silence_ms=defaults["vad_minimum_silence_ms"],
        speech_pad_ms=defaults["vad_speech_pad_ms"],
        maximum_region_seconds=defaults["vad_maximum_region_seconds"],
    )
    return CaptureService(
        config["workspace"]["wsl_path"],
        primary_asr,
        transcription_retry_backend=retry_asr,
        silence_verification_detector=(
            speech_region_detector
            if defaults.get("vad_silence_verification_enabled", True)
            else None
        ),
        targeted_retry_region_detector=(
            speech_region_detector
            if defaults.get("vad_targeted_retry_region_detection_enabled", True)
            else None
        ),
        diarizer=diarizer,
        diarization_embedding_backend=diarization_embedding_backend,
        language_detector=language_detector,
        translation_backend=translation_backend,
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
        audio_candidate_policies=tuple(
            defaults.get(
                "audio_candidate_policies",
                ("raw", "downmix", "channels"),
            )
        ),
        maximum_audio_candidate_channels=defaults.get(
            "maximum_audio_candidate_channels", 4
        ),
        diarization_chunk_seconds=defaults.get(
            "diarization_chunk_seconds", 2 * 60 * 60
        ),
        diarization_global_speaker_threshold=defaults.get(
            "diarization_global_speaker_threshold", 0.80
        ),
        diarization_global_speaker_margin=defaults.get(
            "diarization_global_speaker_margin", 0.05
        ),
        minimum_free_bytes=defaults.get("minimum_free_bytes", 0),
        maximum_audio_seconds=defaults.get("maximum_audio_seconds", 24 * 60 * 60),
        outbox_dir=config.get("giga_outbox", {}).get("path"),
        inbox_dir=config["landing_inbox"]["wsl_path"],
    )
