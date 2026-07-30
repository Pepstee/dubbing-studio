from __future__ import annotations

import importlib
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from dubbing.diarization.base import DiarizationBackend
from dubbing.diarization.models import (
    DiarizationError,
    DiarizationResult,
    SpeakerConstraints,
    SpeakerTurn,
    UnsupportedSpeakerConstraintError,
)
from dubbing.media import ffmpeg_executable

_MODEL_NAME = "pyannote-segmentation-3.0+nemo-titanet-small"
_TARGET_SAMPLE_RATE = 16_000
_MAX_AUDIO_SECONDS = 4 * 60 * 60


class SherpaOnnxDiarizationBackend(DiarizationBackend):
    """Offline local diarization using public Sherpa-ONNX models."""

    def __init__(
        self,
        segmentation_model: str | Path,
        embedding_model: str | Path,
        *,
        device: str = "cpu",
        num_threads: int = 2,
        cluster_threshold: float = 0.5,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        if not 0.0 < cluster_threshold <= 1.0:
            raise ValueError("cluster_threshold must be in (0, 1]")
        if min_duration_on < 0 or min_duration_off < 0:
            raise ValueError("minimum durations cannot be negative")
        self.segmentation_model = Path(segmentation_model)
        self.embedding_model = Path(embedding_model)
        self.device = device
        self.num_threads = num_threads
        self.cluster_threshold = cluster_threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

    @property
    def identity(self) -> str:
        return (
            f"sherpa-onnx:{_MODEL_NAME}:{self.device}:{self.num_threads}:"
            f"{self.cluster_threshold}:{self.min_duration_on}:{self.min_duration_off}"
        )

    def _dependencies(self):
        try:
            sherpa_onnx = importlib.import_module("sherpa_onnx")
            numpy = importlib.import_module("numpy")
        except ImportError as exc:
            raise DiarizationError(
                "Sherpa-ONNX diarization dependencies are missing. "
                "Install with `pip install 'dubbing-studio[diarization]'`."
            ) from exc

        if self.device == "cuda":
            provider_library = (
                Path(sherpa_onnx.__file__).parent
                / "lib"
                / "libonnxruntime_providers_cuda.so"
            )
            if not provider_library.is_file():
                raise DiarizationError(
                    "CUDA was requested but the installed sherpa-onnx wheel is CPU-only. "
                    "Install Sherpa's CUDA wheel and its documented CUDA runtime libraries; "
                    "or select --diarization-device cpu explicitly."
                )
        return sherpa_onnx, numpy

    def _validate_models(self) -> None:
        for kind, path in (
            ("segmentation", self.segmentation_model),
            ("speaker embedding", self.embedding_model),
        ):
            if not path.is_file():
                raise DiarizationError(
                    f"{kind} model not found at {path}. Download the public "
                    "Sherpa-ONNX speaker diarization models or pass the correct path."
                )

    @staticmethod
    def _validate_constraints(constraints: SpeakerConstraints) -> int:
        if constraints.num_speakers is None and (
            constraints.min_speakers is not None or constraints.max_speakers is not None
        ):
            raise UnsupportedSpeakerConstraintError(
                "Sherpa-ONNX supports an exact known speaker count or automatic "
                "threshold clustering; it cannot guarantee min/max-only constraints."
            )
        return constraints.num_speakers if constraints.num_speakers is not None else -1

    @staticmethod
    def _media_duration(path: Path) -> float | None:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return None
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return None
        try:
            return float(proc.stdout.strip())
        except ValueError:
            return None

    @staticmethod
    def _decode_media(path: Path, output: Path) -> None:
        ffmpeg = ffmpeg_executable()
        if ffmpeg is None:
            raise DiarizationError(
                "ffmpeg is required to ingest audio/video for diarization; "
                "install ffmpeg and retry."
            )
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(path),
                    "-t",
                    str(_MAX_AUDIO_SECONDS + 1),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(_TARGET_SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    "-y",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "unknown decoder error"
            raise DiarizationError(f"ffmpeg could not decode {path.name}: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise DiarizationError(f"ffmpeg timed out while decoding {path.name}") from exc

    def _load_audio(self, path: Path, numpy):
        duration = self._media_duration(path)
        if duration is not None and duration > _MAX_AUDIO_SECONDS:
            raise DiarizationError(
                f"audio duration {duration:.1f}s exceeds the {_MAX_AUDIO_SECONDS}s safety limit"
            )

        # Avoid an unnecessary ffmpeg dependency for the model's native WAV
        # format. This is also the least lossy path for production PCM inputs.
        try:
            with wave.open(str(path), "rb") as wav:
                native = (
                    wav.getnchannels() == 1
                    and wav.getsampwidth() == 2
                    and wav.getframerate() == _TARGET_SAMPLE_RATE
                    and wav.getcomptype() == "NONE"
                )
                if native:
                    frames = wav.getnframes()
                    if frames > _MAX_AUDIO_SECONDS * _TARGET_SAMPLE_RATE:
                        raise DiarizationError(
                            f"decoded audio exceeds the {_MAX_AUDIO_SECONDS}s safety limit"
                        )
                    raw = wav.readframes(frames)
                    return (
                        numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32768.0
                    )
        except (wave.Error, EOFError):
            pass

        with tempfile.TemporaryDirectory(prefix="dubbing-diarization-") as directory:
            decoded = Path(directory) / "audio.wav"
            self._decode_media(path, decoded)
            with wave.open(str(decoded), "rb") as wav:
                if (
                    wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or wav.getframerate() != _TARGET_SAMPLE_RATE
                ):
                    raise DiarizationError(
                        "internal media conversion did not produce 16 kHz mono 16-bit PCM"
                    )
                frames = wav.getnframes()
                if frames > _MAX_AUDIO_SECONDS * _TARGET_SAMPLE_RATE:
                    raise DiarizationError(
                        f"decoded audio exceeds the {_MAX_AUDIO_SECONDS}s safety limit"
                    )
                raw = wav.readframes(frames)
        return numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32768.0

    def diarize(
        self,
        audio: str | Path,
        constraints: SpeakerConstraints | None = None,
    ) -> DiarizationResult:
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise DiarizationError(f"audio input not found: {audio_path}")
        constraints = constraints or SpeakerConstraints()
        num_clusters = self._validate_constraints(constraints)
        self._validate_models()
        sherpa_onnx, numpy = self._dependencies()

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.segmentation_model)
                ),
                num_threads=self.num_threads,
                provider=self.device,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.embedding_model),
                num_threads=self.num_threads,
                provider=self.device,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters,
                threshold=self.cluster_threshold,
            ),
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )
        if not config.validate():
            raise DiarizationError(
                "Sherpa-ONNX rejected the diarization configuration; check model files."
            )

        samples = self._load_audio(audio_path, numpy)
        try:
            diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
            raw_turns = diarizer.process(samples).sort_by_start_time()
        except Exception as exc:
            hint = (
                " Check that CUDA/CUDNN runtime libraries are discoverable via "
                "LD_LIBRARY_PATH."
                if self.device == "cuda"
                else ""
            )
            raise DiarizationError(f"Sherpa-ONNX diarization failed: {exc}.{hint}") from exc

        label_map: dict[int, str] = {}
        turns: list[SpeakerTurn] = []
        for raw in raw_turns:
            raw_speaker = int(raw.speaker)
            label = label_map.setdefault(
                raw_speaker,
                f"SPEAKER_{len(label_map):02d}",
            )
            start_ms = max(0, round(float(raw.start) * 1000))
            end_ms = round(float(raw.end) * 1000)
            if end_ms <= start_ms:
                continue
            turns.append(
                SpeakerTurn(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker=label,
                    confidence=None,
                )
            )

        return DiarizationResult(
            turns=tuple(
                sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker))
            ),
            backend="sherpa-onnx",
            model=_MODEL_NAME,
            device=self.device,
            confidence_available=False,
        )
