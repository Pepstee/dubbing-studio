from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

_SCHEMA = "dubbing.personal-capture-model-manifest.v2"
_REVISION = re.compile(r"^[a-f0-9]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(directory: Path, *, allow_cache_symlinks: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    if not directory.is_dir():
        raise ValueError(f"model directory not found: {directory}")
    files: dict[str, str] = {}
    symlinks: dict[str, str] = {}
    cache_root = directory.parent.parent if directory.parent.name == "snapshots" else None
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            if not allow_cache_symlinks or cache_root is None:
                raise ValueError(f"model directory must be self-contained; symlink found: {path}")
            resolved = path.resolve(strict=True)
            if cache_root != resolved and cache_root not in resolved.parents:
                raise ValueError(f"model symlink escapes its cache repository: {path}")
            if not resolved.is_file():
                raise ValueError(f"model symlink does not resolve to a file: {path}")
            symlinks[relative.as_posix()] = resolved.relative_to(cache_root).as_posix()
            files[relative.as_posix()] = _sha256(resolved)
        elif path.is_file():
            files[relative.as_posix()] = _sha256(path)
    if not files:
        raise ValueError(f"model directory is empty: {directory}")
    return files, symlinks


def _model_entry(
    directory: str | Path,
    revision: str,
    *,
    backend: str,
) -> dict:
    path = Path(directory).resolve()
    if not _REVISION.fullmatch(revision):
        raise ValueError("model revision must be an exact 40-character commit SHA")
    files, symlinks = _inventory(path, allow_cache_symlinks=backend == "mlx")
    entry = {
        "directory": str(path),
        "revision": revision,
        "backend": backend,
        "files": files,
    }
    if symlinks:
        entry["cache_symlinks"] = symlinks
    return entry


def _model_file_entry(model: str | Path) -> dict:
    path = Path(model).resolve()
    if not path.is_file():
        raise ValueError(f"model file not found: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def create_model_manifest(
    *,
    asr_directory: str | Path,
    asr_revision: str,
    translation_directory: str | Path | None,
    translation_revision: str | None,
    segmentation_model: str | Path,
    embedding_model: str | Path,
    output: str | Path,
    asr_retry_directory: str | Path | None = None,
    asr_retry_revision: str | None = None,
    asr_backend: str = "faster-whisper",
    asr_retry_backend: str = "faster-whisper",
    translation_backend: str = "nllb",
) -> Path:
    """Hash every self-contained model file and bind it to an exact revision."""
    document = {
        "schema_version": _SCHEMA,
        "models": {
            "asr": _model_entry(asr_directory, asr_revision, backend=asr_backend),
            "diarization_segmentation": _model_file_entry(segmentation_model),
            "diarization_embedding": _model_file_entry(embedding_model),
        },
    }
    if translation_backend == "nllb":
        if translation_directory is None or translation_revision is None:
            raise ValueError("NLLB translation directory and revision are required")
        document["models"]["translation"] = _model_entry(
            translation_directory,
            translation_revision,
            backend="nllb",
        )
    elif translation_backend != "none":
        raise ValueError("translation backend must be nllb or none")
    if (asr_retry_directory is None) != (asr_retry_revision is None):
        raise ValueError("retry model directory and revision must be provided together")
    if asr_retry_directory is not None and asr_retry_revision is not None:
        document["models"]["asr_retry"] = _model_entry(
            asr_retry_directory,
            asr_retry_revision,
            backend=asr_retry_backend,
        )
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def verify_model_manifest(config: dict) -> dict:
    """Fail closed unless configured model directories exactly match their manifest."""
    path = Path(config["defaults"]["model_manifest"]).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read model manifest: {path}") from exc
    if document.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported Personal Capture model manifest")
    models = document.get("models")
    if not isinstance(models, dict):
        raise ValueError("model manifest is missing its models object")
    defaults = config["defaults"]
    expected_directories = {
        "asr": Path(defaults["asr_model"]).resolve(),
        "asr_retry": Path(defaults["asr_retry_model"]).resolve(),
    }
    if defaults.get("translation_backend", "nllb") == "nllb":
        expected_directories["translation"] = Path(defaults["translation_model"]).resolve()
    for name, expected_directory in expected_directories.items():
        entry = models.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"model manifest is missing {name}")
        if Path(entry.get("directory", "")).resolve() != expected_directory:
            raise ValueError(f"{name} manifest directory does not match deployment config")
        revision = entry.get("revision")
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise ValueError(f"{name} manifest revision is not an exact commit SHA")
        expected_files = entry.get("files")
        if not isinstance(expected_files, dict) or not expected_files:
            raise ValueError(f"{name} manifest has no file hashes")
        expected_backend = (
            defaults["asr_backend"]
            if name == "asr"
            else defaults["asr_retry_backend"]
            if name == "asr_retry"
            else "nllb"
        )
        if entry.get("backend", expected_backend) != expected_backend:
            raise ValueError(f"{name} manifest backend does not match deployment config")
        actual_files, actual_symlinks = _inventory(
            expected_directory,
            allow_cache_symlinks=expected_backend == "mlx",
        )
        if actual_files != expected_files:
            raise ValueError(f"{name} model files do not match the approved manifest")
        if entry.get("cache_symlinks", {}) != actual_symlinks:
            raise ValueError(f"{name} model cache symlinks do not match the approved manifest")
    expected_model_files = {
        "diarization_segmentation": Path(
            config["defaults"]["segmentation_model"]
        ).resolve(),
        "diarization_embedding": Path(config["defaults"]["embedding_model"]).resolve(),
    }
    for name, expected_path in expected_model_files.items():
        entry = models.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"model manifest is missing {name}")
        if Path(entry.get("path", "")).resolve() != expected_path:
            raise ValueError(f"{name} manifest path does not match deployment config")
        expected_size = entry.get("size_bytes")
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"{name} manifest size is invalid")
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[a-f0-9]{64}", expected_sha256
        ):
            raise ValueError(f"{name} manifest SHA-256 is invalid")
        actual = _model_file_entry(expected_path)
        if actual != entry:
            raise ValueError(f"{name} model file does not match the approved manifest")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an exact offline model revision-and-hash manifest"
    )
    parser.add_argument("--asr-directory", required=True)
    parser.add_argument("--asr-revision", required=True)
    parser.add_argument(
        "--asr-backend",
        choices=("faster-whisper", "mlx", "whisperkit"),
        default="faster-whisper",
    )
    parser.add_argument("--asr-retry-directory", required=True)
    parser.add_argument("--asr-retry-revision", required=True)
    parser.add_argument(
        "--asr-retry-backend",
        choices=("faster-whisper", "mlx", "whisperkit"),
        default="faster-whisper",
    )
    parser.add_argument("--translation-directory")
    parser.add_argument("--translation-revision")
    parser.add_argument(
        "--translation-backend", choices=("nllb", "none"), default="nllb"
    )
    parser.add_argument("--segmentation-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = create_model_manifest(
        asr_directory=args.asr_directory,
        asr_revision=args.asr_revision,
        asr_backend=args.asr_backend,
        asr_retry_directory=args.asr_retry_directory,
        asr_retry_revision=args.asr_retry_revision,
        asr_retry_backend=args.asr_retry_backend,
        translation_directory=args.translation_directory,
        translation_revision=args.translation_revision,
        translation_backend=args.translation_backend,
        segmentation_model=args.segmentation_model,
        embedding_model=args.embedding_model,
        output=args.output,
    )
    print(destination)


if __name__ == "__main__":
    main()
