from __future__ import annotations

import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from dubbing.control_plane import cli


def _retry_args(**overrides) -> Namespace:
    values = {
        "retry_backend": None,
        "retry_model": None,
        "retry_device": "auto",
        "retry_compute_type": "default",
        "retry_mlx_temperature": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_retry_backend_is_disabled_by_default() -> None:
    assert cli.build_retry_backend(_retry_args()) is None


def test_retry_backend_refuses_an_uncached_model(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_cached_hugging_face_snapshot", lambda _model: None)

    with pytest.raises(SystemExit, match="No download was performed"):
        cli.build_retry_backend(
            _retry_args(retry_backend="faster-whisper", retry_model="missing/model")
        )


def test_builds_offline_faster_whisper_retry_once(tmp_path) -> None:
    model = tmp_path / "faster-whisper"
    model.mkdir()

    backend = cli.build_retry_backend(
        _retry_args(
            retry_backend="faster-whisper",
            retry_model=str(model),
            retry_device="cuda",
            retry_compute_type="int8_float16",
        )
    )

    assert backend.model == str(model.resolve())
    assert backend.device == "cuda"
    assert backend.compute_type == "int8_float16"
    assert backend.local_files_only is True


def test_main_passes_persistent_retry_backend_and_binds_receipt(monkeypatch, tmp_path) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"fixture")
    output = tmp_path / "output"
    primary = SimpleNamespace(identity="whisperkit:primary")
    retry = SimpleNamespace(identity="mlx-whisper:independent")
    captured = {}

    class _Coordinator:
        def __init__(self, backend, checkpoint_dir, **kwargs):
            captured["primary"] = backend
            captured["retry"] = kwargs["retry_backend"]
            captured["checkpoint_dir"] = checkpoint_dir
            checkpoint_dir.mkdir(parents=True)

        def run(self, source, options):
            captured["source"] = source
            captured["options"] = options
            return SimpleNamespace(source_sha256="a" * 64, segments=()), {"status": "PASS"}

    monkeypatch.setattr(cli, "build_backend", lambda _args: primary)
    monkeypatch.setattr(cli, "build_retry_backend", lambda _args: retry)
    monkeypatch.setattr(cli, "AdaptiveLongFormCoordinator", _Coordinator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dubbing-long-transcribe",
            str(media),
            "--output",
            str(output),
            "--retry-backend",
            "mlx",
        ],
    )

    cli.main()

    receipt = json.loads((output / "run-receipt.json").read_text(encoding="utf-8"))
    assert captured["primary"] is primary
    assert captured["retry"] is retry
    assert receipt["backend_identity"] == primary.identity
    assert receipt["retry_backend_identity"] == retry.identity


def test_main_rejects_same_primary_and_retry_identity(monkeypatch, tmp_path) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"fixture")
    backend = SimpleNamespace(identity="mlx-whisper:same")
    monkeypatch.setattr(cli, "build_backend", lambda _args: backend)
    monkeypatch.setattr(cli, "build_retry_backend", lambda _args: backend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dubbing-long-transcribe",
            str(media),
            "--output",
            str(tmp_path / "output"),
            "--retry-backend",
            "mlx",
        ],
    )

    with pytest.raises(SystemExit, match="requires an independent"):
        cli.main()
