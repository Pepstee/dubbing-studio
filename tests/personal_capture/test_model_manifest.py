from __future__ import annotations

import json

import pytest

from dubbing.apps.personal_capture.model_manifest import (
    create_model_manifest,
    verify_model_manifest,
)
from tests.personal_capture.test_config import deployment_config


def _model(directory, files):
    directory.mkdir()
    for name, content in files.items():
        (directory / name).write_bytes(content)
    return directory


def test_model_manifest_binds_exact_revisions_and_file_hashes(tmp_path):
    config = deployment_config(tmp_path)
    asr = _model(
        tmp_path / "asr-model",
        {"config.json": b"{}", "model.bin": b"asr", "tokenizer.json": b"{}"},
    )
    translation = _model(
        tmp_path / "translation-model",
        {
            "config.json": b"{}",
            "model.safetensors": b"translation",
            "sentencepiece.bpe.model": b"tokenizer",
        },
    )
    output = create_model_manifest(
        asr_directory=asr,
        asr_revision="a" * 40,
        translation_directory=translation,
        translation_revision="b" * 40,
        output=config["defaults"]["model_manifest"],
    )

    document = verify_model_manifest(config)

    assert output.stat().st_mode & 0o077 == 0
    assert document["models"]["asr"]["revision"] == "a" * 40
    assert "model.bin" in document["models"]["asr"]["files"]


def test_model_manifest_rejects_changed_weights(tmp_path):
    config = deployment_config(tmp_path)
    asr = _model(tmp_path / "asr-model", {"model.bin": b"approved"})
    translation = _model(
        tmp_path / "translation-model",
        {"model.safetensors": b"approved"},
    )
    create_model_manifest(
        asr_directory=asr,
        asr_revision="a" * 40,
        translation_directory=translation,
        translation_revision="b" * 40,
        output=config["defaults"]["model_manifest"],
    )
    (asr / "model.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="do not match"):
        verify_model_manifest(config)


def test_model_manifest_rejects_moving_revision_labels(tmp_path):
    asr = _model(tmp_path / "asr-model", {"model.bin": b"approved"})
    translation = _model(
        tmp_path / "translation-model",
        {"model.safetensors": b"approved"},
    )

    with pytest.raises(ValueError, match="exact 40-character"):
        create_model_manifest(
            asr_directory=asr,
            asr_revision="main",
            translation_directory=translation,
            translation_revision="b" * 40,
            output=tmp_path / "manifest.json",
        )


def test_model_manifest_rejects_unapproved_extra_file(tmp_path):
    config = deployment_config(tmp_path)
    asr = _model(tmp_path / "asr-model", {"model.bin": b"approved"})
    translation = _model(
        tmp_path / "translation-model",
        {"model.safetensors": b"approved"},
    )
    create_model_manifest(
        asr_directory=asr,
        asr_revision="a" * 40,
        translation_directory=translation,
        translation_revision="b" * 40,
        output=config["defaults"]["model_manifest"],
    )
    (translation / "unexpected.json").write_text(json.dumps({"new": True}))

    with pytest.raises(ValueError, match="do not match"):
        verify_model_manifest(config)
