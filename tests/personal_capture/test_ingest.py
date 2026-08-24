from __future__ import annotations

import json
from pathlib import Path

import pytest

from dubbing.apps.personal_capture import ingest
from dubbing.apps.personal_capture.ingest import ingest_capture
from dubbing.transcription.models import TranscriptionError


def _config(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    inbox = workspace / "recordings" / "inbox"
    inbox.mkdir(parents=True)
    return {
        "workspace": {"wsl_path": str(workspace)},
        "landing_inbox": {"wsl_path": str(inbox)},
        "paths": {"state": "state"},
    }


def test_ingest_copies_exact_bytes_atomically_and_replays(tmp_path):
    config = _config(tmp_path)
    source = tmp_path / "TASCAM_0001.wav"
    source.write_bytes(b"unaltered recorder bytes")

    first = ingest_capture(source, config)
    destination = Path(first["landing_path"])
    receipt_path = (
        Path(config["workspace"]["wsl_path"])
        / "state"
        / "ingest"
        / f"{first['capture_id']}.json"
    )
    destination_stat = destination.stat()
    receipt_bytes = receipt_path.read_bytes()

    replay = ingest_capture(source, config)

    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mtime_ns == destination_stat.st_mtime_ns
    assert receipt_path.read_bytes() == receipt_bytes
    assert first["replayed"] is False
    assert replay["replayed"] is True
    assert replay["capture_id"] == first["capture_id"]
    assert first["giga_admission_allowed"] is False
    assert not tuple(Path(config["landing_inbox"]["wsl_path"]).glob("*.partial"))
    stored = json.loads(receipt_bytes)
    assert stored["landing"]["atomic_partial_publish"] is True
    assert stored["landing"]["publish_method"] == "hard-link-no-overwrite"


def test_ingest_rejects_destination_collision_with_different_bytes(tmp_path):
    config = _config(tmp_path)
    source = tmp_path / "TASCAM_0002.wav"
    source.write_bytes(b"new recorder bytes")
    destination = Path(config["landing_inbox"]["wsl_path"]) / source.name
    destination.write_bytes(b"unrelated existing bytes")

    with pytest.raises(TranscriptionError, match="different bytes"):
        ingest_capture(source, config)

    assert destination.read_bytes() == b"unrelated existing bytes"
    assert not (Path(config["workspace"]["wsl_path"]) / "state").exists()


def test_ingest_publish_race_cannot_overwrite_destination(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = tmp_path / "TASCAM_RACE.wav"
    source.write_bytes(b"source bytes")
    destination = Path(config["landing_inbox"]["wsl_path"]) / source.name
    real_link = ingest.os.link

    def competing_link(partial, target, **kwargs):
        destination.write_bytes(b"competing bytes")
        return real_link(partial, target, **kwargs)

    monkeypatch.setattr(ingest.os, "link", competing_link)

    with pytest.raises(FileExistsError):
        ingest_capture(source, config)

    assert destination.read_bytes() == b"competing bytes"
    assert not tuple(Path(config["landing_inbox"]["wsl_path"]).glob("*.partial"))


def test_ingest_rejects_direct_source_symlink(tmp_path):
    config = _config(tmp_path)
    target = tmp_path / "real.wav"
    target.write_bytes(b"audio")
    alias = tmp_path / "alias.wav"
    alias.symlink_to(target)

    with pytest.raises(TranscriptionError, match="symbolic link"):
        ingest_capture(alias, config)

    assert not tuple(Path(config["landing_inbox"]["wsl_path"]).iterdir())


def test_ingest_detects_source_mutation_during_copy(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source = tmp_path / "TASCAM_0003.wav"
    source.write_bytes(b"a" * (ingest._COPY_BLOCK_BYTES + 100))
    real_read = ingest.os.read
    real_create_source_binding = ingest.create_source_binding
    calls = 0

    def mutating_read(descriptor, size):
        nonlocal calls
        block = real_read(descriptor, size)
        calls += 1
        if calls == 1:
            source.write_bytes(b"b" * (ingest._COPY_BLOCK_BYTES + 100))
        return block

    initial_binding_created = False

    def create_binding_then_arm_copy(*args, **kwargs):
        nonlocal initial_binding_created
        binding = real_create_source_binding(*args, **kwargs)
        if not initial_binding_created:
            initial_binding_created = True
            monkeypatch.setattr(ingest.os, "read", mutating_read)
        return binding

    monkeypatch.setattr(ingest, "create_source_binding", create_binding_then_arm_copy)

    with pytest.raises(TranscriptionError, match="changed during ingest copy"):
        ingest_capture(source, config)

    assert not (Path(config["landing_inbox"]["wsl_path"]) / source.name).exists()
    assert not tuple(Path(config["landing_inbox"]["wsl_path"]).glob("*.partial"))


def test_ingest_existing_receipt_fails_if_landing_file_is_missing(tmp_path):
    config = _config(tmp_path)
    source = tmp_path / "TASCAM_0004.wav"
    source.write_bytes(b"audio")
    first = ingest_capture(source, config)
    Path(first["landing_path"]).unlink()

    with pytest.raises(TranscriptionError, match="missing or unsafe"):
        ingest_capture(source, config)


@pytest.mark.parametrize("dangling", (False, True), ids=("live", "dangling"))
def test_ingest_rejects_symlinked_receipt(tmp_path, dangling):
    config = _config(tmp_path)
    source = tmp_path / "TASCAM_0005.wav"
    source.write_bytes(b"audio")
    first = ingest_capture(source, config)
    receipt_path = (
        Path(config["workspace"]["wsl_path"])
        / "state"
        / "ingest"
        / f"{first['capture_id']}.json"
    )
    receipt_bytes = receipt_path.read_bytes()
    receipt_path.unlink()
    target = tmp_path / "forged-receipt.json"
    if not dangling:
        target.write_bytes(receipt_bytes)
    receipt_path.symlink_to(target)

    with pytest.raises(TranscriptionError, match="receipt must not be a symbolic link"):
        ingest_capture(source, config)


@pytest.mark.parametrize("name", (".hidden.wav", "notes.txt"))
def test_ingest_rejects_unsupported_media_filename(tmp_path, name):
    config = _config(tmp_path)
    source = tmp_path / name
    source.write_bytes(b"not admitted")

    with pytest.raises(TranscriptionError, match="unsupported media filename"):
        ingest_capture(source, config)
