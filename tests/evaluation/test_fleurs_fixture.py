import io
import json
import tarfile
import urllib.parse
import wave

import pytest

from scripts.build_fleurs_fixture import (
    DATASET_API,
    LANGUAGES,
    REVISION,
    build_fleurs_fixture,
)


def _wav_bytes(frames: int = 1600) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * frames)
    return output.getvalue()


def test_build_fleurs_fixture_is_four_language_and_hash_bound(tmp_path):
    audio = _wav_bytes()

    def get(url: str) -> bytes:
        if url == DATASET_API:
            return json.dumps({"sha": REVISION, "cardData": {"license": ["cc-by-4.0"]}}).encode()
        if "datasets-server" in url:
            config = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["config"][0]
            return json.dumps(
                {
                    "rows": [
                        {
                            "row_idx": 0,
                            "row": {
                                "id": 1,
                                "num_samples": 1600,
                                "transcription": f"reference {config}",
                                "raw_transcription": f"Reference {config}.",
                                "audio": [
                                    {
                                        "src": f"https://assets/{REVISION}/{config}.wav",
                                        "type": "audio/wav",
                                    }
                                ],
                            },
                        }
                    ]
                }
            ).encode()
        return audio

    fixture = build_fleurs_fixture(
        tmp_path / "clips",
        tmp_path / "fixture.json",
        samples_per_language=1,
        split="test",
        http_get=get,
    )
    assert fixture["included_count"] == 4
    assert fixture["source"]["split"] == "test"
    assert all("-test-" in span["id"] for span in fixture["spans"])
    assert fixture["language_counts"] == {language: 1 for language in LANGUAGES.values()}
    assert fixture["coverage_gaps"] == []
    assert fixture["selection_policy"]["accuracy_certification_eligible"]
    assert all(span["timestamp_human_verified"] for span in fixture["spans"])
    assert all(span["clip_relative_path"].endswith(".wav") for span in fixture["spans"])
    assert all(len(span["clip_sha256"]) == 64 for span in fixture["spans"])
    assert not fixture["giga_admission_emitted"]


def test_build_fleurs_fixture_fails_if_revision_moves(tmp_path):
    def get(_: str) -> bytes:
        return json.dumps({"sha": "changed", "cardData": {"license": ["cc-by-4.0"]}}).encode()

    with pytest.raises(RuntimeError, match="revision changed"):
        build_fleurs_fixture(tmp_path / "clips", tmp_path / "fixture.json", http_get=get)


def test_build_fleurs_fixture_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split must be"):
        build_fleurs_fixture(
            tmp_path / "clips", tmp_path / "fixture.json", split="train"
        )


def test_build_fleurs_fixture_rejects_negative_source_offset(tmp_path):
    with pytest.raises(ValueError, match="source_offset"):
        build_fleurs_fixture(
            tmp_path / "clips", tmp_path / "fixture.json", source_offset=-1
        )


def test_build_fleurs_fixture_streams_bounded_archive_prefix(tmp_path):
    audio = _wav_bytes()
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for filename in ("100.wav", "200.wav"):
            info = tarfile.TarInfo(f"test/{filename}")
            info.size = len(audio)
            archive.addfile(info, io.BytesIO(audio))
    tsv = (
        "1\t100.wav\tRaw one.\traw one\tr a w | o n e |\t1600\tFEMALE\n"
        "2\t200.wav\tRaw two.\traw two\tr a w | t w o |\t1600\tMALE\n"
    ).encode()

    def get(url: str) -> bytes:
        if url == DATASET_API:
            return json.dumps({"sha": REVISION, "cardData": {"license": ["cc-by-4.0"]}}).encode()
        if url.endswith(".tsv"):
            return tsv
        raise AssertionError(url)

    fixture = build_fleurs_fixture(
        tmp_path / "clips",
        tmp_path / "fixture.json",
        samples_per_language=1,
        split="test",
        source_mode="tar_stream",
        http_get=get,
        http_open=lambda _: io.BytesIO(archive_bytes.getvalue()),
    )
    assert fixture["included_count"] == 4
    assert fixture["selection_policy"]["source_mode"] == "tar_stream"
    assert all(span["dataset_row_id"] == 1 for span in fixture["spans"])
    assert all(span["clip_relative_path"].endswith("000000-1.wav") for span in fixture["spans"])


def test_build_fleurs_fixture_streams_bounded_archive_offset(tmp_path):
    audio = _wav_bytes()
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for filename in ("100.wav", "200.wav", "300.wav"):
            info = tarfile.TarInfo(f"test/{filename}")
            info.size = len(audio)
            archive.addfile(info, io.BytesIO(audio))
    tsv = (
        "1\t100.wav\tRaw one.\traw one\tr a w | o n e |\t1600\tFEMALE\n"
        "2\t200.wav\tRaw two.\traw two\tr a w | t w o |\t1600\tMALE\n"
        "3\t300.wav\tRaw three.\traw three\tr a w | t h r e e |\t1600\tFEMALE\n"
    ).encode()

    def get(url: str) -> bytes:
        if url == DATASET_API:
            return json.dumps(
                {"sha": REVISION, "cardData": {"license": ["cc-by-4.0"]}}
            ).encode()
        if url.endswith(".tsv"):
            return tsv
        raise AssertionError(url)

    fixture = build_fleurs_fixture(
        tmp_path / "clips",
        tmp_path / "fixture.json",
        samples_per_language=1,
        split="test",
        source_mode="tar_stream",
        source_offset=1,
        http_get=get,
        http_open=lambda _: io.BytesIO(archive_bytes.getvalue()),
    )

    assert fixture["selection_policy"]["source_offset"] == 1
    assert "offset1" in fixture["fixture_id"]
    assert all(span["dataset_row_id"] == 2 for span in fixture["spans"])
    assert all(span["clip_relative_path"].endswith("000001-2.wav") for span in fixture["spans"])
