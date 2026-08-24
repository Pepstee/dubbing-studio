import wave
from array import array

import pytest

from dubbing.transcription.audio_candidates import build_audio_candidates


def _stereo_wave(path):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        samples = array("h")
        for index in range(1600):
            samples.extend((index % 1000, -(index % 500)))
        audio.writeframes(samples.tobytes())


def test_candidate_builder_preserves_source_and_derives_bounded_channels(tmp_path):
    source = tmp_path / "source.wav"
    _stereo_wave(source)
    before = source.read_bytes()

    result = build_audio_candidates(source, tmp_path / "candidates")

    assert source.read_bytes() == before
    assert [item.identifier for item in result.candidates] == [
        "raw",
        "downmix",
        "channel-0",
        "channel-1",
    ]
    assert result.failures == ()
    assert all(
        item.source_sha256 == result.candidates[0].source_sha256
        for item in result.candidates
    )
    assert result.candidates[0].candidate_sha256 == result.candidates[0].source_sha256
    for candidate in result.candidates[1:]:
        with wave.open(str(candidate.path), "rb") as audio:
            assert audio.getnchannels() == 1


def test_mono_source_does_not_create_duplicate_channel_candidates(tmp_path):
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(array("h", [100] * 1600).tobytes())

    result = build_audio_candidates(source, tmp_path / "candidates")

    assert [item.identifier for item in result.candidates] == ["raw"]


def test_rejected_speech_normalization_remains_explicitly_opt_in(tmp_path):
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(array("h", [100] * 1600).tobytes())

    result = build_audio_candidates(
        source,
        tmp_path / "candidates",
        policies=("raw", "speech-band-normalized"),
    )

    assert [item.identifier for item in result.candidates] == [
        "raw",
        "speech-band-normalized",
    ]


@pytest.mark.parametrize(
    "policies",
    [(), ("downmix", "raw"), ("raw", "raw"), ("raw", "unknown")],
)
def test_candidate_policy_is_bounded_and_source_first(tmp_path, policies):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")

    with pytest.raises(ValueError):
        build_audio_candidates(source, tmp_path / "candidates", policies=policies)
