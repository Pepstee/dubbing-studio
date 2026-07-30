import pytest

from dubbing.apps.personal_capture.benchmark import _normalize, run_benchmark


def test_benchmark_normalization_is_case_and_punctuation_insensitive():
    assert _normalize("Hello, WORLD!") == "hello world"
    assert _normalize("Astăzi.") == "astăzi"


@pytest.mark.parametrize("minimum", [-0.01, 1.01])
def test_benchmark_rejects_invalid_similarity_threshold(minimum, tmp_path):
    with pytest.raises(ValueError, match="between 0 and 1"):
        run_benchmark(
            tmp_path / "config.json",
            tmp_path / "audio",
            tmp_path / "output",
            minimum_similarity=minimum,
        )
