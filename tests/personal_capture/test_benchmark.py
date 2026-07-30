from dubbing.apps.personal_capture.benchmark import _normalize


def test_benchmark_normalization_is_case_and_punctuation_insensitive():
    assert _normalize("Hello, WORLD!") == "hello world"
    assert _normalize("Astăzi.") == "astăzi"
