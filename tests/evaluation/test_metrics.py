from dubbing.evaluation.metrics import TimedText, evaluate_documents, levenshtein_distance


def test_levenshtein_and_error_rates_are_exact():
    assert levenshtein_distance(list("kitten"), list("sitting")) == 3
    report = evaluate_documents("one two three", "one too three")
    assert report["wer"]["edits"] == 1
    assert report["wer"]["rate"] == 1 / 3
    assert report["cer"]["edits"] > 0


def test_window_metrics_require_timed_reference_and_candidate():
    unavailable = evaluate_documents("hello", "hello")
    assert not unavailable["time_window_evaluation_available"]
    assert unavailable["per_time_window"] == []

    available = evaluate_documents(
        "hello world",
        "hello word",
        reference_segments=(TimedText(0, 1000, "hello world", "en"),),
        candidate_segments=(TimedText(0, 1000, "hello word", "en"),),
        duration_ms=1000,
        window_ms=1000,
    )
    assert available["time_window_evaluation_available"]
    assert available["per_time_window"][0]["wer"]["edits"] == 1


def test_script_buckets_do_not_pretend_to_be_language_labels():
    report = evaluate_documents("hello привет 안녕", "hello привет 안녕")
    assert set(report["per_script"]) >= {"latin", "cyrillic", "hangul"}
    assert "true per-language WER" in report["language_note"]
