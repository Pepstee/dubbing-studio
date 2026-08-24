from scripts.run_faster_whisper_language_sweep import preceding_context_prompt


def test_preceding_context_prompt_excludes_overlapping_and_future_segments():
    transcript = {
        "segments": [
            {"start_ms": 0, "end_ms": 1000, "text": "old"},
            {"start_ms": 5000, "end_ms": 6000, "text": "near context"},
            {"start_ms": 5900, "end_ms": 7000, "text": "overlaps target"},
            {"start_ms": 8000, "end_ms": 9000, "text": "future"},
        ]
    }
    assert (
        preceding_context_prompt(transcript, 6500, context_seconds=3, max_chars=240)
        == "near context"
    )


def test_preceding_context_prompt_keeps_bounded_suffix():
    transcript = {
        "segments": [{"start_ms": 1000, "end_ms": 2000, "text": "alpha beta gamma delta"}]
    }
    assert (
        preceding_context_prompt(transcript, 3000, context_seconds=3, max_chars=12) == "gamma delta"
    )
