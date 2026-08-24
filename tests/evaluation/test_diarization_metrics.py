from dubbing.diarization.models import SpeakerTurn
from dubbing.evaluation.diarization import diarization_error_report, speaker_attributed_wer
from dubbing.transcription.models import TranscriptSegment


def test_der_maps_anonymous_speaker_labels_before_scoring():
    reference = (SpeakerTurn(0, 1000, "alice"), SpeakerTurn(1000, 2000, "bob"))
    hypothesis = (SpeakerTurn(0, 1000, "S1"), SpeakerTurn(1000, 2000, "S2"))
    report = diarization_error_report(reference, hypothesis)
    assert report["der"] == 0
    assert report["jer"] == 0


def test_der_counts_missed_speech():
    reference = (SpeakerTurn(0, 2000, "alice"),)
    hypothesis = (SpeakerTurn(0, 1000, "S1"),)
    report = diarization_error_report(reference, hypothesis)
    assert report["miss_ms"] == 1000
    assert report["der"] == 0.5


def test_speaker_attributed_wer_penalizes_wrong_identity():
    reference = (TranscriptSegment(0, 1000, "hello", speaker="alice"),)
    hypothesis = (TranscriptSegment(0, 1000, "hello", speaker="bob"),)
    assert speaker_attributed_wer(reference, hypothesis)["rate"] == 1.0
