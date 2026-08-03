from dubbing.diarization.identity import UNKNOWN, VoiceprintRegistry


def test_voiceprint_registry_requires_calibrated_threshold_and_margin(tmp_path):
    registry = VoiceprintRegistry(tmp_path / "voices.json", threshold=0.8, margin=0.1)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    registry.add_reference("alice", first, (1.0, 0.0), consent_record="consent-a")
    registry.add_reference("bob", second, (0.0, 1.0), consent_record="consent-b")
    assert registry.match((0.99, 0.01))["identity"] == "alice"
    assert registry.match((0.7, 0.7))["identity"] == UNKNOWN


def test_manual_identity_correction_is_append_only_lineage(tmp_path):
    registry = VoiceprintRegistry(tmp_path / "voices.json")
    registry.record_manual_correction(
        recording_sha256="a" * 64,
        start_ms=100,
        end_ms=200,
        previous_identity=UNKNOWN,
        corrected_identity="alice",
        operator="operator",
    )
    document = registry._load()
    assert document["corrections"][0]["previous_identity"] == UNKNOWN
