# Pluggable Transcription Module — Mac Implementation Report

## Outcome

The repository now has a local, injectable speech-to-text subsystem that is
independent of both the existing dubbing pipeline and the speaker-diarisation
backend. The default Mac implementation uses MLX Whisper with
`mlx-community/whisper-large-v3-turbo`.

Delivered:

- a stable, versioned transcript schema with segments, words, timing,
  confidence, language, source hash, and optional speaker attribution;
- an abstract transcription backend and an MLX Whisper implementation;
- JSON, SRT, and plain-text output;
- resumable, source-validated chunk processing for long recordings;
- optional composition with the existing Sherpa-ONNX diariser;
- explicit silence, overlap, ambiguous, and unattributed states;
- a `dubbing transcribe` CLI and optional Mac dependency groups;
- deterministic unit and integration tests.

## Validation

The complete repository test suite passes:

```text
741 passed, 2 skipped
```

Ruff and Python bytecode compilation also pass.

A real local end-to-end run used an 8.39-second synthetic bilingual
English/Russian recording:

```text
This is Artyom testing the new local transcription module on his Mac.
А теперь мы проверяем русскую речь и переключение языка.
```

The production MLX backend returned two segments and 21 timestamped words. Its
source hash matched the input. A warm transcription run completed in 11.89
seconds with approximately 2.44 GB peak memory footprint. The generated SRT was
then passed through the existing macOS `say` dubbing backend, which produced an
8.26-second output WAV.

This validates the real chain:

```text
audio -> MLX Whisper -> transcript model -> SRT -> existing dubbing pipeline -> audio
```

## Deliberate boundaries

This module accepts files; it is not yet an always-on recorder or watched inbox.
It does not automatically write memories into GIGA, identify persistent people
across unrelated recordings, or implement consent and retention policy. Those
belong in the later ingestion and memory layer, not inside the transcription
backend.

No code or data has been transferred to the Gigabyte as part of this Mac-first
implementation.
