# History and intent

## Timeline

- June 2026 began as an orchestrator-generated subtitle/TTS project. An earlier
  `tts_studio` tree and mock backend were deleted in commit `48ddc20`; current `dubbing`
  became the implementation.
- June 11–12 added real alignment/assembly, prosody/language delivery, adversarial input
  hardening, overlap mixing, timeline caps, whole-request caps, and bounded web jobs.
- July 27 added Sherpa-ONNX speaker diarization on branch history.
- July 29 added pluggable MLX transcription (`85312c9`), Faster-Whisper/NVIDIA support,
  multilingual Personal Capture (`d18db46`), the persistent review service (`54a4813`),
  and benchmark/outbox verification (`9c0cf42`, `7a0e7d6`).
- The current uncommitted work reorganizes application, CLI, deploy, docs, samples, and
  tests; adds resumable diarization/translation, preflight, schema, templates/static, and
  full-day gates. Because destinations are untracked, Git currently presents the move
  mostly as 53 deletions plus new directories.

## Intent inferred and verified

The history consistently favors deterministic local processing, explicit failure instead
of silent fallback, and adversarial guards. The current Personal Capture changes extend
that intent to provenance, human review, and source retention. Chunk-local speaker labels
are an explicit refusal to invent cross-chunk identity.

## Ambiguities

- “Certified release” has documentation but no release receipt or clean commit yet.
- The outbox producer was added before a consumer contract became self-contained.
- The manual retry route appears inherited from an intended queued workflow, while store
  claim behavior implements automatic retry.
- The current filesystem reorganization was interrupted before tests/config fixtures and
  generated residues were synchronized.

Git blame/log were used for these ambiguous areas only; current source remains authority.
