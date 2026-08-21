# Dubbing Studio architecture

## Repository ownership

```text
dubbing/
├── apps/
│   ├── dubbing_web.py
│   └── personal_capture/
│       ├── config.py
│       ├── preflight.py
│       ├── runtime.py
│       ├── watcher.py
│       ├── service.py
│       ├── store.py
│       ├── outbox.py
│       ├── review.py
│       ├── templates/
│       └── static/
├── cli/
├── backends/                 text-to-speech plugins
├── transcription/            speech-to-text contracts and checkpoint jobs
├── diarization/              speaker-turn contracts and checkpoint jobs
├── translation/              language/translation contracts and checkpoint jobs
└── media.py                  shared local media-tool discovery
```

The engine packages do not depend on Personal Capture. Personal Capture is an
application assembled from the engine packages. The GIGA adapter consumes only
approved outbox events and has no authority inside Dubbing Studio.

Accuracy evaluation is also outside runtime admission. Per-fixture evaluators produce a scoped
measurement gate and held-out benchmark claim. A separate portfolio gate combines clean,
noisy/code-switched, natural long-form, uncertainty/coverage and speaker-attributed evidence.
Host preflight, aggregate WER and a single successful fixture cannot set production accuracy.

## Full-day capture transaction

```text
hidden .partial transfer
→ atomic rename into recordings/inbox
→ stable-age admission
→ source snapshot replay check
→ SHA-256 claim in SQLite
→ free-space and duration gates
→ automatic media probe and silence-aware adaptive ASR chunks
→ all-stream/discrete-channel preservation
→ bounded overlap reconciliation and failed-span-only retries
→ raw-first, independently corroborated channel/enhancement candidates for rejected spans
→ 2-hour diarization chunks
→ per-segment translation checkpoints
→ source snapshot and SHA-256 re-verification
→ atomic review-package publication
→ explicit human review
→ approval
→ atomic, self-contained, hash-verified outbox bundle
```

The operator supplies one original media file; chunk planning is entirely an
internal execution detail. Every expensive processing stage writes a source/configuration-bound manifest,
incremental checkpoints, a progress document and a final result. A restart
reuses completed work. ASR options and language-detector identity are part of
the checkpoint key. Adaptive checkpoints use a separate namespace from the
retained legacy fixed-job checkpoints, so migration and rollback cannot confuse
the two formats. Malformed or mismatched checkpoints fail closed.

Audio preprocessing is an adjudication boundary, not a destructive normalization stage. Every
source audio stream is merged into a lossless working chunk with discrete channels retained.
Alternate downmix and per-channel WAVs exist only inside a failed-span retry workspace. An
experimental speech-normalized policy is implemented but disabled after a failed benchmark.
Candidate source and output hashes plus processing graph are
recorded. Raw cross-model consensus has precedence; processed evidence requires two-model
agreement and conflicting processed consensuses remain uncertain. A qualifying raw consensus
short-circuits processed-audio escalation because those candidates cannot change the selected
verdict; the receipt preserves which scheduled decodes were skipped.

Review edits regenerate the canonical JSON, plain-text and SRT projections.
Editing source text invalidates word-level text evidence for that segment and
marks an unchanged translation as requiring another review. Approval fails
closed until transcript and translation representations agree.

## Speaker-label boundary

Long-audio diarization checkpoints preserve their chunk-local labels:

```text
CHUNK_0000_SPEAKER_00
CHUNK_0001_SPEAKER_00
```

Those internal labels are reconciled into recording-global anonymous
`SPEAKER_XX` labels with overlap-safe complete-link voice clustering. Temporal
overlap is a hard cannot-link constraint. Missing or ambiguous acoustic
evidence stays separate. Human identity remains a later, consented evidence
layer and is never inferred from the anonymous cluster number.

## Storage boundary

The application is source-preserving:

- it never copies or deletes the admitted recording;
- runtime bytes remain below the project-owned
  `life-logging/audio-processing` root;
- packages, checkpoints, databases and outbox files have distinct directories;
- each approved outbox directory contains its event and every referenced evidence file;
- source bytes are hashed before and after processing;
- unchanged reviewed sources are recognized by their stored snapshot before
  another multi-gigabyte hash is attempted.

## Deployment boundary

The Mac is the development and certification authority. The Gigabyte is a
release target. Working-tree files are not mirrored directly. A certified
commit is pushed to GitHub and the Gigabyte checks out that exact commit.
