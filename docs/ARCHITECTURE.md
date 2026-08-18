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

## Full-day capture transaction

```text
hidden .partial transfer
→ atomic rename into recordings/inbox
→ stable-age admission
→ source snapshot replay check
→ SHA-256 claim in SQLite
→ free-space and duration gates
→ automatic media probe and silence-aware adaptive ASR chunks
→ bounded overlap reconciliation and failed-span-only retries
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

Review edits regenerate the canonical JSON, plain-text and SRT projections.
Editing source text invalidates word-level text evidence for that segment and
marks an unchanged translation as requiring another review. Approval fails
closed until transcript and translation representations agree.

## Speaker-label boundary

Long-audio diarization labels are deliberately chunk-local:

```text
CHUNK_0000_SPEAKER_00
CHUNK_0001_SPEAKER_00
```

Equal suffixes in different chunks do not claim a shared human identity.
Cross-chunk voice matching and known-speaker enrolment are later evidence
layers. The current contract prefers explicit uncertainty over false identity
continuity.

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
