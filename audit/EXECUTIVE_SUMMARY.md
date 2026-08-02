# Executive summary

## What the repository actually is

Dubbing Studio is two local-audio products sharing one Python package:

1. a mature subtitle-to-speech dubbing engine (SRT parsing, prosody, local TTS,
   timeline alignment/assembly, batch CLI, and an unauthenticated demonstration web app);
2. a newer Personal Capture application (watched audio inbox, local ASR,
   diarization, translation, SQLite state, browser review, and a file outbox intended
   for GIGA).

The reusable engine boundaries are real: TTS, transcription, diarization, and
translation have injectable contracts. Personal Capture composes those engines rather
than being imported by them. The current filesystem, however, is an unfinished
reorganization at commit `7a0e7d6716a2b24f5756e36789e139b32050b9e9`: 53 tracked
paths are absent, 9 tracked files are modified, and 82 pre-audit files are untracked.
The new paths build, but the source tree is not a clean, installable release.

## Release verdict

**Not ready for the first valuable full-day recording.** Compilation, Ruff, wheel/sdist
construction, package-data inclusion, and 783 tests pass, but 5 tests fail and 2 skip.
The local `.venv` is stale: its `dubbing-web` command imports the deleted
`dubbing.web`, and it has none of the GPU/capture/preflight commands. The current source
wheel has the right entrypoints, but it is not what the checkout's active environment
is running.

## Highest-risk findings

| Priority | Finding | Consequence |
|---|---|---|
| P0 | Review edits update `transcript.json` but not `transcript.txt`/`.srt`; translation edits do not update `translation.txt`, and edited transcript text is not propagated into translation `source_text`. | One approved package can contain contradictory evidence representations. |
| P0 | The GIGA outbox copies only `giga-event.json`. Its payload says `transcript_file: transcript.json` (and possibly `translation.json`) but neither file is beside the event and no package path is supplied. | A consumer cannot resolve the claimed evidence from the outbox contract alone. |
| P0 | Preflight checks Python modules and diarization paths, but not that Faster-Whisper and NLLB weights are locally cached/loadable. Both use `from_pretrained`/model-name loading. | A `"ready": true` host can download on first recording or fail offline after admitting valuable audio. |
| P1 | Failed and already-`processing` captures are automatically reclaimed on every 15-second scan; there is no backoff or lease. | A persistent failure can consume GPU indefinitely; a second caller can process the same capture concurrently. |
| P1 | Transcription checkpoint manifests omit `TranscriptionOptions` (language, task, initial prompt, word-timestamp policy). Translation manifests omit detector identity. | Changed inference semantics can silently reuse old checkpoints. |
| P1 | The legacy dubbing web app binds `0.0.0.0:7432` with no authentication; `dubbing-web --help` starts it instead of showing help. | Accidental invocation exposes a synthesis endpoint on all interfaces. |
| P1 | The shell wrapper file `acceptance` may install Flask into the active/system interpreter before starting a live listener and local TTS when passed to a shell. | A seemingly harmless check has network/package-manager and process side effects. |
| P1 | The full suite fails (5 failures): an orphan `dubbing/capture/__pycache__` directory violates the canonical layout, and new config fixtures omit a newly required boundary key. | The certification document's “all gates pass” premise is false. |

## Hidden and disconnected functionality

- `dubbing.apps.dubbing_web` is a second web application, independent of Personal
  Capture. It is reachable from packaging metadata even though no deployment unit uses it.
- `dubbing.apps.personal_capture.benchmark` is not a console script. It auto-processes,
  auto-approves, and writes outbox events when invoked as a module.
- `dubbing-gpu` is an `execve` launcher that mutates `LD_LIBRARY_PATH` and re-executes
  `python -m dubbing`; it does not launch watcher/review services.
- The checked-out `.venv`, root egg-info, old wheel/sdist, pytest cache, bytecode,
  and Graphify graph describe older layouts. They are generated islands, not current
  authority.
- The JSON Schema is never loaded by production code. Runtime validation is a separate,
  hand-written implementation with a different surface.

## Disposition

Do not deploy or ingest real audio yet. First fix the evidence-representation and
outbox contracts, make retries operator-controlled/backed off, bind checkpoint manifests
to all semantic options, prove models are present in preflight, synchronize tests, remove
generated legacy residue, reinstall from the certified wheel, and rerun host-specific
preflight/benchmark on the Gigabyte. Detailed one-choice recommendations are in
`DISPOSITION_LEDGER.csv`.

## Coverage

The frozen pre-audit corpus contains exactly **30,389 regular files**, **3 symlinks**,
and **2,897 directories** (including the repository root). `FILE_LEDGER.csv` contains
exactly 30,389 rows: 143 files were read fully, 336 vendored manifests/wrappers were
inspected structurally, and 29,910 generated/vendored/binary/Git-internal files were
inventoried by metadata and SHA-256. No file is silently unaccounted for. See
`COVERAGE_REPORT.md` for the scope boundary and limitations.
