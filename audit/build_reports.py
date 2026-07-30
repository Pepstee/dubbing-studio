#!/usr/bin/env python3
"""Render the forensic audit reports from the frozen ledgers and traced evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent
INVENTORY = json.loads((AUDIT / "inventory.json").read_text(encoding="utf-8"))


def write(name: str, text: str) -> None:
    (AUDIT / name).write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    directories = sorted(
        path.relative_to(REPO).as_posix() or "."
        for path in REPO.rglob("*")
        if path.is_dir()
        and (not path.relative_to(REPO).parts or path.relative_to(REPO).parts[0] != "audit")
    )
    directories.insert(0, ".")
    (AUDIT / "DIRECTORY_LEDGER.txt").write_text(
        "\n".join(directories) + "\n", encoding="utf-8"
    )

    write(
        "EXECUTIVE_SUMMARY.md",
        """
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
""",
    )

    write(
        "ENTRYPOINTS.md",
        """
# Entrypoints

## Packaged console commands

| Command | Target | Actual behavior | Current source status |
|---|---|---|---|
| `dubbing-cli` | `dubbing.cli:main` | `dub`, `batch`, `diarize`, `transcribe`, and nested `capture` commands | Builds and clean-wheel help passes |
| `dubbing-gpu` | `dubbing.gpu_launcher:main` | finds pip-installed NVIDIA library directories, prepends `LD_LIBRARY_PATH`, then `execve`s `python -m dubbing` | Builds; help fails on this Mac before CLI parsing because NVIDIA libraries are absent |
| `dubbing-web` | `dubbing.apps.dubbing_web:main` | immediately starts unauthenticated Flask dev server on `0.0.0.0:7432` | Builds; ignores `--help` |
| `dubbing-capture-watch` | `dubbing.apps.personal_capture.watcher:main` | continuous/one-shot stable-inbox scanner | Builds; clean-wheel help passes |
| `dubbing-capture-review` | `dubbing.apps.personal_capture.review:main` | authenticated review/upload Flask dev server | Builds; clean-wheel help passes |
| `dubbing-capture-preflight` | `dubbing.apps.personal_capture.preflight:main` | validates/prepares host and emits JSON report | Builds; clean-wheel help passes |

The active repository `.venv` is stale. It has only `dubbing-cli` (old target
`dubbing.__main__:main`) and `dubbing-web` (deleted target `dubbing.web:main`).
The latter raises `ModuleNotFoundError`. It has none of the other four commands.

## CLI command tree

```text
dubbing-cli
├── dub SRT [--source-audio ...]
├── batch GLOB
├── diarize AUDIO [--srt ...] [--checkpoint-dir ...]
├── transcribe AUDIO [--checkpoint-dir ...] [--diarize]
└── capture
    ├── scan INBOX --workspace ...
    ├── approve CAPTURE_ID --workspace ...
    └── list --workspace ...
```

`capture scan` always enables translation because `--translate-to` defaults to `en`
and has no disabled value. CLI `capture approve` constructs a workspace-only service,
so it writes a package event but cannot publish to the configured GIGA outbox.

## Python/module entrypoints

- `python -m dubbing` delegates to `dubbing.cli.main`.
- `python -m dubbing.apps.personal_capture.watcher --config ...`
- `python -m dubbing.apps.personal_capture.review --config ...`
- `python -m dubbing.apps.personal_capture.preflight --config ...`
- `python -m dubbing.apps.personal_capture.benchmark --config ... --audio-dir ...`
- `python acceptance.py` starts real TTS and a loopback Flask server.
- Non-executable shell wrapper `acceptance` conditionally runs pip installation when
  passed to a shell, then runs `acceptance.py`.

## Web routes

### Legacy dubbing app (`0.0.0.0:7432`, no authentication)

| Method | Route | Effect |
|---|---|---|
| GET | `/` | inline upload form |
| POST | `/dub` | parse SRT, select local TTS, render WAV into bounded in-memory job store |
| GET | `/stream/<job_id>` | stream in-memory WAV |
| GET | `/download/<job_id>` | download in-memory WAV |

### Personal Capture review (`0.0.0.0:7433` in canonical config)

All routes, including health and static assets, pass bearer/query-token or session
authentication; state-changing requests also require CSRF.

| Method | Route | Effect |
|---|---|---|
| GET | `/` | queue |
| POST | `/upload` | size/extension check, hidden partial write, atomic inbox rename |
| GET | `/capture/<id>` | review detail |
| GET | `/capture/<id>/audio` | source playback |
| POST | `/capture/<id>/save` | edit, approve, or reject |
| POST | `/capture/<id>/retry` | mark rejected/failed capture as failed for scanner |
| GET | `/health` | return watcher health JSON |

## Scheduled/background/deployment entrypoints

Two user systemd units start the watcher and review modules. Both use
`Restart=on-failure`, `ProtectSystem=strict`, `PrivateTmp=true`, and permit writes only
below the life-logging audio-processing workspace. There are no cron jobs, migrations
frameworks, queue workers, Docker entrypoints, CI workflows, submodules, or Git hooks
installed beyond Git's sample hooks.
""",
    )

    write(
        "ARCHITECTURE.md",
        """
# Architecture

## Real component model

```mermaid
flowchart LR
  SRT["SRT / plain text"] --> Core["DubbingPipeline"]
  Core --> TTS["TTSBackend"]
  TTS --> Say["macOS say"]
  TTS --> Piper["Piper CLI"]
  TTS --> ES["eSpeak NG"]
  Core --> Align["TimelineAligner"]
  Align --> Assemble["WAV assembler"]

  Audio["Audio / video"] --> ASR["TranscriptionBackend"]
  Audio --> Diar["DiarizationBackend"]
  ASR --> Attr["Speaker attribution"]
  Diar --> Attr
  Attr --> Translate["LanguageDetector + TranslationBackend"]

  Inbox["Personal Capture inbox"] --> Watch["Watcher"]
  Watch --> Service["CaptureService"]
  Service --> ASR
  Service --> Diar
  Service --> Translate
  Service --> DB[("capture.sqlite3")]
  Service --> Checkpoints["processing checkpoints"]
  Service --> Packages["review packages"]
  Review["Review app :7433"] --> Service
  Packages --> Outbox["GIGA outbox exporter"]

  Web["Legacy web app :7432"] --> Core
  CLI["CLI"] --> Core
  CLI --> ASR
  CLI --> Diar
  CLI --> Service
```

## Process and trust boundaries

- TTS/ASR/diarization/translation run in the invoking Python process; media conversion
  and TTS CLIs are subprocesses.
- The watcher and review server are separate persistent processes sharing SQLite,
  packages, source files, and health JSON.
- SQLite is the capture state authority; checkpoint JSON is the expensive-work resume
  authority; package JSON is review evidence; the outbox SQLite file is delivery
  deduplication state.
- Source audio and speaker-derived data are sensitive. The source inbox is trusted local
  storage; browser uploads and SRT are untrusted input.
- The Tailscale-only claim is deployment convention, not enforced by application code.
  The review app binds all interfaces and relies on token/session authentication.
- The legacy web app has no authentication and also binds all interfaces.

## State machines

Capture states are `processing`, `review`, `approved`, `failed`, and `rejected`.
New hashes begin at `processing`; success moves to `review`; operator action moves to
`approved` or `rejected`; exceptions move to `failed`. Both `failed` and `processing`
are reclaimable on the next scan. This is effectively an automatic retry loop, not a
queued manual retry state.

## Storage

Canonical host paths are:

```text
life-logging/audio-processing/
├── recordings/inbox/        source files
├── processing/<sha256>/     ASR/diarization/translation checkpoints
├── outputs/packages/<sha256>/
├── state/capture.sqlite3
├── outbox/giga/
├── capture.log
├── health.json
└── watcher.lock
```

No source models or recordings are tracked. The repository itself contains ignored
synthetic audio/transcript outputs, a stale build, generated graph data, and a 1.2 GB
virtual environment.
""",
    )

    write(
        "DATA_FLOWS.md",
        """
# Data flows

## Subtitle dubbing

`SRT/plain text → UTF-8 parse and timestamp validation → prosody stripping →
per-segment parallel TTS subprocesses → timeline alignment → PCM decode/resample/mix →
WAV and JSON output`.

The assembler enforces a four-hour default timeline cap before allocation. Overlaps are
mixed and clamped. The web path additionally caps the whole multipart body and stores
completed WAVs in a count/byte-bounded process dictionary.

## Long-audio understanding

`audio → ffprobe duration → SHA-256 → FFmpeg 16 kHz chunks → Faster-Whisper/MLX →
atomic chunk JSON → merged transcript → Sherpa chunking → chunk-local labels →
speaker attribution → language detection/NLLB → translation checkpoints`.

ASR chunks are 30 minutes with five-second overlap in production. Segment midpoint
admission deduplicates overlap. Diarization chunks are two hours and deliberately prefix
labels with the chunk ID. The 18-hour logical test exercises 36 ASR and 9 diarization
chunks without allocating an 18-hour media file.

Checkpoint manifests bind source digest, duration, backend identity, and chunk sizing.
They do **not** bind transcription language/task/prompt/word-timestamp options, and the
translation manifest does not bind detector identity.

## Personal Capture

```mermaid
sequenceDiagram
  participant U as Operator/browser
  participant I as Inbox
  participant W as Watcher
  participant S as CaptureService
  participant DB as SQLite
  participant P as Package
  participant O as GIGA outbox

  U->>I: hidden partial then rename
  W->>I: stable-age discovery
  W->>S: process(path)
  S->>DB: snapshot lookup and hash claim
  S->>S: disk/duration gates and checkpointed ML
  S->>I: stat + SHA-256 re-verification
  S->>P: staging directory then rename
  S->>DB: state=review
  U->>P: edit JSON through review app
  U->>S: approve
  S->>P: approval + event + manifest
  S->>O: verify source/transcript and copy event JSON
```

The watcher writes `health.json` after every scan and logs non-replayed outcomes.
It catches scan exceptions, but individual processing exceptions become `failed`
outcomes. The service never deletes source files.

## Evidence inconsistency path

Review saves mutate `transcript.json` and optionally `translation.json`; they do not
regenerate `transcript.txt`, `transcript.srt`, or `translation.txt`. A changed transcript
also leaves each translation segment's `source_text` unchanged. Approval hashes only
`transcript.json`, so contradictory sibling representations can survive into an approved
package.

## Outbox handoff

Approval creates a `giga.personal-capture-event.v1` document with source/transcript
hashes. Export rehashes source and transcript, inserts the event ID into outbox SQLite,
and atomically writes `<capture_sha>.json`. Only the event is copied. Its relative
`transcript_file`/`translation_file` values therefore do not resolve within the outbox.
There is no implemented GIGA consumer in this repository.

## Network/model effects

No explicit HTTP client exists in production modules, but model constructors from
Faster-Whisper/Hugging Face Transformers can fetch named models on cache miss.
The acceptance shell wrapper can invoke pip. Tailscale Serve is configured outside
the application.
""",
    )

    write(
        "RUNTIME_GRAPH.md",
        """
# Directed runtime graph

## High-centrality nodes

- `CaptureService` bridges inbox admission, three ML stages, SQLite, packages, review,
  and outbox.
- `TranscriptionResult` is the data contract shared by ASR, diarization attribution,
  translation, rendering, review, and packaging.
- `dubbing.cli.main` exposes most engine functionality but not the deployed runtime
  configuration path.
- `ffmpeg_executable` is a shared external-tool boundary for ASR and diarization.

```mermaid
flowchart TD
  PYC["pyproject console scripts"] --> CLI["dubbing.cli.main"]
  PYC --> GPU["gpu_launcher"]
  PYC --> DW["dubbing_web"]
  PYC --> CW["capture watcher"]
  PYC --> CR["capture review"]
  PYC --> PF["preflight"]
  GPU -->|execve| CLI
  CW --> Config["load_config/validate_config"]
  CR --> Config
  PF --> Config
  Config --> Runtime["build_service"]
  Runtime --> CS["CaptureService"]
  CW --> CS
  CR --> CS
  CS --> Store["CaptureStore/SQLite"]
  CS --> RTJ["ResumableTranscriptionJob"]
  CS --> RDJ["ResumableDiarizationJob"]
  CS --> RLJ["ResumableTranslationJob"]
  RTJ --> FW["FasterWhisper"]
  RDJ --> Sherpa["Sherpa ONNX"]
  RLJ --> NLLB["Lingua + NLLB"]
  CS --> Package["package JSON/text/SRT"]
  CR --> Package
  CS --> Export["export_approved"]
  Export --> Event["outbox event JSON"]
  CLI --> Dubbing["DubbingPipeline"]
  DW --> Dubbing
  Dubbing --> Backend["say / Piper / eSpeak"]
  Dubbing --> Assembler["TimelineAligner + assembler"]
```

## Graphify status

The existing `graphify-out/graph.json` was generated before the current layout and is
stale. A refresh failed because no supported LLM API key is configured. A deterministic
Graphify AST attempt inside `audit/` also failed under Python 3.14 multiprocessing and
returned zero nodes; it is not used as evidence. The authoritative directed graph for
this audit is derived from the full Python AST ledger and source traces. Relationship
confidence is:

- imports, decorators, console targets, routes, subprocess calls: **EXTRACTED**;
- file-convention consumers and operational coupling: **INFERRED** and source-checked;
- future GIGA consumer behavior and cross-host state: **AMBIGUOUS**.
""",
    )

    write(
        "CONFIGURATION_SURFACE.md",
        """
# Configuration surface

## Environment variables

| Variable | Consumer | Effect |
|---|---|---|
| `PIPER_MODEL` | backend selection and Piper backend | enables auto-selection and supplies voice model |
| `DUBBING_DIARIZATION_SEGMENTATION_MODEL` | CLI factory | fallback segmentation ONNX path |
| `DUBBING_DIARIZATION_EMBEDDING_MODEL` | CLI factory | fallback embedding ONNX path |
| `LD_LIBRARY_PATH` | GPU launcher/runtime linker | preserved and prefixed with pip NVIDIA libraries |
| `PATH` | all subprocess/tool discovery | selects ffmpeg, ffprobe, piper, say, afconvert, espeak-ng, nvidia-smi |

## Active deployment keys

- `workspace.wsl_path`
- `landing_inbox.wsl_path`, `minimum_file_age_seconds`
- `paths.packages`, `paths.processing`, `paths.state`
- ASR model/device/compute type
- translation target/model/device
- diarization model paths/device
- resumable/chunk/overlap/duration/free-space settings
- `service.poll_seconds`
- review bind host/port/upload limit/token path
- GIGA outbox path
- no-auto-promotion and no-source-deletion boundary keys

## Present but operationally inert keys

`machine`, Windows path, `purpose`/retention prose, `defaults.asr_backend`,
`defaults.review_required`, `service.restart_policy`, `network.exposure`,
`boundaries.watcher_enabled`, and `boundaries.network_upload_by_dubbing_studio` are not
used by runtime code. The systemd unit, not JSON, controls restart behavior. Runtime
hardcodes Faster-Whisper regardless of `asr_backend`.

## Schema drift

`schemas/personal-capture-deployment.v1.schema.json` is documentation/tooling only;
`load_config` never loads it. The hand-written validator is stricter in path containment
and required model keys, while the JSON Schema permits many extra fields and cannot
express overlap < half chunk. `jsonschema` is not installed in the audited environment,
so schema-library validation was unavailable. The canonical JSON did pass the actual
runtime validator.

## Secrets

No credential value is present in the frozen repository. The deployment records only the
location and permission requirement for `capture.token`. The application accepts the
token in a URL query for first authentication; that can leave it in browser history or
proxy/access logs even though response headers disable referrers and caching.

## Feature/plugin/dynamic surfaces

There is no general plugin registry. Replaceability is constructor injection plus CLI
factory choices. Dynamic imports are limited to optional ML/media packages
(`faster_whisper`, `mlx_whisper`, `sherpa_onnx`, `numpy`, `imageio_ffmpeg`). Backend
auto-selection is executable/environment convention.
""",
    )

    write(
        "HIDDEN_FUNCTIONALITY.md",
        """
# Hidden functionality

## Material findings

1. **Automatic retry is hidden in claim semantics.** `CaptureStore.claim` treats both
   `failed` and `processing` as immediately claimable. The watcher polls every 15 seconds,
   so failures retry continuously without the `/retry` route.
2. **Concurrent processing is possible outside the watcher lock.** The filesystem lock
   protects watcher instances only. CLI/module callers can reclaim a `processing` row
   after the transaction ends and share checkpoint paths.
3. **First-use network access is implicit.** Named Faster-Whisper and NLLB models can be
   resolved/downloaded by their libraries. No production module contains an HTTP call, so
   a textual network search alone misses this.
4. **The benchmark auto-approves evidence.** It is reachable only as a module/Python API,
   invokes full local ML, approves every successful package, and writes benchmark/outbox
   files.
5. **The acceptance shell wrapper installs packages.** It conditionally calls pip,
   potentially with `--break-system-packages`, before running TTS and a web listener.
6. **`dubbing-web --help` is not help.** Its `main` ignores argv and starts an
   unauthenticated all-interface Flask development server.
7. **CLI capture and deployed capture differ.** CLI scan always translates, lacks the
   deployment outbox binding, and approval cannot publish. Deployed runtime always uses
   Faster-Whisper and translation.
8. **Generated bytecode preserves the old package island.** `dubbing/capture` contains
   only old `.pyc` files. It is not a valid current implementation but makes the legacy
   directory exist and fails the structure gate.
9. **Installed metadata is an alternate entrypoint registry.** Root egg-info, `.venv`
   dist-info, `.venv/bin`, and old distributions still target `dubbing.web` and omit
   current commands.
10. **The GIGA connection is file-convention only.** No consumer or import connects it;
    event filenames, schemas, hashes, and shared workspace convention are the interface.

## Negative findings

No `eval`, source `exec`, monkey patching, plugin discovery, cron, Docker, CI-only
registration, database migration framework, authentication bypass, debug/admin route,
socket code, or hard-coded credential value was found. `os.execve` exists only in the GPU
launcher. Test monkey-patching is confined to tests.
""",
    )

    write(
        "ORPHANS_AND_ISLANDS.md",
        """
# Orphans, duplication, and islands

| Component | Classification | Evidence | Recommendation |
|---|---|---|---|
| `dubbing/capture/__pycache__` | ORPHANED | source moved; only old CPython 3.12/3.14 bytecode remains; structure test fails | QUARANTINE then remove as generated residue |
| root egg-info and `.venv/bin` | LEGACY | entrypoints target deleted `dubbing.web`; capture commands absent | CONSOLIDATE by reinstalling certified wheel |
| `output/dist/*` | LEGACY | old wheel/sdist omit Personal Capture and Faster-Whisper changes | QUARANTINE; never deploy |
| `graphify-out/*` | GENERATED / STALE | manifest predates current reorganization | regenerate after source is stable |
| old pytest/ruff/bytecode caches | GENERATED | contain both flat and reorganized test paths | remove after explicit approval; not source |
| legacy dubbing web app | DORMANT_BUT_REACHABLE | packaged console command, no systemd deployment | DOCUMENT or isolate; it is not Personal Capture |
| benchmark module | DORMANT_BUT_REACHABLE | module main, not console metadata | DOCUMENT as destructive/expensive host validation |
| JSON Schema | DORMANT_BUT_REACHABLE | present and documented, never loaded | CONNECT to validation or declare documentation-only |
| GIGA consumer | ORPHANED interface | producer exists; consumer absent; relative payload files unresolved | CONNECT before production |
| Say/eSpeak/Piper backends | ACTIVE parallel implementations | same TTS contract, platform-specific behavior | KEEP_AS_IS; similarity is intentional |
| MLX/Faster ASR backends | ACTIVE parallel implementations | same ASR contract, host-specific runtimes | KEEP_AS_IS |
| resumable ASR/diarization/translation jobs | ACTIVE parallel pattern | similar atomic checkpoint code but distinct schemas | CONSOLIDATE shared checkpoint utilities cautiously |

There are 167 byte-identical SHA-256 groups, overwhelmingly empty marker files,
certificates, licenses, or duplicated vendored/generated metadata. No byte-identical
first-party implementation pair was found in the current filesystem. Historical
old-to-new moves cannot appear as byte duplicates because the old tracked files are
currently absent; Git status records them as deletions while the destinations are
untracked.

Static no-caller results in `SYMBOL_LEDGER.csv` are marked ambiguous, never dead. Public
API methods, Flask decorators, console metadata, module mains, systemd, tests, and
file-convention invocation were checked before classification.
""",
    )

    write(
        "DOCUMENTATION_DRIFT.md",
        """
# Documentation versus reality

| Claim | Rating | Evidence |
|---|---|---|
| Personal Capture is separated under `dubbing.apps.personal_capture` | VERIFIED in current filesystem | imports respect engine/application direction |
| Full repository suite passes / certification gates are green | CONTRADICTED | 5 failed, 783 passed, 2 skipped |
| Active environment exposes all console commands | CONTRADICTED | local `.venv` is stale; web command broken; four commands missing |
| `dubbing-web --help` proves entrypoint | CONTRADICTED | starts server; no argparse |
| Every checkpoint is source/configuration-bound | PARTIALLY VERIFIED | ASR options and detector identity omitted |
| Every stage is local | PARTIALLY VERIFIED | inference is local after weights exist; named models can download on cache miss |
| Preflight certifies model files/runtime before audio | CONTRADICTED for ASR/NLLB | only diarization files and dependency importability are checked |
| Approval publishes a self-contained GIGA event | CONTRADICTED | event references package-relative transcript/translation not present in outbox |
| Review preserves coherent transcript/translation outputs | CONTRADICTED | JSON edits leave text/SRT and translation source text stale |
| Failed captures “remain retryable” | MISLEADING | they retry automatically every poll; manual retry is not the gate |
| Pipeline calls `synthesize` once per run | STALE | current implementation submits one backend call per segment in parallel |
| Legacy web jobs cannot exhaust memory | PARTIALLY VERIFIED | registry is bounded except one legal job may exceed 256 MiB by design |
| Health file is under `~/.local/share/...` | STALE | canonical config writes workspace `health.json` |
| Long-audio diarization has a four-hour limit | PARTIALLY VERIFIED | direct backend does; resumable production chunks allow up to 24 hours |
| JSON Schema defines production validation | PARTIALLY VERIFIED | schema exists but runtime uses independent manual validation |
| Certified commits only go to Gigabyte | NOT TESTABLE locally | deployment convention, no local mechanism proves remote state |

The README is also two product manuals in one long document. Its opening Personal Capture
path and later legacy dubbing/web material are both real, but the relationship and security
differences are easy to miss.
""",
    )

    write(
        "HISTORY_AND_INTENT.md",
        """
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
""",
    )

    write(
        "VALIDATION.md",
        """
# Safe validation

All commands ran against the frozen source state. Bytecode/build/install artifacts were
redirected under `audit/`; pytest cache and bytecode writes were disabled.

| Validation | Exit | Duration/result | Classification |
|---|---:|---|---|
| `PYTHONPYCACHEPREFIX=audit/validation-pyc python -m compileall -q dubbing tests acceptance.py` | 0 | 0.35 s | pass |
| `.venv/bin/ruff check --no-cache . --exclude audit` | 0 | 0.02 s | pass |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider` | 1 | 69.84 s; 783 passed, 5 failed, 2 skipped | code/test-state defects |
| build isolated copy into `audit/build-dist` | 0 | 3.79 s; sdist + wheel | pass |
| inspect wheel package data | 0 | 3 HTML templates + CSS + entrypoint metadata present | pass |
| clean audit venv install with `--no-deps` | 0 | installed current wheel | pass |
| clean wheel help: CLI/watch/review/preflight | 0 | all four pass | pass |
| clean wheel `dubbing-gpu --help` | 1 | no NVIDIA wheel libraries on Mac | environmental absence plus launcher-before-parser behavior |
| patched `Flask.run` probe of `dubbing-web --help` | 0 probe | would bind `0.0.0.0:7432`; help ignored | code behavior defect |
| canonical deployment JSON through actual `load_config` | 0 | pass | pass |
| JSON Schema library validation | not run | `jsonschema` unavailable | missing dependency |
| Graphify refresh | 1 | no supported LLM API key | environmental blocker |
| Graphify deterministic AST attempt | incomplete | Python 3.14 spawn failure; 0 nodes | tool defect; ignored |

## Test failures

1. `dubbing/capture` still exists because of ignored legacy bytecode.
2. The test config fixture omits
   `giga_outbox.automatic_interpreted_memory_promotion=false`, now required.
3. The promotion-error assertion is reached at the missing outbox key first.
4. Preflight fixture fails for the same missing key.
5. Review-app fixture fails for the same missing key.

## Unsafe or host-specific checks not run

- `sh acceptance` / `acceptance.py`: the wrapper can install packages; both paths can
  run real TTS and start a listener.
- legacy or review Flask servers: listeners intentionally not started.
- Personal Capture benchmark: loads models, mutates packages/SQLite/outbox, and can download.
- real ASR, diarization, NLLB inference or large-media rehearsal: models/host not certified.
- Gigabyte preflight, CUDA, model files, systemd, Tailscale, paths, disk, permissions, or
  remote commit: unavailable from the audited Mac and intentionally not changed.
- GIGA consumer end-to-end: no consumer exists in this repository.
- deployment/publish/migration: outside read-only scope.
""",
    )

    write(
        "UNRESOLVED_QUESTIONS.md",
        """
# Unresolved questions

1. What exact contract should make outbox evidence resolvable: copy transcript/translation,
   embed content, or publish an immutable package URI?
2. Which package representation is authoritative after review, and should text/SRT be
   regenerated or removed?
3. Should translation be recomputed after transcript edits, or explicitly marked stale?
4. Should failures retry only on operator request, with exponential backoff, or on a
   bounded schedule?
5. What lease/ownership prevents CLI and watcher from processing the same capture?
6. Which ASR/NLLB model revisions and hashes are approved, and where must preflight find
   them without network access?
7. Should checkpoint identity include all transcription options, detector identity, model
   revisions, code/schema version, and media decoder version?
8. Is the unauthenticated legacy web product still required? If yes, where may it bind and
   what authentication/rate limits apply?
9. Is query-string token bootstrap acceptable given history/access-log exposure?
10. Should `capture scan` allow translation to be disabled, and should CLI approval accept
    deployment config so it can publish consistently?
11. What is the authoritative runtime root on the Gigabyte, and does it match the systemd
    `WorkingDirectory=/home/gutua/software-factory/projects/dubbing-studio`?
12. What GIGA consumer validates and consumes `giga.personal-capture-event.v1`?
13. Are ignored synthetic transcript/audio artifacts safe to retain, and are any derived
    from a real voice? Metadata suggests a synthetic Mac E2E run; this was not independently
    re-proven.
14. Is NLLB's non-commercial checkpoint acceptable for the intended lifecycle?
15. What retention, encryption-at-rest, backup, and deletion policy applies to source
    recordings and biometric-like speaker evidence?
""",
    )

    counts = INVENTORY["counts"]
    write(
        "COVERAGE_REPORT.md",
        f"""
# Coverage report

## Frozen scope and reconciliation

Root: `/Users/admin/Documents/dubbing-studio`

The corpus was frozen before `audit/` existed. Post-freeze `audit/` files are a deliberate
exclusion to avoid recursive self-ledger/self-hash impossibility. No other path is excluded.

| Metric | Count |
|---|---:|
| regular filesystem files | {INVENTORY['regular_file_count']:,} |
| `FILE_LEDGER.csv` rows | {INVENTORY['ledger_rows']:,} |
| directories including root | {len(directories):,} |
| symlinks | {len(INVENTORY['symlinks']):,} |
| other special entries | 0 |
| FULL | {counts['read_status'].get('FULL', 0):,} |
| STRUCTURAL | {counts['read_status'].get('STRUCTURAL', 0):,} |
| METADATA_ONLY | {counts['read_status'].get('METADATA_ONLY', 0):,} |
| UNREADABLE | {counts['read_status'].get('UNREADABLE', 0):,} |
| UNSAFE | {counts['read_status'].get('UNSAFE', 0):,} |
| DEFERRED | {counts['read_status'].get('DEFERRED', 0):,} |
| human-authored/current first-party | {counts['generated_or_human'].get('human', 0):,} |
| generated | {counts['generated_or_human'].get('generated', 0):,} |
| vendored dependency files | {counts['classification'].get('vendored_dependency', 0):,} |
| ignored | {counts['ignored_status'].get('IGNORED', 0):,} |
| actual tracked files present | {counts['tracked_status'].get('TRACKED', 0):,} |
| tracked paths missing locally | {len(INVENTORY['tracked_missing']):,} |
| untracked, non-ignored pre-audit files | {counts['tracked_status'].get('UNTRACKED', 0):,} |
| Git-internal files | {counts['tracked_status'].get('GIT_INTERNAL', 0):,} |

`find` and `rg --files --hidden --no-ignore`, both excluding only `audit/`, independently
returned 30,389 regular files. The ledger matches exactly.

The adversarial second pass recomputed every SHA-256 and size from a fresh traversal:
30,389 paths matched, with zero missing paths and zero hash/size mismatches. A separate
textual definition count found 1,295 Python `class`/`def` declarations; Python AST found
1,295 and the symbol ledger contains the same 1,295 definition rows. All 110 current
first-party Python files parsed without error. Fresh nested-repository, symlink,
executable, archive, string-entrypoint, environment, listener, subprocess, and
configuration-activation searches produced no unrecorded entrypoint.

## Classification

| Class | Count |
|---|---:|
""" + "\n".join(
            f"| {key} | {value:,} |"
            for key, value in sorted(counts["classification"].items())
        ) + """

All 130 human/current first-party files were read fully; three Git control/config files and
ten selected generated manifests were also read fully, giving 143 FULL rows. All current
Python files were parsed with `ast`; TOML/JSON/systemd/HTML/CSS/SRT/shell structures were
validated. 336 virtual-environment entry wrappers and dist-info manifests were inspected
structurally. Remaining vendored source, binaries, archives, caches, Git objects, and
generated outputs were hashed/classified by metadata, as permitted for non-first-party
contents.

## Files not read semantically

There are no UNREADABLE, UNSAFE, or DEFERRED ledger rows. The 29,910 METADATA_ONLY rows
are individually named in `FILE_LEDGER.csv` with reasons. They consist of:

- Git objects/logs/index/sample hooks;
- virtual-environment dependency implementation, licenses/data, native libraries;
- `.pyc`, pytest and Ruff caches;
- generated Graphify cache/graph;
- stale build distributions and synthetic output audio/images;
- generated package metadata not selected for full/structural inspection.

Three large vendored binaries exceed 100 MiB: llvmlite's dylib, Torch CPU dylib, and MLX
Metal library. Two project archives (old wheel/sdist) were inventoried and their member
lists/entrypoints inspected; nested vendored test-data archives were inventoried only.

## Git/ignore state

Branch `codex/personal-capture-v0`, commit
`7a0e7d6716a2b24f5756e36789e139b32050b9e9`, tracking the equal remote branch.
Pre-audit state: 9 modified tracked files, 53 deleted tracked paths, 82 untracked
non-ignored files grouped by Git into 15 displayed roots, and 30,066 ignored files.
There are no submodules, linked secondary worktrees, tags, nested repositories, or global
Git exclude file. Nested ignore rules exist in generated `.ruff_cache`; project ignores
cover bytecode, `.venv`, caches, egg-info, Graphify, and output.

Symlinks are the three `.venv/bin/python*` links; all resolve to Homebrew Python 3.12.
Their exact targets are in `inventory.json`.

## Incomplete or blocked analysis

- Current Graphify rebuild blocked on absent LLM API key; deterministic fallback failed.
- Git object contents and vendored implementations were not recursively interpreted.
- Native binaries and media were not disassembled/transcribed; they were typed, sized,
  hashed, and integration boundaries inspected.
- The Mac cannot establish Gigabyte runtime/systemd/CUDA/Tailscale state.
- No GIGA consumer exists to validate the outbox downstream.
- Near-duplicate similarity beyond architecture-level manual comparison was not computed
  for 29,692 vendored files; byte-identical groups were computed globally.

## Assumptions

- `[REPOSITORY]` in the attached template means Dubbing Studio, established by the active
  conversation and path.
- Pre-existing uncommitted reorganization is user-owned and was not altered after audit
  start.
- `audit/` post-freeze files are excluded from corpus counts.
- Generated output audio is synthetic as its neighboring report/history claims; content
  was not re-transcribed.
- Tailscale exposure and Gigabyte paths are claims until host validation.

## Remaining hypotheses and confidence

| Subsystem | Confidence | Remaining hypothesis |
|---|---|---|
| dubbing core | high | real platform TTS acceptance not rerun |
| CLI/package metadata | high | source wheel proven; target-host install unproven |
| long ASR/diarization jobs | high for control flow; medium for ML quality | no real long source/model run |
| translation | medium | model cache/licence/quality and edited-text policy unresolved |
| Personal Capture state/storage | high for local code | concurrency/retry behavior not load-tested |
| review security | medium-high | Tailscale/proxy and query-token logging not observed |
| GIGA handoff | high for producer defect; low downstream | consumer absent |
| Gigabyte deployment | low | host not inspected in this audit |

## Validation exclusions

Every unsafe/host-specific command not run is listed in `VALIDATION.md`. No claim of full
production certification is made.
""",
    )

    disposition_fields = [
        "item", "recommendation", "evidence", "present_role", "reachability",
        "dependencies", "risk_of_change", "tests", "proposed_destination_or_connection",
        "confidence", "verification_before_mutation",
    ]
    rows = [
        ("dubbing core", "KEEP_AS_IS", "783 passing tests; mature history", "SRT-to-WAV engine", "CLI/web/Python API", "TTS tools", "medium", "core test suites", "retain package boundaries", "high", "full green suite"),
        ("Personal Capture evidence renderings", "CONSOLIDATE", "review updates JSON only", "parallel evidence formats", "review/approval/package", "renderers", "high", "review tests incomplete", "one canonical model with regenerated projections", "high", "add edit/approve consistency tests"),
        ("GIGA outbox payload", "CONNECT", "relative transcript not exported", "reviewed-event producer", "approved package export", "source/package/outbox", "high", "benchmark only counts events", "self-contained bundle or immutable resolvable URI", "high", "consumer contract test"),
        ("preflight model checks", "CONNECT", "ASR/NLLB cache not checked", "host admission gate", "console/systemd runbook", "model runtimes/cache", "high", "mocked preflight only", "verify exact revisions/hashes offline", "high", "cold/offline host test"),
        ("capture retry semantics", "ORGANISE", "failed/processing reclaimed each poll", "error recovery", "watcher/store", "SQLite/watcher", "high", "state tests partial", "lease + bounded backoff/operator retry policy", "high", "failure-loop and concurrency tests"),
        ("checkpoint identity", "CONSOLIDATE", "semantic options omitted", "resume authority", "ASR/diar/translation", "backend/model/options", "high", "resume tests only source changes", "shared complete identity contract", "high", "option-change invalidation tests"),
        ("legacy dubbing web", "DOCUMENT", "packaged unauthenticated 0.0.0.0 app", "demo web UI", "dubbing-web", "Flask/TTS", "high if exposed", "web security tests", "separate demo product and safe bind/auth policy", "high", "decide retention"),
        ("acceptance shell wrapper", "QUARANTINE", "pip install + listener/TTS side effects", "developer convenience", "non-executable shell file", "pip/Flask/TTS", "high", "mocked acceptance tests", "replace with explicit documented environment setup", "high", "approve behavior change"),
        ("current test/config drift", "ORGANISE", "5 failures", "certification gates", "pytest", "fixtures/caches", "low", "full suite", "synchronize fixture and canonical layout", "high", "rerun full suite"),
        ("local .venv installation", "CONSOLIDATE", "old/broken entrypoints", "developer runtime", "shell commands", "editable install", "medium", "clean-wheel probes", "recreate/install certified wheel", "high", "verify all commands without starting servers"),
        ("old output distributions", "QUARANTINE", "omit current source/commands", "historical build", "ignored files", "packaging", "low", "member inspection", "retain only as labelled audit evidence or remove", "high", "compare hashes/release receipts"),
        ("JSON Schema", "CONNECT", "not runtime-loaded", "declared config contract", "tools/docs only", "schema validator", "medium", "manual validation tests", "generate/manual validation from one source", "high", "schema conformance tests"),
        ("benchmark module", "DOCUMENT", "auto-approves and mutates outbox", "host acceptance", "module main/API", "all ML models", "medium-high", "minimal benchmark test", "explicit destructive/expensive runbook command", "high", "idempotent rerun test"),
        ("generated caches and old bytecode", "DELETE_CANDIDATE", "stale paths; structure failure", "no runtime authority", "ignored only", "tools", "low", "structure test", "remove after source changes are committed", "high", "fresh traversal and suite"),
        ("Graphify output", "QUARANTINE", "stale and failed refresh", "secondary graph", "ignored artifact", "Graphify/API key", "low", "none", "regenerate from stable commit", "high", "node/file reconciliation"),
        ("source recordings and speaker evidence", "DOCUMENT", "sensitive biometric-like data", "production inputs/derived evidence", "inbox/packages", "filesystem policy", "high", "no retention tests", "formal retention/encryption/backup policy", "high", "operator/legal review"),
        ("NLLB backend", "INVESTIGATE_FURTHER", "CC-BY-NC model and quality unproven", "private translation", "runtime", "torch/transformers/model", "high commercially", "mocked unit tests", "approved local model/revision", "high", "licence and quality evaluation"),
        ("systemd/Tailscale deployment", "INVESTIGATE_FURTHER", "Mac-only evidence", "persistent host services", "Gigabyte", "WSL/systemd/Tailscale/CUDA", "high", "none local", "exact-commit host certification receipt", "high", "run host preflight/benchmark"),
    ]
    with (AUDIT / "DISPOSITION_LEDGER.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(disposition_fields)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
