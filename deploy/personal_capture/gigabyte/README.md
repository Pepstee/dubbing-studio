# Gigabyte Personal Capture deployment

## Durable locations

- Inbox: `/home/gutua/software-factory/giga-user/life-logging/audio-processing/recordings/inbox`
- Workspace: `/home/gutua/software-factory/giga-user/life-logging/audio-processing`
- Config: `/home/gutua/.config/dubbing-studio/personal-capture.json`
- Secret: `/home/gutua/.config/dubbing-studio/capture.token` (`0600`)
- Processing checkpoints: `<workspace>/processing`
- Review packages: `<workspace>/outputs/packages`
- State: `<workspace>/state`
- GIGA transactional outbox: `<workspace>/outbox/giga`
- Primary Faster-Whisper model:
  `/home/gutua/software-factory/.control/dubbing-models/faster-whisper-large-v3-turbo`
- Independent retry Faster-Whisper model:
  `/home/gutua/software-factory/.control/dubbing-models/faster-whisper-large-v3-edaa852`
- NLLB model:
  `/home/gutua/software-factory/.control/dubbing-models/nllb-200-distilled-600M`
- FFmpeg and FFprobe:
  `/home/gutua/software-factory/.control/ffmpeg/bin`

This host receives certified GitHub commits only. Do not copy individual
working-tree files into the deployment.

The systemd units prepend the host-owned FFmpeg directory to `PATH`. This keeps
the media toolchain available to the unprivileged WSL service account without
requiring a mutable system package install. Install both `ffmpeg` and `ffprobe`
there from one pinned release, verify its publisher-provided SHA-256 before
extraction, and record the release tag, archive name and digest in the host
release receipt.

The three model settings in `personal-capture.json` are absolute local directories,
not registry identifiers. Production runtime sets local-only loading, so a cache
miss cannot trigger a network download after audio is admitted.

After downloading each model at an exact 40-character repository commit—not a
moving branch or tag—create the approved manifest:

```bash
dubbing-capture-model-manifest \
  --asr-directory /home/gutua/software-factory/.control/dubbing-models/faster-whisper-large-v3-turbo \
  --asr-revision ASR_COMMIT_SHA \
  --asr-retry-directory /home/gutua/software-factory/.control/dubbing-models/faster-whisper-large-v3-edaa852 \
  --asr-retry-revision edaa852ec7e145841d8ffdb056a99866b5f0a478 \
  --translation-directory /home/gutua/software-factory/.control/dubbing-models/nllb-200-distilled-600M \
  --translation-revision NLLB_COMMIT_SHA \
  --output /home/gutua/software-factory/.control/dubbing-models/personal-capture-model-manifest.json
```

This rejects symlinked/cache-dependent directories and hashes every model file.
Watcher startup and host preflight both fail closed if a revision, file set or
file hash differs.

## Operations

```bash
systemctl --user restart dubbing-capture-watch dubbing-capture-review
systemctl --user status dubbing-capture-watch dubbing-capture-review
journalctl --user -u dubbing-capture-watch -u dubbing-capture-review
```

Before enabling either unit:

```bash
export PATH=/home/gutua/software-factory/.control/ffmpeg/bin:$PATH

dubbing-capture-preflight \
  --config /home/gutua/.config/dubbing-studio/personal-capture.json \
  --prepare \
  --load-models
```

Preflight `ready: true` means the host and pinned models are operational; it does not mean 90%
accuracy or production certification. Generate v2 scoped reports with
`python -m dubbing.evaluation.language_sweep`, then combine them with
`python -m dubbing.evaluation.accuracy_gate`. Only the latter can make a production accuracy
claim, and only when every natural/noisy/language/speaker/coverage requirement is present.

The watcher has a filesystem lock, a 60-second stability window, bounded
15-second polling, durable SQLite state and checkpointed long-audio stages.
The operator uploads one original file. The watcher probes it, plans adaptive
silence-aware chunks, reconciles contextual overlap and retries only rejected
spans internally; no manual cutting is part of the operating procedure. A
targeted retry is never accepted on the primary model's evidence alone. The
independent full `large-v3` model must return a healthy candidate. Cross-model
agreement can produce a clean replacement; disagreement is retained as an
explicitly uncertain span and remains blocked from GIGA admission.

Rejected spans are additionally passed through the bundled local Silero VAD.
A strict pass creates precise micro-regions; a more sensitive pass may add only
non-overlapping regions, so it cannot swallow stricter boundaries. Each selected
region is decoded independently by both ASR models. Long uncovered intervals,
failed micro-regions and model disagreements remain explicit uncertainty rather
than being silently treated as no speech. The thresholds and detector identity
are checkpoint-bound in the deployment config. Exact text agreement is also
rejected when both models expose acoustic confidence and either falls below the
control plane's `0.35` minimum.
One-token agreement also remains uncertain because two Whisper-family models
can confidently agree on the wrong short phonetic neighbor.

Every source audio stream is retained as discrete channels in the lossless working chunk. For a
rejected span the watcher can compare raw audio, downmix and up to four individual channels.
The implemented speech-normalization experiment is disabled because it reduced accuracy on
both production ASR models. Raw cross-model consensus always wins.
A processed candidate is clean only when both pinned ASR models agree on that same candidate;
disagreement between independently corroborated processed candidates remains explicit
uncertainty. The original recording is never rewritten, and every derived candidate is
hash-bound with its processing graph in the retry receipt.

Diarization remains resumable in two-hour chunks, but anonymous speaker labels are reconciled
across the whole recording. The offline Sherpa fallback uses an explicit `0.85` clustering
distance threshold. Its already-installed TitaNet model extracts embeddings only from
non-overlapping speech; the global complete-link reconciler requires cosine similarity `0.80`,
uses a `0.05` margin against competing temporally incompatible voices, and treats temporal
overlap as a hard cannot-link constraint. It can therefore repair non-overlapping fragmentation
inside a chunk without erasing real overlap. Missing or ambiguous evidence creates a new
`SPEAKER_XX` label instead of a guess. These settings and the embedding provider are
checkpoint-bound, and full merge/ambiguity evidence is retained in provenance.

pyannote Community-1 is supported as the preferred local reference backend. Production selects
it only from an absolute local model path after the operator has independently obtained access;
the service never accepts model terms, fetches a gated model, or falls back across that boundary
silently. Sherpa/TitaNet remains available as both fallback diarizer and independent global
embedding provider.

Failed captures remain failed until the operator explicitly queues a retry.
Work interrupted by a dead watcher is requeued once by its replacement after
that process acquires the exclusive watcher lock.

## Synthetic host benchmark

The benchmark is intentionally expensive and mutating: it loads all configured
models, processes four synthetic fixtures, approves their packages and publishes
outbox bundles. Run it only against the dedicated certification workspace:

```bash
python -m dubbing.apps.personal_capture.benchmark \
  --config /home/gutua/.config/dubbing-studio/personal-capture.json \
  --audio-dir /path/to/synthetic-four-language-fixtures \
  --output-dir /path/to/release-receipt/benchmark \
  --generator edge-tts
```

Reruns are idempotent: success is based on verification of all four resulting
outbox bundles, not on the count newly delivered during that invocation.
Certification also requires every fixture to meet the configured transcript
similarity threshold, match its expected source language and produce a
non-empty translation result. The command exits nonzero if either semantic
quality or the evidence handoff fails.

## Private network

The service listens only on WSL loopback port 7433. No public firewall rule is created.
Tailscale Serve
must be enabled for the tailnet once by its owner, then configured on Windows:

```powershell
tailscale serve --bg --https=443 http://localhost:7433
tailscale serve status
```

Open the returned HTTPS URL, then enter the contents of `capture.token` in the sign-in form.
The token is submitted in the request body, never in browser history or an access-log URL.
All later modifying forms also require CSRF.
The Python process is served by bounded Waitress worker threads; Flask's
development server is not used by the deployed entrypoint.

## GIGA boundary

Approval writes `giga-event.json` inside the derived package. The outbox exporter recomputes
source-audio and evidence hashes, publishes an atomic directory containing the event,
transcript, optional translation and approval record, and records the event id in its own
SQLite ledger. A separate GIGA consumer is intentionally required to map this reviewed
evidence into TwinStore; there is no automatic interpreted-memory write.
