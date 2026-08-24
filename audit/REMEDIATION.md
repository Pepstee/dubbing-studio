# Forensic audit remediation receipt

Date: 2026-07-30

The original audit files remain an immutable description of the pre-remediation
checkout. This receipt records the implementation authorized after that audit.

## Remediated locally

- Review edits now regenerate transcript JSON, text and SRT plus translation JSON
  and text. Edited segment words are invalidated, translation source text is
  synchronized, and stale translations block approval.
- Approval validates every evidence representation and hashes transcript,
  translation and approval records.
- The GIGA outbox now atomically publishes a self-contained directory per capture
  with every referenced evidence file. Bundle verification and SQLite delivery
  locking are idempotent and fail closed on tampering.
- Failed captures no longer retry every polling cycle. Operator retry creates an
  explicit queued state; concurrent processing claims are rejected; a replacement
  watcher requeues only work abandoned by its predecessor after acquiring the
  exclusive lock.
- ASR checkpoints include all transcription options; translation checkpoints
  include detector identity; backend identity includes every inference-affecting
  setting and exact model revision.
- Production model settings are absolute local directories. Watcher startup and
  preflight verify a manifest containing exact 40-character revisions and the
  SHA-256 of every self-contained model file. Preflight cannot report ready unless
  the ASR, translation and diarization models load offline.
- Query-string authentication was removed. Personal Capture uses a body-submitted
  token, strict session cookies, CSRF and loopback-only Tailscale Serve exposure.
  Its deployed entrypoint uses Waitress.
- The legacy dubbing demonstration defaults to loopback and requires an explicit
  flag for remote binding. All help paths are side-effect free, including the GPU
  launcher.
- The acceptance wrapper no longer invokes a package manager.
- Runtime/manual configuration validation is the declared production authority;
  the JSON Schema is the tooling contract. Previously inert production boundary
  keys are now validated.
- Generated legacy `dubbing/capture` bytecode was removed, pytest discovery is
  restricted to the canonical test tree, the editable environment was reinstalled,
  and all packaged entrypoints were verified.

## Local certification

- `python -m compileall`: pass
- `ruff check .`: pass
- full pytest: **802 passed, 2 skipped**
- isolated wheel and sdist build: pass
- wheel inspection: model-manifest module, login template, stylesheet and console
  entrypoint metadata present
- installed environment: `pip check` pass
- installed help commands: all seven pass

## External release gates still required

These are not local code defects and cannot truthfully be certified from the Mac:

1. download the approved ASR and NLLB revisions to the configured Gigabyte paths;
2. generate the exact model manifest and run preflight with `--load-models`;
3. verify Gigabyte CUDA, disk reserve, permissions, systemd and Tailscale Serve;
4. run the synthetic four-language benchmark and one disposable long-media rehearsal;
5. record Windows/WSL encryption and backup exposure under
   `docs/PRIVACY_AND_RETENTION.md`;
6. connect and test the separate GIGA consumer before claiming automatic downstream
   ingestion (automatic interpreted-memory promotion remains disabled);
7. retain NLLB only for private non-commercial dogfooding unless its licence and
   quality are separately approved.
