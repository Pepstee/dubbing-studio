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
- Faster-Whisper model:
  `/home/gutua/software-factory/.control/dubbing-models/faster-whisper-large-v3-turbo`
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

The two model settings in `personal-capture.json` are absolute local directories,
not registry identifiers. Production runtime sets local-only loading, so a cache
miss cannot trigger a network download after audio is admitted.

After downloading each model at an exact 40-character repository commit—not a
moving branch or tag—create the approved manifest:

```bash
dubbing-capture-model-manifest \
  --asr-directory /home/gutua/software-factory/.control/dubbing-models/faster-whisper-large-v3-turbo \
  --asr-revision ASR_COMMIT_SHA \
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

The watcher has a filesystem lock, a 60-second stability window, bounded
15-second polling, durable SQLite state and checkpointed long-audio stages.
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
  --output-dir /path/to/release-receipt/benchmark
```

Reruns are idempotent: success is based on verification of all four resulting
outbox bundles, not on the count newly delivered during that invocation.

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
