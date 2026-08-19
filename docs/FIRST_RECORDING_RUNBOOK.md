# First full-day recording runbook

## 1. Certify the host

Confirm the encryption, backup and retention items in
`PRIVACY_AND_RETENTION.md`, then run:

Run before transferring any real recording:

```bash
dubbing-capture-preflight \
  --config ~/.config/dubbing-studio/personal-capture.json \
  --prepare \
  --load-models \
  --output ~/personal-capture-preflight.json
```

Do not admit audio unless the report contains `"ready": true`. This verifies
runtime directories, permissions, disk reserve, FFmpeg/FFprobe, local ML
dependencies, model files, offline model loading, CUDA visibility and the
private review token. A static file check alone cannot produce a ready report.
The exact model commit revisions and every self-contained model-file hash must
also match `defaults.model_manifest`.

## 2. Transfer atomically

Never copy directly to its final filename. Copy using a hidden `.partial`
suffix and rename only after the transfer has completed.

From WSL:

```bash
inbox=/home/gutua/software-factory/giga-user/life-logging/audio-processing/recordings/inbox
cp /source/path/2026-07-30-full-day.wav \
  "$inbox/.2026-07-30-full-day.wav.partial"
mv "$inbox/.2026-07-30-full-day.wav.partial" \
  "$inbox/2026-07-30-full-day.wav"
```

The watcher ignores `.partial` files. The final rename is atomic when source
and destination are on the same filesystem.

Large recordings that exceed the browser-upload limit must use this filesystem
procedure. The browser is intended for shorter captures and review.

## 3. Observe processing

```bash
systemctl --user status dubbing-capture-watch
journalctl --user -fu dubbing-capture-watch

find \
  /home/gutua/software-factory/giga-user/life-logging/audio-processing/processing \
  -name progress.json -print
```

The source remains in `recordings/inbox`. Restarting the service is safe:

```bash
systemctl --user restart dubbing-capture-watch
```

Completed ASR, diarization and translation checkpoints are reused.

## 4. Review

Open the private review interface through Tailscale. Verify:

- language and transcript text;
- uncertain words and missing speech;
- recording-global anonymous speaker consistency and any unresolved clusters;
- English translations;
- source playback at disputed timestamps.

Saving edits does not approve the package. Approval is a separate action.
If source transcript text changes while its English translation remains
unchanged, the translation is visibly marked for another review and approval
is blocked until it is saved again as reviewed.

## 5. Approve

Approval:

1. validates and regenerates every reviewed transcript/translation projection;
2. hashes the reviewed transcript, translation and approval record;
3. writes an idempotent evidence event;
4. verifies the source and evidence against the package;
5. atomically publishes one self-contained directory to
   `outbox/giga/<capture-sha256>/`.

It does not create interpreted autobiographical memory. A separate GIGA
consumer remains responsible for any later promotion.

## Failure rules

- Never delete checkpoints to “make it retry.”
- Never edit the source recording in place.
- Never replace a final inbox file while processing is active.
- Preserve failed packages and error state.
- Failed captures stay failed until the operator presses **Retry processing**.
- A watcher restart requeues only rows abandoned in `processing`, after the
  replacement watcher owns the exclusive lock.
- If a checkpoint manifest does not match, create a new capture/job identity;
  do not overwrite provenance.
