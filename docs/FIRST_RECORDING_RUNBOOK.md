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
The watcher preserves all source audio streams and channels internally. If a rejected span may
benefit from a downmix or isolated channel, it creates and adjudicates those candidates itself;
the operator must not pre-process or split the recording. Normalization is not enabled in the
production policy because it failed the human-corrected difficult-span benchmark.

For a direct Mac control-plane run, use the existing WhisperKit model as the primary and an
already-cached MLX model as the independent rejected-span adjudicator:

```bash
dubbing-long-transcribe "/path/to/recording.wav" \
  --output "/path/to/checkpoints/<capture-id>" \
  --backend whisperkit \
  --model-path "/path/to/existing/whisperkit-model" \
  --start-server \
  --retry-backend mlx \
  --retry-model mlx-community/whisper-large-v3-turbo \
  --retry-mlx-temperature 0
```

The primary and retry models are each instantiated once and reused across chunks. Both identities
are bound into `manifest.json` and `run-receipt.json`. The retry option accepts only an existing
model directory or an unambiguous Hugging Face cache entry. If it cannot resolve the model locally,
the run stops before transcription and performs no download. Faster-Whisper can be selected with
`--retry-backend faster-whisper --retry-model /path/to/existing/ctranslate2-model`; it is always
constructed with offline-only model loading.

## 4. Run the authorized month-one cloud shadow

For recordings made during the month-one programme, run Scribe v2 only after the local
`transcript.json` exists. Dubbing Studio performs all lossless chunking and checkpointing; the
operator does not cut the recording.

```bash
dubbing-cloud-teacher "$inbox/2026-07-30-full-day.wav" \
  --local-result "/home/gutua/software-factory/giga-user/life-logging/audio-processing/outputs/packages/<capture-id>/transcript.json" \
  --policy /home/gutua/software-factory/dubbing-studio/deploy/cloud_teacher/month-one-2026-08.json \
  --programme-state /home/gutua/software-factory/giga-user/life-logging/audio-processing/state/cloud-teacher-usage.json \
  --output "/home/gutua/software-factory/giga-user/life-logging/audio-processing/outputs/cloud-teacher/<capture-id>"
```

This command remains blocked until the provider data-use opt-out is attested in the policy and
`ELEVENLABS_API_KEY` exists only in the local service environment. It automatically removes
VAD-confirmed long silence while retaining padded speech context; the operator must not cut the
recording. Pauses up to 15 seconds remain continuous and each cloud packet comes from one source
interval, preserving provider diarization context. `compaction-plan.json` preserves the exact
reversible source-time map and reports the duration sent for billing. The programme state reserves
cost before upload; if an upload outcome becomes ambiguous, do not delete the reservation or retry
manually. A failure leaves the local package untouched. Review `report.json`,
both candidate transcripts and the excluded disagreement set; do not interpret a cloud-only span
as ground truth.

## 5. Review

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

## 6. Approve

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
- Never denoise, normalize, downmix or extract channels before admission; retain the recorder's
  original file so the control plane can compare raw and derived evidence.
- Never replace a final inbox file while processing is active.
- Preserve failed packages and error state.
- Failed captures stay failed until the operator presses **Retry processing**.
- A watcher restart requeues only rows abandoned in `processing`, after the
  replacement watcher owns the exclusive lock.
- If a checkpoint manifest does not match, create a new capture/job identity;
  do not overwrite provenance.
