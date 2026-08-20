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

On the canonical Mac host, use the checked-in host configuration. The token option creates a
private `0600` token only if it is absent and never prints or overwrites it:

```bash
cd /Users/admin/Documents/dubbing-studio
.venv/bin/dubbing-capture-preflight \
  --config deploy/personal_capture/mac/personal-capture.json \
  --prepare \
  --generate-review-token \
  --load-models \
  --output /Users/admin/Documents/giga-user/life-logging/audio-processing/state/mac-preflight.json
```

The Mac deployment uses the existing WhisperKit primary, cached MLX independent retry and Sherpa
diarization models. It disables translation rather than making the first recording depend on an
uninstalled NLLB model. This does not weaken original-language transcript quality or review gates.

### DR-10L Pro format evidence

The production watcher has been exercised with a four-minute, real English/Russian dialogue
encoded at the DR-10L Pro's demanding normal setting: mono BWF/WAV, 48 kHz, 32-bit float. Fresh
model-loading preflight passed, ASR quality passed with no uncertain segments, diarization assigned
all eligible speech duration, source rehash/stat verification passed, and an unchanged replay was
byte-identical without creating another capture or package. The machine-readable receipt is
`benchmarks/fixtures/lesson-2026-08-01-193908/dr10l-pro-format-shadow-receipt.json`.

This proves container/codec compatibility only. The signal was derived from a hash-bound lesson
recording, not captured through the physical recorder, lavalier or analog front end. Do not call the
Tascam path acoustically certified until a real, unedited recorder file completes the same watcher
path and its speaker-attributed review package is inspected. The format-shadow package remains in
review because one genuine overlap requires acknowledgement; it emitted no GIGA event.

The same production path also completed a 56m40s mono 48 kHz 32-bit-float BWF shadow in 582.58
seconds, including 50 adaptive chunks, source-integrity verification, diarization and package
construction. An unchanged replay took 2.27 seconds and produced byte-identical package evidence.
The run remained fail-closed with nine uncertain spans and no GIGA event. Its receipt is
`benchmarks/fixtures/lesson-2026-08-01-193908/dr10l-pro-long-format-shadow-receipt.json`.
This establishes long-form format and operational compatibility, not physical-recorder acoustics
or ground-truth accuracy.

### Recorder settings for the first real capture

Set the DR-10L Pro to `WAV`, `MONO`, `48 kHz`, `32-bit float`. This is the exact format exercised
by both shadow canaries. `POLY` duplicates the same lavalier signal into two channels and roughly
doubles storage without adding another acoustic perspective. At these mono settings, four to five
hours is approximately 2.76–3.46 GB. Keep the original BWF/WAV untouched; do not normalize,
downmix or convert it before ingest. The supported settings are documented in the
[official DR-10L Pro specification](https://tascam.com/amer-es/product/dr-10l_pro).

## 2. Transfer atomically

Never copy directly to its final filename. On the Mac, use the hash-verifying ingest command. It
rejects symlinks, source mutation and filename collisions; copies through an exclusive hidden
partial; verifies the exact bytes; atomically publishes without overwrite; and writes an idempotent
receipt under the capture state directory:

```bash
cd /Users/admin/Documents/dubbing-studio
.venv/bin/dubbing-capture-ingest "/Volumes/<TASCAM CARD>/<recording>.wav" \
  --config deploy/personal_capture/mac/personal-capture.json
```

The command lands the untouched file in the configured inbox. It does not split, normalize,
approve, upload to cloud or emit a GIGA event. Run the watcher afterward, or leave the continuous
watcher active.

On WSL deployments without the ingest CLI, copy using a hidden `.partial` suffix and rename only
after the transfer has completed.

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

For an ASR-only diagnostic on the Mac, the lower-level control-plane command remains available:

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

Do not use this lower-level command as the first-recording production path because it does not
construct the diarization/review package. The Mac Personal Capture watcher is the single
end-to-end path:

```bash
.venv/bin/dubbing-capture-watch \
  --config deploy/personal_capture/mac/personal-capture.json \
  --once
```

## 4. Run the authorized month-one cloud shadow

For recordings made during the month-one programme, run Scribe v2 only after the local
`transcript.json` exists. Dubbing Studio performs all lossless chunking and checkpointing; the
operator does not cut the recording.

```bash
dubbing-cloud-teacher "$inbox/2026-07-30-full-day.wav" \
  --local-result "/home/gutua/software-factory/giga-user/life-logging/audio-processing/outputs/packages/<capture-id>/transcript.json" \
  --policy /home/gutua/software-factory/dubbing-studio/deploy/cloud_teacher/month-one-2026-08.json \
  --programme-state /home/gutua/software-factory/giga-user/life-logging/audio-processing/state/cloud-teacher-usage.json \
  --output "/home/gutua/software-factory/giga-user/life-logging/audio-processing/outputs/cloud-teacher/<capture-id>" \
  --keychain-service "dubbing-studio-elevenlabs" \
  --keychain-account "$USER"
```

This command remains blocked until the provider data-use opt-out is attested in the policy and
the credential is available from the explicitly selected macOS Keychain item. The existing item
uses service `dubbing-studio-elevenlabs` and account `$USER`. To update it without placing the API
key in shell history or process arguments, keep `-w` last so `security` prompts:

```bash
/usr/bin/security add-generic-password -U \
  -a "$USER" \
  -s "dubbing-studio-elevenlabs" \
  -w
```

On non-macOS deployments, omit both Keychain options and keep using `ELEVENLABS_API_KEY` only in
the local service environment. The command automatically removes
VAD-confirmed long silence while retaining padded speech context; the operator must not cut the
recording. Pauses up to 15 seconds remain continuous and each cloud packet comes from one source
interval, preserving provider diarization context. `compaction-plan.json` preserves the exact
reversible source-time map and reports the duration sent for billing. The programme state reserves
cost before upload; if an upload outcome becomes ambiguous, do not delete the reservation or retry
manually. A failure leaves the local package untouched. Review `report.json`,
both candidate transcripts and the excluded disagreement set; do not interpret a cloud-only span
as ground truth.

If the provider returns HTTP success but its timestamped response is malformed, the current client
stores a hash-bound `provider-response-rejections/packet-*.json`, counts the reservation and denies
resend. Re-running the same command may only reparse that saved response locally; it cannot upload
the packet again. A legacy `response_rejected` entry without a saved response is permanently
non-retryable and must remain in the programme ledger.

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
