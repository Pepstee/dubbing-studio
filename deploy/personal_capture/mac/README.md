# Mac Personal Capture deployment

This deployment uses the canonical private runtime rooted at
`/Users/admin/Documents/giga-user/life-logging/audio-processing`. It binds the existing official
WhisperKit model as primary ASR, the independently implemented cached MLX model for rejected-span
retry, and the installed Sherpa/TitaNet models for recording-wide anonymous diarization.

Translation is deliberately disabled on this host. Original-language English, Russian, Korean and
Romanian text remains authoritative, and the absence of a translation model must not block the
first recording. Cloud transcription remains a separate explicit operator gate.

Generate the private review token and perform the real offline model probes before placing audio in
the inbox:

```bash
cd /Users/admin/Documents/dubbing-studio

.venv/bin/dubbing-capture-preflight \
  --config deploy/personal_capture/mac/personal-capture.json \
  --prepare \
  --generate-review-token \
  --load-models \
  --output /Users/admin/Documents/giga-user/life-logging/audio-processing/state/mac-preflight.json

jq -e '.ready == true and .model_load_certified == true' \
  /Users/admin/Documents/giga-user/life-logging/audio-processing/state/mac-preflight.json
```

The token generator creates `/Users/admin/.config/dubbing-studio/capture.token` with mode `0600`
only when absent. It never prints or overwrites the secret, and the token is not stored in this
repository.

After transferring a complete recorder file with a hidden `.partial` name and atomically renaming
it into `recordings/inbox`, bind that exact file into preflight:

```bash
.venv/bin/dubbing-capture-preflight \
  --config deploy/personal_capture/mac/personal-capture.json \
  --audio "/Users/admin/Documents/giga-user/life-logging/audio-processing/recordings/inbox/<file>.wav" \
  --load-models \
  --output /Users/admin/Documents/giga-user/life-logging/audio-processing/state/recording-preflight.json

jq -e '.ready == true and .audio.sha256 != null' \
  /Users/admin/Documents/giga-user/life-logging/audio-processing/state/recording-preflight.json
```

Run the watcher with the one original file. Adaptive silence-aware chunks, rejected-span retry,
channel candidates, diarization, checkpoints and package generation are internal; the operator
must not split, normalize, downmix or denoise the recording.

```bash
.venv/bin/dubbing-capture-watch \
  --config deploy/personal_capture/mac/personal-capture.json \
  --once
```

This host is not certified merely because the static manifest verifies. `ready: true` from the
model-loading preflight and a real short end-to-end package with speaker attribution are separate
required gates.
