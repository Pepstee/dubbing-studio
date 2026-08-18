# Personal Capture certification

A release is ready to receive real audio only when every gate below passes.

## Structural gates

- Personal Capture lives under `dubbing.apps.personal_capture`.
- Reusable ML engines do not import the application.
- Tests mirror package ownership.
- Templates and static assets are included in the built wheel.
- Deployment assets live under `deploy/personal_capture/<host>`.
- No runtime output, cache, database, model or recording is tracked.
- The host encryption/backup/retention receipt required by
  `PRIVACY_AND_RETENTION.md` is complete.

`schemas/personal-capture-deployment.v1.schema.json` is the tooling/reference
contract. `dubbing.apps.personal_capture.config.validate_config` is the runtime
authority and enforces additional containment and cross-field rules that JSON
Schema does not express; the canonical deployment must pass both structural
JSON parsing and that runtime validator.

## Deterministic gates

```bash
python3 -m compileall -q dubbing
python3 -m ruff check .
python3 -m pytest -q
python3 -m build
```

Install the built wheel into a clean environment and prove all console entry
points:

```bash
dubbing-cli --help
dubbing-gpu --help
dubbing-web --help
dubbing-capture-watch --help
dubbing-capture-review --help
dubbing-capture-preflight --help
dubbing-capture-model-manifest --help
```

## Full-day logical gate

`tests/personal_capture/test_full_day_gate.py` retains a deterministic stress
gate for the legacy fixed-chunk primitive:

- 36 resumable 30-minute ASR chunks;
- 9 resumable 2-hour diarization chunks;
- replay without repeated backend calls;
- deterministic chunk-local speaker labels.

Production Personal Capture uses the adaptive coordinator instead. Its gate
submits one media path, automatically creates multiple silence-aware ASR
checkpoints, reconciles overlap, and resumes without asking the operator to
prepare chunks.

## Host gate

On the release target:

```bash
dubbing-capture-preflight \
  --config ~/.config/dubbing-studio/personal-capture.json \
  --prepare \
  --load-models
```

Then run the synthetic multilingual benchmark and one disposable long-media
rehearsal. A Mac-only test pass cannot certify CUDA libraries, model files,
systemd units or Gigabyte filesystem permissions.

Preflight intentionally reports `"ready": false` unless every configured model
is both present at its absolute local path and successfully loaded with network
fetching disabled.

## Release rule

Deploy only a clean, versioned Git commit fetched from GitHub. Do not `rsync`
working-tree modules to the Gigabyte. Record the commit, preflight report,
test totals and benchmark hashes in the release receipt.
