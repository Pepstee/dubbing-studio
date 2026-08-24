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
