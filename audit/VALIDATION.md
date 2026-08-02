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
