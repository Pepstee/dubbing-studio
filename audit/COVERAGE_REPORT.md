# Coverage report

## Frozen scope and reconciliation

Root: `/Users/admin/Documents/dubbing-studio`

The corpus was frozen before `audit/` existed. Post-freeze `audit/` files are a deliberate
exclusion to avoid recursive self-ledger/self-hash impossibility. No other path is excluded.

| Metric | Count |
|---|---:|
| regular filesystem files | 30,389 |
| `FILE_LEDGER.csv` rows | 30,389 |
| directories including root | 2,897 |
| symlinks | 3 |
| other special entries | 0 |
| FULL | 143 |
| STRUCTURAL | 336 |
| METADATA_ONLY | 29,910 |
| UNREADABLE | 0 |
| UNSAFE | 0 |
| DEFERRED | 0 |
| human-authored/current first-party | 130 |
| generated | 30,259 |
| vendored dependency files | 29,692 |
| ignored | 30,066 |
| actual tracked files present | 48 |
| tracked paths missing locally | 53 |
| untracked, non-ignored pre-audit files | 82 |
| Git-internal files | 193 |

`find` and `rg --files --hidden --no-ignore`, both excluding only `audit/`, independently
returned 30,389 regular files. The ledger matches exactly.

The adversarial second pass recomputed every SHA-256 and size from a fresh traversal:
30,389 paths matched, with zero missing paths and zero hash/size mismatches. A separate
textual definition count found 1,295 Python `class`/`def` declarations; Python AST found
1,295 and the symbol ledger contains the same 1,295 definition rows. All 110 current
first-party Python files parsed without error. Fresh nested-repository, symlink,
executable, archive, string-entrypoint, environment, listener, subprocess, and
configuration-activation searches produced no unrecorded entrypoint.

## Classification

| Class | Count |
|---|---:|
| cache | 257 |
| configuration | 2 |
| deployment | 3 |
| documentation | 8 |
| fixture | 1 |
| frontend_asset | 1 |
| generated_knowledge_graph | 101 |
| generated_output | 10 |
| generated_package_metadata | 6 |
| git_internal | 193 |
| schema | 1 |
| script_or_source | 2 |
| source | 53 |
| template | 3 |
| test | 52 |
| test_fixture_or_helper | 4 |
| vendored_dependency | 29,692 |

All 130 human/current first-party files were read fully; three Git control/config files and
ten selected generated manifests were also read fully, giving 143 FULL rows. All current
Python files were parsed with `ast`; TOML/JSON/systemd/HTML/CSS/SRT/shell structures were
validated. 336 virtual-environment entry wrappers and dist-info manifests were inspected
structurally. Remaining vendored source, binaries, archives, caches, Git objects, and
generated outputs were hashed/classified by metadata, as permitted for non-first-party
contents.

## Files not read semantically

There are no UNREADABLE, UNSAFE, or DEFERRED ledger rows. The 29,910 METADATA_ONLY rows
are individually named in `FILE_LEDGER.csv` with reasons. They consist of:

- Git objects/logs/index/sample hooks;
- virtual-environment dependency implementation, licenses/data, native libraries;
- `.pyc`, pytest and Ruff caches;
- generated Graphify cache/graph;
- stale build distributions and synthetic output audio/images;
- generated package metadata not selected for full/structural inspection.

Three large vendored binaries exceed 100 MiB: llvmlite's dylib, Torch CPU dylib, and MLX
Metal library. Two project archives (old wheel/sdist) were inventoried and their member
lists/entrypoints inspected; nested vendored test-data archives were inventoried only.

## Git/ignore state

Branch `codex/personal-capture-v0`, commit
`7a0e7d6716a2b24f5756e36789e139b32050b9e9`, tracking the equal remote branch.
Pre-audit state: 9 modified tracked files, 53 deleted tracked paths, 82 untracked
non-ignored files grouped by Git into 15 displayed roots, and 30,066 ignored files.
There are no submodules, linked secondary worktrees, tags, nested repositories, or global
Git exclude file. Nested ignore rules exist in generated `.ruff_cache`; project ignores
cover bytecode, `.venv`, caches, egg-info, Graphify, and output.

Symlinks are the three `.venv/bin/python*` links; all resolve to Homebrew Python 3.12.
Their exact targets are in `inventory.json`.

## Incomplete or blocked analysis

- Current Graphify rebuild blocked on absent LLM API key; deterministic fallback failed.
- Git object contents and vendored implementations were not recursively interpreted.
- Native binaries and media were not disassembled/transcribed; they were typed, sized,
  hashed, and integration boundaries inspected.
- The Mac cannot establish Gigabyte runtime/systemd/CUDA/Tailscale state.
- No GIGA consumer exists to validate the outbox downstream.
- Near-duplicate similarity beyond architecture-level manual comparison was not computed
  for 29,692 vendored files; byte-identical groups were computed globally.

## Assumptions

- `[REPOSITORY]` in the attached template means Dubbing Studio, established by the active
  conversation and path.
- Pre-existing uncommitted reorganization is user-owned and was not altered after audit
  start.
- `audit/` post-freeze files are excluded from corpus counts.
- Generated output audio is synthetic as its neighboring report/history claims; content
  was not re-transcribed.
- Tailscale exposure and Gigabyte paths are claims until host validation.

## Remaining hypotheses and confidence

| Subsystem | Confidence | Remaining hypothesis |
|---|---|---|
| dubbing core | high | real platform TTS acceptance not rerun |
| CLI/package metadata | high | source wheel proven; target-host install unproven |
| long ASR/diarization jobs | high for control flow; medium for ML quality | no real long source/model run |
| translation | medium | model cache/licence/quality and edited-text policy unresolved |
| Personal Capture state/storage | high for local code | concurrency/retry behavior not load-tested |
| review security | medium-high | Tailscale/proxy and query-token logging not observed |
| GIGA handoff | high for producer defect; low downstream | consumer absent |
| Gigabyte deployment | low | host not inspected in this audit |

## Validation exclusions

Every unsafe/host-specific command not run is listed in `VALIDATION.md`. No claim of full
production certification is made.
