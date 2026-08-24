# Orphans, duplication, and islands

| Component | Classification | Evidence | Recommendation |
|---|---|---|---|
| `dubbing/capture/__pycache__` | ORPHANED | source moved; only old CPython 3.12/3.14 bytecode remains; structure test fails | QUARANTINE then remove as generated residue |
| root egg-info and `.venv/bin` | LEGACY | entrypoints target deleted `dubbing.web`; capture commands absent | CONSOLIDATE by reinstalling certified wheel |
| `output/dist/*` | LEGACY | old wheel/sdist omit Personal Capture and Faster-Whisper changes | QUARANTINE; never deploy |
| `graphify-out/*` | GENERATED / STALE | manifest predates current reorganization | regenerate after source is stable |
| old pytest/ruff/bytecode caches | GENERATED | contain both flat and reorganized test paths | remove after explicit approval; not source |
| legacy dubbing web app | DORMANT_BUT_REACHABLE | packaged console command, no systemd deployment | DOCUMENT or isolate; it is not Personal Capture |
| benchmark module | DORMANT_BUT_REACHABLE | module main, not console metadata | DOCUMENT as destructive/expensive host validation |
| JSON Schema | DORMANT_BUT_REACHABLE | present and documented, never loaded | CONNECT to validation or declare documentation-only |
| GIGA consumer | ORPHANED interface | producer exists; consumer absent; relative payload files unresolved | CONNECT before production |
| Say/eSpeak/Piper backends | ACTIVE parallel implementations | same TTS contract, platform-specific behavior | KEEP_AS_IS; similarity is intentional |
| MLX/Faster ASR backends | ACTIVE parallel implementations | same ASR contract, host-specific runtimes | KEEP_AS_IS |
| resumable ASR/diarization/translation jobs | ACTIVE parallel pattern | similar atomic checkpoint code but distinct schemas | CONSOLIDATE shared checkpoint utilities cautiously |

There are 167 byte-identical SHA-256 groups, overwhelmingly empty marker files,
certificates, licenses, or duplicated vendored/generated metadata. No byte-identical
first-party implementation pair was found in the current filesystem. Historical
old-to-new moves cannot appear as byte duplicates because the old tracked files are
currently absent; Git status records them as deletions while the destinations are
untracked.

Static no-caller results in `SYMBOL_LEDGER.csv` are marked ambiguous, never dead. Public
API methods, Flask decorators, console metadata, module mains, systemd, tests, and
file-convention invocation were checked before classification.
