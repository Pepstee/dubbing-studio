# Documentation versus reality

| Claim | Rating | Evidence |
|---|---|---|
| Personal Capture is separated under `dubbing.apps.personal_capture` | VERIFIED in current filesystem | imports respect engine/application direction |
| Full repository suite passes / certification gates are green | CONTRADICTED | 5 failed, 783 passed, 2 skipped |
| Active environment exposes all console commands | CONTRADICTED | local `.venv` is stale; web command broken; four commands missing |
| `dubbing-web --help` proves entrypoint | CONTRADICTED | starts server; no argparse |
| Every checkpoint is source/configuration-bound | PARTIALLY VERIFIED | ASR options and detector identity omitted |
| Every stage is local | PARTIALLY VERIFIED | inference is local after weights exist; named models can download on cache miss |
| Preflight certifies model files/runtime before audio | CONTRADICTED for ASR/NLLB | only diarization files and dependency importability are checked |
| Approval publishes a self-contained GIGA event | CONTRADICTED | event references package-relative transcript/translation not present in outbox |
| Review preserves coherent transcript/translation outputs | CONTRADICTED | JSON edits leave text/SRT and translation source text stale |
| Failed captures “remain retryable” | MISLEADING | they retry automatically every poll; manual retry is not the gate |
| Pipeline calls `synthesize` once per run | STALE | current implementation submits one backend call per segment in parallel |
| Legacy web jobs cannot exhaust memory | PARTIALLY VERIFIED | registry is bounded except one legal job may exceed 256 MiB by design |
| Health file is under `~/.local/share/...` | STALE | canonical config writes workspace `health.json` |
| Long-audio diarization has a four-hour limit | PARTIALLY VERIFIED | direct backend does; resumable production chunks allow up to 24 hours |
| JSON Schema defines production validation | PARTIALLY VERIFIED | schema exists but runtime uses independent manual validation |
| Certified commits only go to Gigabyte | NOT TESTABLE locally | deployment convention, no local mechanism proves remote state |

The README is also two product manuals in one long document. Its opening Personal Capture
path and later legacy dubbing/web material are both real, but the relationship and security
differences are easy to miss.
