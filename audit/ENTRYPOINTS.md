# Entrypoints

## Packaged console commands

| Command | Target | Actual behavior | Current source status |
|---|---|---|---|
| `dubbing-cli` | `dubbing.cli:main` | `dub`, `batch`, `diarize`, `transcribe`, and nested `capture` commands | Builds and clean-wheel help passes |
| `dubbing-gpu` | `dubbing.gpu_launcher:main` | finds pip-installed NVIDIA library directories, prepends `LD_LIBRARY_PATH`, then `execve`s `python -m dubbing` | Builds; help fails on this Mac before CLI parsing because NVIDIA libraries are absent |
| `dubbing-web` | `dubbing.apps.dubbing_web:main` | immediately starts unauthenticated Flask dev server on `0.0.0.0:7432` | Builds; ignores `--help` |
| `dubbing-capture-watch` | `dubbing.apps.personal_capture.watcher:main` | continuous/one-shot stable-inbox scanner | Builds; clean-wheel help passes |
| `dubbing-capture-review` | `dubbing.apps.personal_capture.review:main` | authenticated review/upload Flask dev server | Builds; clean-wheel help passes |
| `dubbing-capture-preflight` | `dubbing.apps.personal_capture.preflight:main` | validates/prepares host and emits JSON report | Builds; clean-wheel help passes |

The active repository `.venv` is stale. It has only `dubbing-cli` (old target
`dubbing.__main__:main`) and `dubbing-web` (deleted target `dubbing.web:main`).
The latter raises `ModuleNotFoundError`. It has none of the other four commands.

## CLI command tree

```text
dubbing-cli
├── dub SRT [--source-audio ...]
├── batch GLOB
├── diarize AUDIO [--srt ...] [--checkpoint-dir ...]
├── transcribe AUDIO [--checkpoint-dir ...] [--diarize]
└── capture
    ├── scan INBOX --workspace ...
    ├── approve CAPTURE_ID --workspace ...
    └── list --workspace ...
```

`capture scan` always enables translation because `--translate-to` defaults to `en`
and has no disabled value. CLI `capture approve` constructs a workspace-only service,
so it writes a package event but cannot publish to the configured GIGA outbox.

## Python/module entrypoints

- `python -m dubbing` delegates to `dubbing.cli.main`.
- `python -m dubbing.apps.personal_capture.watcher --config ...`
- `python -m dubbing.apps.personal_capture.review --config ...`
- `python -m dubbing.apps.personal_capture.preflight --config ...`
- `python -m dubbing.apps.personal_capture.benchmark --config ... --audio-dir ...`
- `python acceptance.py` starts real TTS and a loopback Flask server.
- Non-executable shell wrapper `acceptance` conditionally runs pip installation when
  passed to a shell, then runs `acceptance.py`.

## Web routes

### Legacy dubbing app (`0.0.0.0:7432`, no authentication)

| Method | Route | Effect |
|---|---|---|
| GET | `/` | inline upload form |
| POST | `/dub` | parse SRT, select local TTS, render WAV into bounded in-memory job store |
| GET | `/stream/<job_id>` | stream in-memory WAV |
| GET | `/download/<job_id>` | download in-memory WAV |

### Personal Capture review (`0.0.0.0:7433` in canonical config)

All routes, including health and static assets, pass bearer/query-token or session
authentication; state-changing requests also require CSRF.

| Method | Route | Effect |
|---|---|---|
| GET | `/` | queue |
| POST | `/upload` | size/extension check, hidden partial write, atomic inbox rename |
| GET | `/capture/<id>` | review detail |
| GET | `/capture/<id>/audio` | source playback |
| POST | `/capture/<id>/save` | edit, approve, or reject |
| POST | `/capture/<id>/retry` | mark rejected/failed capture as failed for scanner |
| GET | `/health` | return watcher health JSON |

## Scheduled/background/deployment entrypoints

Two user systemd units start the watcher and review modules. Both use
`Restart=on-failure`, `ProtectSystem=strict`, `PrivateTmp=true`, and permit writes only
below the life-logging audio-processing workspace. There are no cron jobs, migrations
frameworks, queue workers, Docker entrypoints, CI workflows, submodules, or Git hooks
installed beyond Git's sample hooks.
