# Directed runtime graph

## High-centrality nodes

- `CaptureService` bridges inbox admission, three ML stages, SQLite, packages, review,
  and outbox.
- `TranscriptionResult` is the data contract shared by ASR, diarization attribution,
  translation, rendering, review, and packaging.
- `dubbing.cli.main` exposes most engine functionality but not the deployed runtime
  configuration path.
- `ffmpeg_executable` is a shared external-tool boundary for ASR and diarization.

```mermaid
flowchart TD
  PYC["pyproject console scripts"] --> CLI["dubbing.cli.main"]
  PYC --> GPU["gpu_launcher"]
  PYC --> DW["dubbing_web"]
  PYC --> CW["capture watcher"]
  PYC --> CR["capture review"]
  PYC --> PF["preflight"]
  GPU -->|execve| CLI
  CW --> Config["load_config/validate_config"]
  CR --> Config
  PF --> Config
  Config --> Runtime["build_service"]
  Runtime --> CS["CaptureService"]
  CW --> CS
  CR --> CS
  CS --> Store["CaptureStore/SQLite"]
  CS --> RTJ["ResumableTranscriptionJob"]
  CS --> RDJ["ResumableDiarizationJob"]
  CS --> RLJ["ResumableTranslationJob"]
  RTJ --> FW["FasterWhisper"]
  RDJ --> Sherpa["Sherpa ONNX"]
  RLJ --> NLLB["Lingua + NLLB"]
  CS --> Package["package JSON/text/SRT"]
  CR --> Package
  CS --> Export["export_approved"]
  Export --> Event["outbox event JSON"]
  CLI --> Dubbing["DubbingPipeline"]
  DW --> Dubbing
  Dubbing --> Backend["say / Piper / eSpeak"]
  Dubbing --> Assembler["TimelineAligner + assembler"]
```

## Graphify status

The existing `graphify-out/graph.json` was generated before the current layout and is
stale. A refresh failed because no supported LLM API key is configured. A deterministic
Graphify AST attempt inside `audit/` also failed under Python 3.14 multiprocessing and
returned zero nodes; it is not used as evidence. The authoritative directed graph for
this audit is derived from the full Python AST ledger and source traces. Relationship
confidence is:

- imports, decorators, console targets, routes, subprocess calls: **EXTRACTED**;
- file-convention consumers and operational coupling: **INFERRED** and source-checked;
- future GIGA consumer behavior and cross-host state: **AMBIGUOUS**.
