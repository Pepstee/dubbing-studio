# Dubbing Studio

A composable, local-first audio pipeline with replaceable speech-to-text,
speaker-diarisation, and text-to-speech backends. It can transcribe source audio,
identify who spoke when, and turn edited SRT subtitles back into timestamp-aligned
audio.

---

## Install

**Requirements:** Python 3.10+

```bash
pip install -e .
```

This installs the `dubbing-cli` command and makes `python -m dubbing` available. On
Linux, install `espeak-ng` for the built-in local TTS path. On macOS, the existing
`say`/`afconvert` backend remains available. Piper is selected automatically only
when both the `piper` binary and `PIPER_MODEL` are configured.

Flask is required for the web UI (`dubbing-web`). The `say` backend uses macOS
built-ins and requires no additional Python packages.

---

## Quick-start

Run the bundled acceptance test to confirm everything works:

```bash
python acceptance.py
```

---

## Sample SRT and example output

Given `sample.srt`:

```srt
1
00:00:00,000 --> 00:00:02,500
<emotion:happy>Welcome to Dubbing Studio!

2
00:00:03,000 --> 00:00:06,200
<rate:slow>This pipeline converts subtitles into timed audio segments.

3
00:00:07,000 --> 00:00:09,800
<emotion:excited><rate:fast>The say backend synthesises every line locally without any network calls.

4
00:00:10,200 --> 00:00:13,500
Multilingual support enables dubbing in any target language.

5
00:00:14,000 --> 00:00:17,000
<emotion:calm><pitch:low>Voice cloning requires explicit written consent from the voice owner.
```

Running `python -m dubbing dub sample.srt --backend say` prints (timing may vary with real synthesis):

```
[0–2500] Welcome to Dubbing Studio!
[3000–6200] This pipeline converts subtitles into timed audio segments.
[7000–9800] The say backend synthesises every line locally without any network calls.
[10200–13500] Multilingual support enables dubbing in any target language.
[14000–17000] Voice cloning requires explicit written consent from the voice owner.
```

The `say` backend calls macOS `say` to synthesise each segment. The `TimelineAligner` maps every TTS result onto its original SRT window and computes a per-segment `stretch_ratio` from the real synthesis duration. The assembler (`dubbing.assembler.assemble_timeline`) then renders one timeline-true WAV: each segment is anchored at its SRT start time, gaps between subtitles become silence, audio longer than its window is time-compressed to fit exactly, and audio shorter than its window plays at natural speed with the remainder padded by silence. The output WAV always spans the full subtitle timeline — dubbing `sample.srt` yields a WAV exactly 17.0 seconds long.

**Overlap policy.** Subtitle windows that overlap (two speakers talking at once) are *mixed*, never shifted: each segment stays anchored at its own SRT start time and the overlapping region carries the sum of both signals, clamped to the 16-bit PCM range. The total output duration is always the end of the last subtitle window — overlapping entries can never stretch the timeline.

**Timeline cap.** The renderer rejects any subtitle whose timestamps reach beyond 4 hours (`dubbing.assembler.MAX_TIMELINE_MS`) with a clear error instead of allocating audio for it — a hostile few-hundred-byte SRT carrying a `99:59:59,999` timestamp would otherwise demand ~16 GB of silence. Four hours comfortably covers any feature film; pass `max_timeline_ms` to `assemble_timeline` to raise it deliberately.

**Web service limits.** The web UI caps the *entire* request body via Flask's `MAX_CONTENT_LENGTH` (the 1 MB SRT limit plus a small multipart envelope allowance), so an oversized payload smuggled in any form field is rejected with a JSON 413 before it is ever parsed. Finished jobs are held in a bounded registry — at most `MAX_JOBS` (16) renders and `MAX_JOBS_BYTES` (256 MB) in total, oldest evicted first — so repeated dub requests can never exhaust server memory; download your audio promptly after rendering.

---

Dub a single SRT file to stdout:

```bash
python -m dubbing dub subtitles.srt
```

Write the dubbed audio and the segment plan:

```bash
python -m dubbing dub subtitles.srt --output out/
# Writes out/subtitles.wav  (timeline-true dubbed audio)
#    and out/subtitles.json (segment plan: start_ms, end_ms, stretch_ratio, text)
```

---

## Prosody and emotion tag syntax

Embed tags directly in subtitle text using `<name:value>` notation. Tags are stripped before TTS synthesis and stored on the segment for the backend to consume. The built-in `say` backend genuinely delivers them: `rate` and `emotion` tags set the speaking rate (`say -r`, words per minute), and `pitch` tags shift the pitch base via an embedded `[[ pbas N ]]` speech command.

```
<emotion:happy>    Joyful delivery        → say -r 195
<emotion:sad>      Mournful, slow pacing  → say -r 130
<emotion:excited>  High-energy reading    → say -r 215
<emotion:calm>     Measured, even tone    → say -r 150
<rate:slow>        Stretched cadence      → say -r 130
<rate:fast>        Rapid-fire delivery    → say -r 220
<rate:185>         Explicit WPM           → say -r 185
<pitch:low>        Deeper voice register  → [[ pbas 38 ]]
<pitch:high>       Higher voice register  → [[ pbas 54 ]]
```

An explicit `rate` tag overrides any emotion-derived pacing on the same line.

Tags compose freely on a single line:

```
1
00:00:05,000 --> 00:00:08,500
<emotion:excited><rate:fast>We did it — the pipeline is live!
```

Custom tag names are supported; the backend decides which it honours. Unrecognised tags are stored and passed through without error.

---

## Injecting a custom TTS backend

Subclass `dubbing.backends.base.TTSBackend` and implement `synthesize`:

```python
from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, TTSResult


class MyCloudTTS(TTSBackend):
    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        results = []
        for seg in segments:
            # seg.entry.text  — clean text (tags already stripped)
            # seg.tags        — list[ProsodyTag] for expressive control
            # seg.language    — BCP-47 language code, e.g. "es", "ja"
            audio, duration_ms = call_my_api(seg.entry.text, seg.tags, seg.language)
            results.append(TTSResult(segment=seg, audio_bytes=audio, duration_ms=duration_ms))
        return results
```

Pass your backend to the pipeline:

```python
from dubbing.pipeline import DubbingPipeline

pipeline = DubbingPipeline(backend=MyCloudTTS())
segments = pipeline.run("subtitles.srt")
```

The pipeline calls `synthesize` once per `run()` invocation; batching within your backend is up to you.

---

## Multilingual usage

Pass `--lang` to set the target language for the entire file:

```bash
python -m dubbing dub subtitles.srt --lang es   # Spanish
python -m dubbing dub subtitles.srt --lang ja   # Japanese
python -m dubbing dub subtitles.srt --lang fr   # French
```

The language code is stored on every `Segment.language` field and forwarded to `TTSBackend.synthesize`. The built-in `say` backend selects an installed macOS voice for the language — exact locale matches (`es-MX` → an `es_MX` voice) win over base-language matches (`es` → the first `es_*` voice). If no installed voice supports the requested language, synthesis fails with a clear error listing the languages that are available; the backend never silently dubs in the wrong language.

Per-segment language overrides are not yet supported via SRT tags; use the Python API to construct `Segment` objects directly if per-segment language mixing is required.

---

## Batch CLI

Process multiple SRT files at once using a glob pattern:

```bash
# Dub all SRT files in a directory tree, write JSON results to out/
python -m dubbing batch "content/**/*.srt" --output out/
```

The batch command:
- Expands the glob pattern (recursive by default when `**` is used)
- Runs the pipeline independently on each file
- Writes `<stem>.wav` (timeline-true dubbed audio) and `<stem>.json` (the segment plan) to `--output` for every input
- Prints a summary line per file: `path/to/file.srt: N segment(s)`
- Exits non-zero if no files match the pattern

From Python:

```python
from pathlib import Path
from dubbing.batch import batch_dub
from dubbing.backends.say import SayTTSBackend

results = batch_dub(
    inputs=list(Path("content").rglob("*.srt")),
    backend=SayTTSBackend(),
    output_dir="out/",
    language="es",  # optional target language
)
for path, segs in results.items():
    print(f"{path}: {len(segs)} segments")
```

---

## Local transcription

The transcription boundary has independent MLX and Faster-Whisper backends.
Install the backend appropriate for the host.

For Linux/Windows with an NVIDIA GPU:

```bash
pip install -e '.[transcription-faster]'
dubbing-gpu transcribe recording.m4a \
  --asr-backend faster-whisper \
  --asr-model large-v3-turbo \
  --asr-device cuda \
  --asr-compute-type float16 \
  --output transcript.json
```

`int8_float16` uses less VRAM if another GPU workload must run concurrently.
The `dubbing-gpu` launcher exposes the CUDA runtime libraries installed inside
the active virtual environment; it does not require a system-wide CUDA toolkit.

For Apple silicon:

Install the MLX Whisper backend:

```bash
pip install -e '.[transcription-mlx]'
```

Transcribe an audio or video file to versioned JSON:

```bash
python -m dubbing transcribe recording.m4a \
  --output transcript.json \
  --format json
```

SRT and plain-text renderers use the same transcript model:

```bash
python -m dubbing transcribe recording.m4a --format srt --output transcript.srt
python -m dubbing transcribe recording.m4a --format text --output transcript.txt
```

For long recordings, use resumable chunks. The checkpoint is written atomically
and is accepted only when the source hash, backend identity, and chunk settings
still match:

```bash
python -m dubbing transcribe day.m4a \
  --checkpoint-dir checkpoints/day \
  --chunk-seconds 1800 \
  --output day.json
```

Transcription and diarisation remain independent plugins. They can be composed
when local Sherpa-ONNX models are configured:

```bash
python -m dubbing transcribe conversation.wav \
  --diarize \
  --segmentation-model models/segmentation.onnx \
  --embedding-model models/embedding.onnx \
  --output attributed.json
```

Word and segment timestamps are preserved. Speaker attribution explicitly marks
silence, overlap, and ambiguity instead of inventing a dominant speaker. Audio
and transcripts remain local unless the caller deliberately moves them.

---

## Personal Capture Inbox

The dogfood capture layer turns stable local recordings into reviewable evidence
packages. It hashes and deduplicates source files, records retryable processing
state in SQLite, preserves the verbatim transcript, and can add speaker
diarisation plus segment-level English translations for English, Korean,
Romanian, and Russian.

Install the personal translation dependencies separately from the audio stack:

```bash
pip install -e '.[understanding-nvidia,translation-local]'
```

Scan an inbox on the Gigabyte:

```bash
dubbing-gpu capture scan /srv/dubbing/inbox \
  --workspace /srv/dubbing/personal-capture \
  --asr-backend faster-whisper \
  --asr-model large-v3-turbo \
  --asr-device cuda \
  --asr-compute-type float16 \
  --translate-to en \
  --diarize \
  --segmentation-model models/segmentation.onnx \
  --embedding-model models/embedding.onnx
```

Every package initially stops in `review`. After reviewing the transcript and
supplying any known speaker aliases, approve it explicitly:

```bash
dubbing-gpu capture approve CAPTURE_SHA256 \
  --workspace /srv/dubbing/personal-capture \
  --speaker SPEAKER_00=Artiom
```

Approval emits `giga-event.json` beside the transcript; it does not directly
modify GIGA memory. The event retains source and transcript hashes so a later
ingestion adapter can verify provenance and remain idempotent.

The initial NLLB backend is for private dogfooding. Its checkpoint is
CC-BY-NC-4.0 and is not the eventual commercial translation backend.

The permanent Gigabyte paths and safety boundaries are versioned in
`deploy/gigabyte/personal-capture.json`. The landing inbox is Windows-visible,
while the ledger and derived evidence remain on the WSL filesystem.

---

## Local speaker diarisation

Dubbing Studio can answer “who spoke when?” in source audio and map those turns
onto the existing subtitle/dubbing segment plan without moving or splitting any
subtitle timestamp. The production backend is
[Sherpa-ONNX](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html)
with its public ONNX conversion of pyannote segmentation 3.0 and a public NeMo
TitaNet speaker-embedding model. Model inference is local; no audio is uploaded.

### Install

CPU inference, suitable for ordinary laptops:

```bash
pip install -e '.[diarization]'
```

Python 3.12 is the exercised/recommended interpreter for the current Sherpa
wheels.

The standard PyPI wheel is CPU-only. For an NVIDIA GPU, replace it with the
official Sherpa CUDA wheel matching the installed CUDA/CUDNN runtime. The path
exercised on an RTX 4060 Laptop GPU was:

```bash
pip uninstall -y sherpa-onnx
pip install 'sherpa-onnx==1.13.4+cuda12.cudnn9' \
  -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 nvidia-curand-cu12
```

CUDA runtime libraries must be visible to the dynamic linker before Python
starts. With NVIDIA's pip runtime packages:

```bash
DIAR_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$DIAR_SITE_PACKAGES/nvidia/cublas/lib:$DIAR_SITE_PACKAGES/nvidia/cuda_runtime/lib:$DIAR_SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$DIAR_SITE_PACKAGES/nvidia/cudnn/lib:$DIAR_SITE_PACKAGES/nvidia/cufft/lib:$DIAR_SITE_PACKAGES/nvidia/curand/lib:$DIAR_SITE_PACKAGES/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
```

`--diarization-device` is always explicit: `cpu` or `cuda`. A CPU-only wheel
with `cuda` requested is rejected before inference. CUDA initialization/runtime
errors are surfaced; the application never silently falls back to CPU.

### Acquire the public models

Keep weights outside the checkout, for example under
`$XDG_CACHE_HOME/dubbing-studio/models`:

```bash
DIAR_MODELS="${XDG_CACHE_HOME:-$HOME/.cache}/dubbing-studio/models"
mkdir -p "$DIAR_MODELS"
curl -L -o "$DIAR_MODELS/segmentation.tar.bz2" \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar -xjf "$DIAR_MODELS/segmentation.tar.bz2" -C "$DIAR_MODELS"
curl -L -o "$DIAR_MODELS/nemo_en_titanet_small.onnx" \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx

export DUBBING_DIARIZATION_SEGMENTATION_MODEL="$DIAR_MODELS/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
export DUBBING_DIARIZATION_EMBEDDING_MODEL="$DIAR_MODELS/nemo_en_titanet_small.onnx"
```

These release downloads are not gated and require no account or access token.
Review the licence files shipped with each model before commercial deployment.
Do not commit weights or caches.

### CLI

Machine-readable diarisation, with optional subtitle attribution:

```bash
python -m dubbing diarize source.wav \
  --srt source.srt \
  --diarization-device cuda \
  --num-speakers 2 \
  --output out/source.diarization.json
```

Run source-audio diarisation, subtitle attribution, the existing TTS core, and
timeline assembly in one operation:

```bash
python -m dubbing dub source.srt \
  --source-audio source.wav \
  --backend espeak \
  --diarization-device cuda \
  --num-speakers 2 \
  --output out/
```

The integrated command writes:

- `source.wav`: the timeline-true dubbed output;
- `source.json`: the existing segment plan plus `speaker`, `speakers`,
  `speaker_status`, per-speaker overlap durations, and speech coverage;
- `source.diarization.json`: raw typed turns and backend/model/device metadata.

Pass model paths explicitly with `--segmentation-model` and
`--embedding-model` when the environment variables above are not set.
`--num-speakers` supplies a known exact count. Sherpa also supports automatic
threshold clustering (`--cluster-threshold`, default `0.5`). Its API does not
guarantee min/max-only counts, so `--min-speakers`/`--max-speakers` without an
exact count fail clearly instead of pretending the constraint was honoured.

### Python API

```python
from dubbing.backends.espeak import EspeakTTSBackend
from dubbing.diarization import (
    SherpaOnnxDiarizationBackend,
    SpeakerConstraints,
)
from dubbing.pipeline import DubbingPipeline

diarizer = SherpaOnnxDiarizationBackend(
    segmentation_model="/models/segmentation/model.onnx",
    embedding_model="/models/nemo_en_titanet_small.onnx",
    device="cuda",
)
timed, tts, diarization, attribution = DubbingPipeline(
    EspeakTTSBackend()
).run_full_with_diarization(
    "source.srt",
    "source.wav",
    diarizer,
    constraints=SpeakerConstraints(num_speakers=2),
)
```

`SpeakerTurn` uses integer milliseconds and stable first-appearance labels
(`SPEAKER_00`, `SPEAKER_01`, ...). Intervals are half-open, so a turn beginning
at a subtitle's end cannot leak into it. Segment status is explicit:

| Status | Meaning |
|---|---|
| `attributed` | Exactly one speaker overlaps the window. |
| `no_speech` | No diarised speech overlaps; `speaker` is `null`. |
| `speaker_boundary` | The subtitle crosses sequential speakers; `speaker` is `null`. |
| `overlap` | Speakers talk simultaneously; `speaker` is `null` and all labels/durations are retained. |

Sherpa's current offline result object does not expose calibrated per-turn
confidence, so `confidence` is `null` and `confidence_available` is `false`.
No confidence value is fabricated.

### Media, privacy, and limitations

Native 16 kHz mono 16-bit PCM WAV is read directly. Other audio/video formats
and WAV formats needing resampling require `ffmpeg`; missing/invalid media
produces an actionable error. Inputs longer than four hours are rejected before
model inference.

Speaker embeddings encode voice characteristics and should be treated as
sensitive biometric-like data. This backend keeps embeddings inside the
inference process and persists only anonymous labels/timestamps, but operators
must still protect source recordings, derived JSON, caches, logs, and any future
backend that chooses to persist embeddings. Delete source media and outputs
according to the project's retention policy.

Diarisation accuracy is data-dependent. Clean conversational speech with
distinct voices and known speaker count performs best. Crosstalk, very short
turns, noise, reverberation, music, synthetic voices, and domain/language shift
can cause missed speech or speaker confusion. Anonymous labels identify a
consistent cluster within one file, not a real-world identity and not the same
person across files. Inspect consequential output; this module does not claim a
universal error rate.

### Architecture

`DiarizationBackend` is injectable and returns a typed `DiarizationResult`.
`SherpaOnnxDiarizationBackend` owns media decoding and model inference.
`attribute_timed_segments` joins speaker turns to the studio's existing
`TimedSegment` windows, and `DubbingPipeline.run_full_with_diarization` composes
that join with the existing parse → prosody → TTS → align flow. This keeps
diarisation replaceable and prevents speaker analysis from becoming a second,
divergent dubbing pipeline.

---

## Industry parity

Dubbing Studio implements the core flow shared by commercial dubbing platforms such as ElevenLabs Dubbing and Rask AI. The table below maps each capability to the pipeline component that delivers it.

| Capability | Commercial equivalent | Dubbing Studio component |
|---|---|---|
| **SRT-in → dubbed-audio-out** | Upload subtitle file, receive lip-synced audio | `dubbing-cli dub file.srt --output out/` writes a timeline-true `.wav` spanning the full subtitle timeline; `POST /dub` via the web UI returns a downloadable audio file |
| **Per-segment timing alignment** | Each dubbed phrase snaps to the original subtitle window | `dubbing.aligner.TimelineAligner` computes per-segment stretch ratios from real synthesis durations; `dubbing.assembler` places each segment at its SRT start time, fills gaps with silence, and time-compresses overlong audio to fit its window |
| **Multilingual support** | Target-language selection per project or per segment | `--lang` flag propagates a BCP-47 code to `Segment.language` on every segment; the `say` backend selects an installed voice for the language and errors clearly when none exists |
| **Prosody / emotion tags** | Expressive-speech controls (emotion, pacing, pitch) built into the platform UI | `<emotion:happy>`, `<rate:slow>`, `<pitch:low>` etc. parsed from SRT text by `dubbing.prosody`; the `say` backend delivers them as speaking rate (`-r`) and pitch base (`[[ pbas N ]]`) |
| **Batch mode** | Project-level bulk processing of multiple subtitle tracks | `dubbing-cli batch "content/**/*.srt" --output out/` writes one dubbed `.wav` plus one `.json` plan per matched input |

The `SayTTSBackend` uses macOS `say` for local synthesis (no network, no cost). Swap in any cloud TTS backend — ElevenLabs, Google Cloud TTS, Azure Cognitive Services — by subclassing `TTSBackend` and implementing `synthesize`; the pipeline, aligner, and web UI require no changes.

---

## Voice-cloning consent requirement

**This requirement applies to any backend that clones or mimics a specific person's voice.**

Before synthesising audio that uses a cloned voice you **must**:

1. Obtain **explicit, written consent** from the voice owner before processing begins.
2. Store a durable record of that consent (date, scope, medium) accessible for audit.
3. Limit synthesis to the scope the owner consented to (language, content type, distribution channel).
4. Provide a mechanism for the voice owner to revoke consent; cease synthesis immediately upon revocation.
5. Never use a cloned voice to produce content the owner has not approved (misleading, defamatory, or harmful material).

The built-in `SayTTSBackend` uses the local system voice and does not clone any person's voice; it is exempt from this requirement. Third-party backends that interface with voice-cloning APIs are responsible for surfacing consent controls to callers.

Failure to comply with these requirements may violate applicable laws (e.g. right-of-publicity statutes, GDPR, the EU AI Act) and platform terms of service.
