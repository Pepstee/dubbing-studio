# Dubbing Studio

A composable pipeline that converts SRT subtitle files into timed audio segments using a pluggable TTS backend. The pipeline parses subtitles, strips and records prosody/emotion tags, calls a TTS backend, and aligns the rendered audio to the original SRT timestamps.

---

## Install

**Requirements:** Python 3.10+

```bash
pip install -e .
```

This installs the `dubbing-cli` command and makes `python -m dubbing` available.

Flask is required for the web UI (`dubbing-web`). The `say` backend uses macOS built-ins and requires no additional packages.

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
