# Dubbing Studio

A composable pipeline that converts SRT subtitle files into timed audio segments using a pluggable TTS backend. The pipeline parses subtitles, strips and records prosody/emotion tags, calls a TTS backend, and aligns the rendered audio to the original SRT timestamps.

---

## Install

**Requirements:** Python 3.10+

```bash
pip install -e .
```

This installs the `dubbing-cli` command and makes `python -m dubbing` available.

No external dependencies are required for the mock backend. Real TTS backends may add their own.

---

## Quick-start

Run the bundled acceptance demo to confirm everything works:

```bash
python acceptance.py sample.srt --mock
```

---

## Sample SRT and mock-backend output

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
<emotion:excited><rate:fast>The mock backend calculates durations without any network calls.

4
00:00:10,200 --> 00:00:13,500
Multilingual support enables dubbing in any target language.

5
00:00:14,000 --> 00:00:17,000
<emotion:calm><pitch:low>Voice cloning requires explicit written consent from the voice owner.
```

Running `python -m dubbing dub sample.srt` prints:

```
[0–2500] Welcome to Dubbing Studio!
[3000–6200] This pipeline converts subtitles into timed audio segments.
[7000–9800] The mock backend calculates durations without any network calls.
[10200–13500] Multilingual support enables dubbing in any target language.
[14000–17000] Voice cloning requires explicit written consent from the voice owner.
```

The mock backend assigns `duration_ms = len(clean_text) * 60`. The `TimelineAligner` always maps the TTS duration onto the original SRT window, so output timestamps match the SRT exactly. Prosody tags are stripped before synthesis and forwarded to the backend as `ProsodyTag` objects on each `Segment`.

---

Dub a single SRT file to stdout:

```bash
python -m dubbing dub subtitles.srt
```

Write the segment plan as JSON:

```bash
python -m dubbing dub subtitles.srt --output out/
# Writes out/subtitles.json
```

---

## Prosody and emotion tag syntax

Embed tags directly in subtitle text using `<name:value>` notation. Tags are stripped before TTS synthesis and stored on the segment for the backend to consume.

```
<emotion:happy>    Joyful delivery
<emotion:sad>      Mournful, slow pacing
<emotion:excited>  High-energy reading
<emotion:calm>     Measured, even tone
<rate:slow>        Stretched cadence
<rate:fast>        Rapid-fire delivery
<pitch:low>        Deeper voice register
<pitch:high>       Higher voice register
```

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

The language code is stored on every `Segment.language` field and forwarded to `TTSBackend.synthesize`. The mock backend ignores it; real backends use it to select voice and phoneme rules.

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
- Prints a summary line per file: `path/to/file.srt: N segment(s)`
- Exits non-zero if no files match the pattern

Output JSON files are written to `--output` (defaults to current directory), one file per input with the same stem and a `.json` extension.

From Python:

```python
from pathlib import Path
from dubbing.batch import batch_dub
from dubbing.backends.mock import MockTTSBackend

results = batch_dub(
    paths=list(Path("content").rglob("*.srt")),
    backend=MockTTSBackend(),
    output_dir="out/",
)
for path, segs in results.items():
    print(f"{path}: {len(segs)} segments")
```

---

## Voice-cloning consent requirement

**This requirement applies to any backend that clones or mimics a specific person's voice.**

Before synthesising audio that uses a cloned voice you **must**:

1. Obtain **explicit, written consent** from the voice owner before processing begins.
2. Store a durable record of that consent (date, scope, medium) accessible for audit.
3. Limit synthesis to the scope the owner consented to (language, content type, distribution channel).
4. Provide a mechanism for the voice owner to revoke consent; cease synthesis immediately upon revocation.
5. Never use a cloned voice to produce content the owner has not approved (misleading, defamatory, or harmful material).

The built-in `MockTTSBackend` generates no real speech and is exempt from this requirement. Third-party backends that interface with voice-cloning APIs are responsible for surfacing consent controls to callers.

Failure to comply with these requirements may violate applicable laws (e.g. right-of-publicity statutes, GDPR, the EU AI Act) and platform terms of service.
