# Multilingual long-video transcription control plane

## Operational status

This control plane is implemented and fail-closed, but it is **not production-certified**.
The existing personal-capture review and GIGA outbox services remain in place. New packages
carry a transcript quality report, and technical failure states cannot be approved or emitted.

The authoritative transcript always remains in its original language. Translation is a
separate derivative whose rows bind the source segment ID, source text hash, timestamps and
speaker. A translation never replaces original text.

## One local processing command

```bash
dubbing-long-transcribe \
  "/path/to/long-video.mov" \
  --output "/path/to/private-workspace/recording-id" \
  --backend whisperkit \
  --start-server \
  --whisperkit-cli /opt/homebrew/bin/whisperkit-cli \
  --model-path "/path/to/existing/openai_whisper-model"
```

Preview the source-bound adaptive plan without loading a model:

```bash
dubbing-long-transcribe \
  "/path/to/long-video.mov" \
  --output "/path/to/plan" \
  --dry-run
```

The production path probes media, preserves the selected audio stream's channel count and
sample rate, selects silence-aware 1–8 minute chunks, applies deterministic overlap
reconciliation, checkpoints every span atomically and retries only rejected spans. The
WhisperKit local-server process remains alive across all chunks, so the model is loaded once.
MLX remains an experimental fallback.

## One evaluation command

```bash
dubbing-evaluate-transcript \
  --manifest benchmarks/fixtures/lesson-2026-08-01-193908/manifest.json \
  --output /tmp/lesson-evaluation.json
```

Exit code `2` means the candidate was structurally readable but failed semantic quality.
The evaluator computes exact Unicode token WER and character error rate. True per-window and
per-language WER require timestamped, language-labelled reference turns; the current
MacWhisper reference has neither, so the report states that limitation instead of inventing
window scores. Script-level error buckets are still reported.

## Quality states

| State | Meaning | Approval/outbox |
|---|---|---|
| `PASS` | No configured technical or semantic defect detected | technically eligible; explicit review still applies |
| `PASS_WITH_UNCERTAIN_SPANS` | Original text contains explicit unresolved turns | blocked pending resolution/review |
| `REPROCESS_REQUIRED` | Decoder loop, exhausted fallback, failed span, impossible rate or timestamp failure | blocked |
| `HUMAN_REVIEW_REQUIRED` | Evidence such as unexplained long gaps needs a human/VAD decision | blocked without explicit review notes |
| `FAILED` | Empty or malformed usable output | blocked |

Observed repetitions such as `Loops` ×109, `tree` ×23, `second` ×16, `Mm-hmm` ×20 and
`됐다` ×33 are adversarial regression tests. Scattered legitimate duplicate replies do not
count as a loop; consecutive runs and aggregate pathology thresholds do.

## Provider boundaries

- **WhisperKit:** official `argmax-cli`/`whisperkit-cli` OpenAI-compatible local server,
  bound to loopback. Model tree hash, CLI version, configuration, timestamps and available
  decoder diagnostics are retained.
- **Faster-Whisper:** persistent local model instance and provider diagnostics.
- **MLX Whisper:** experimental fallback with its configured fallback schedule preserved.
- **Diarization:** pyannote Community-1 is the preferred local reference backend; Sherpa is
  retained as the lightweight fallback. pyannote Precision-2 has a separate disabled remote
  boundary.
- **Cloud ASR adjudication:** Deepgram Nova-3 Multilingual, AssemblyAI Universal-2,
  ElevenLabs Scribe v2 and OpenAI `gpt-4o-transcribe-diarize` are represented by offline-testable
  adapters. `cloud_allowed=false` is the default. Only explicitly authorized 20–60 second
  failed spans, never an entire private recording, can cross the boundary.

Cloud authorization requires all of the following: recording hash, operator authorization
ID, named provider and a credential already configured in the local environment. Receipts
bind provider/model, span hash, retention configuration, request receipt and reported cost.
No account, payment method, API key or private upload is created by the repository.

## Diarization and identity

Diarization asks whether two regions contain the same voice. Identity asks whose voice that
is. They remain separate.

The voiceprint registry accepts multiple consented reference clips per person, hashes every
clip, applies an explicit similarity threshold and inter-candidate margin, and returns
`UNKNOWN` rather than guessing. Manual corrections append lineage. DER, JER and
speaker-attributed WER evaluators run only when suitable reference labels exist.

## Current measured fixture result

The 4,987.833-second fixture planned into 19 adaptive chunks: 16 silence boundaries, two
maximum-duration boundaries and one end boundary. With the already-installed WhisperKit
`openai_whisper-small` model:

- end-to-end control-plane runtime: **227.097 seconds**;
- provisional-reference WER: **55.3433%**;
- CER: **51.1559%**;
- two spans exhausted local candidates;
- 42 segments remained explicitly uncertain;
- Cyrillic and Hangul reference tokens were not recovered.

Therefore WhisperKit-small is **rejected for promotion** despite being fast. The result is
saved only as a local ignored benchmark artefact under
`output/control-plane-whisperkit-small-v3/`.

The authorized 4-bit `openai_whisper-large-v3-v20240930_626MB` benchmark and targeted
recovery are complete. The repaired candidate has **42.0543% WER**, **37.6801% CER**, no
failed spans and no decoder-fallback exhaustion. It remains fail-closed at
`PASS_WITH_UNCERTAIN_SPANS`: 39 turns totalling 30.5 seconds still need a human decision.

The operator subsequently corrected 38 of those turns and classified one impossible 20 ms
segment as `no_speech`. The reviewed result has no uncertain spans and passes the pathology
contract. Against the provisional, non-human MacWhisper reference it measures **43.6323%
WER** and **38.9265% CER**—1.5781 and 1.2464 percentage points worse than the pre-review
candidate. This disagreement cannot be interpreted as operator error because the reference is
not ground truth. Structural promotion passes; accuracy and speaker-attribution certification
remain blocked pending a small timestamped human-ground-truth sample. No GIGA event was emitted.

## Local uncertain-span review

Build a source- and transcript-hash-bound package without copying the full private recording:

```bash
dubbing-transcript-review-package \
  "/path/to/source.mov" \
  --transcript "/path/to/result.json" \
  --output "/path/to/private-review-package"
```

Then start the loopback-only reviewer:

```bash
dubbing-transcript-review \
  --package "/path/to/private-review-package" \
  --port 7444
```

Open `http://127.0.0.1:7444`. Every uncertain segment has a short WAV clip with bounded
context. Approve unchanged text, save an explicit correction, or leave it unclear. Decisions
are written atomically and are resumable. A segment that contains no transcribable speech can
be removed with an explicit `no_speech` lineage decision. Export is blocked while any item is pending. A span
explicitly marked unclear remains uncertain in the local export, so the quality contract still
blocks approval and GIGA admission. Every export retains correction lineage, re-runs the
transcript quality contract and creates no GIGA event.
