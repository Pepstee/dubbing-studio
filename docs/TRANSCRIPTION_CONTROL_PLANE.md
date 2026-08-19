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
count as a loop. Quality policy v3 maps normalized tokens back to source segments, searches
primitive repeated phrases up to 20 tokens, and emits exact failure timestamps for targeted
retry. Length-sensitive thresholds allow six short conversational acknowledgements while
still rejecting eight identical tokens, six short phrases, or four longer phrases covering
at least 24 repeated tokens. The observed seven-token `I don't know what to do` decoder loop
is a permanent regression fixture.
Quality policy v3 also rejects known multilingual Whisper outro/subtitle boilerplate when the
decoder simultaneously reports a no-speech probability of at least 0.6. This catches shared
training-data hallucinations that can appear identically in two Whisper-family models without
rejecting the same ordinary phrase when speech evidence is strong.

Production targeted repair uses two pinned, local models. `large-v3-turbo` remains the fast
primary decoder; full `large-v3` is loaded lazily only for rejected 20–60 second spans. The
coordinator evaluates every eligible primary and independent candidate, retains their text,
quality report, model identity and language in the receipt, and requires token agreement of at
least 0.75 before a replacement becomes clean. A same-model retry cannot self-corroborate. If
the independent model is unavailable or unhealthy the span becomes a failed marker; if healthy
models disagree, the transcript receives an explicit disagreement marker while every candidate
remains in the receipt. A plausible-looking model guess never becomes authoritative text merely
because it carries an uncertainty flag. Both outcomes are
blocked by the existing approval and GIGA admission gates.
Because the rejected text may itself be a wrong-language hallucination, its script only
prioritizes the retry order; it never removes English, Russian, Romanian or Korean from the
search. Agreement is calculated only between candidates decoded under the same language
constraint, preventing a high-confidence wrong-language candidate from being compared with a
different decoding condition.

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
be removed with an explicit `no_speech` lineage decision. Export is blocked while any item is
pending. A span explicitly marked unclear remains uncertain in the local export, so the quality
contract still blocks approval and GIGA admission. Every export retains correction lineage,
re-runs the transcript quality contract and creates no GIGA event.

## Automatic local adjudication (default)

Manual calibration is optional, not an operational prerequisite. Compare the reviewed primary
against one or more independently executed, source-bound local candidates:

```bash
dubbing-adjudicate-local \
  --source "/path/to/source.mov" \
  --primary "/path/to/reviewed-result.json" \
  --comparator "/path/to/independent-local-result.json" \
  --output "/path/to/local-adjudication.json"
```

Comparators that fail the transcript quality contract remain visible as negative evidence but
cannot vote in consensus. The policy computes symmetric whole-document and one-minute-window
token agreement. Low-agreement windows fail closed; model consensus is explicitly not represented
as human ground truth. The command never emits a GIGA event. A consensus pass only makes the
package eligible for the existing separate explicit approval gate.
Supplying `--source` hash-binds the media and recomputes every quality report with local silence
evidence; omit it only when consuming an already-bound review package with a matching sidecar.

For an MLX comparator, deterministic fail-fast decoding avoids the expensive and increasingly
hallucination-prone temperature escalation used by the legacy baseline:

```bash
dubbing-long-transcribe "/path/to/source.mov" \
  --output "/path/to/mlx-comparator" \
  --backend mlx \
  --model mlx-community/whisper-large-v3-turbo \
  --mlx-temperature 0
```

For an isolated NVIDIA/Windows comparator, `scripts/run_faster_whisper_gpu.py` loads one local
CTranslate2 model persistently, binds an audio derivative to the original media SHA-256, records
GPU/runtime provenance and emits `dubbing.transcription.v1`. Use temperature 0 and evaluate both
VAD-on and VAD-off candidates; neither is eligible until the pathology gate passes. If their
failure modes are complementary, `dubbing-fuse-local-transcripts` replaces only detected
pathological intervals, marks admitted alternate spans uncertain and records hash lineage.

The 83-minute RTX 4060 fixture established that whole-file speed is not quality: the no-VAD run
decoded in 120 seconds but looped a Japanese phrase 73 times, while VAD removed too much quiet
speech. Targeted fusion produced `PASS_WITH_UNCERTAIN_SPANS` in 711 segments with two uncertain
spans. Model agreement remains evidence, not ground truth, and no command in this workflow emits
a GIGA event.

## Optional human-ground-truth calibration

For a formal accuracy certificate, build a deterministic 8–12 minute
stratified calibration set from a reviewed transcript:

```bash
dubbing-calibration-package \
  "/path/to/source.mov" \
  --transcript "/path/to/reviewed-result.json" \
  --output "/path/to/private-calibration-package"
```

Start the loopback-only turn reviewer separately from the uncertain-span reviewer:

```bash
dubbing-calibration-review \
  --package "/path/to/private-calibration-package" \
  --port 7445
```

The reviewer pre-fills timestamped candidate turns but requires the operator to verify exact
text, language, approximate boundaries and anonymous `Speaker A`/`Speaker B` labels. Intelligible
speech cannot be saved as `UNKNOWN`; overlapping speech and `[unclear]` remain explicit. Export
is blocked until every clip is reviewed, records full correction lineage and never emits a
GIGA event. Languages absent from the source remain declared coverage gaps rather than being
fabricated into the sample.
