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

The production path probes media and preserves every audio stream as discrete channels in the
lossless PCM working chunk. It preserves the highest source sample rate, selects silence-aware
1–8 minute chunks, applies deterministic overlap
reconciliation, checkpoints every span atomically and retries only rejected spans. The
WhisperKit local-server process remains alive across all chunks, so the model is loaded once.
MLX remains an experimental fallback.

## One evaluation command

```bash
dubbing-evaluate-transcript \
  --manifest benchmarks/fixtures/lesson-2026-08-01-193908/manifest.json \
  --output /tmp/lesson-evaluation.json
```

## Historical English/Russian canary

The real lesson canary is separate from the synthetic multilingual benchmark. It binds four
original videos and their provisional MacWhisper transcripts by SHA-256. Lesson 6 is the only
development recording, lessons 7 and 9 are untouched holdouts, and lesson 8 is the long-duration
stress recording. The execution configuration is frozen before a holdout can run.

Run the development recording first from the original video:

```bash
dubbing-canary \
  --manifest benchmarks/fixtures/lessons-en-ru-canary-v1/manifest.json \
  --output output/canary-lessons-en-ru-v1 \
  --lesson lesson-06 \
  --backend whisperkit \
  --start-server
```

After the development configuration is frozen, run a holdout with the same output directory and
backend arguments. The command owns chunking, starts one persistent local model process, reuses
source-bound checkpoints, evaluates pathology and provisional-reference drift, and always records
`giga_admission_allowed=false`.

An operational canary pass is deliberately narrower than an accuracy claim. The historical text
has no trusted timestamps or speaker labels and is not human ground truth, so it cannot certify
window-level WER, DER, JER or speaker-attributed WER. `PASS_WITH_UNCERTAIN_SPANS` also fails the
canary: unresolved text is useful evidence but is not ready output.

Exit code `2` means the candidate was structurally readable but failed semantic quality.
The evaluator computes exact Unicode token WER and character error rate. True per-window and
per-language WER require timestamped, language-labelled reference turns; the current
MacWhisper reference has neither, so the report states that limitation instead of inventing
window scores. Script-level error buckets are still reported.

Language-sweep reports use the claim-scoped accuracy gate rather than an ambiguous aggregate
`target_passed` boolean. Generate scoped validation/test reports, then evaluate the complete
portfolio separately:

```bash
python -m dubbing.evaluation.language_sweep \
  --fixture /path/to/fixture.json \
  --sweep /path/to/sweep.json \
  --output /path/to/scoped-report.json

python -m dubbing.evaluation.accuracy_gate \
  --report /path/to/clean-holdout-report.json \
  --report /path/to/noisy-code-switch-report.json \
  --report /path/to/natural-long-form-report.json \
  --output /path/to/accuracy-portfolio.json
```

`measurement_gate`, `benchmark_claim` and `production_claim` are deliberately separate. Every
required language must pass observed WER and CER, minimum evidence counts and the finite-sample
guard. Reference-informed selectors are never benchmark-eligible. A single report always leaves
production evidence insufficient, even when its scoped benchmark passes.

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

Rejected spans also have a bounded, source-preserving audio-candidate stage. The production
set is raw audio, a multichannel downmix and up to four discrete source channels. The engine
also exposes an opt-in speech-band/dynamic-normalization candidate, but it is not enabled in the
Gigabyte deployment. No candidate overwrites the source or a raw
checkpoint. Each candidate is hash-bound with its exact FFmpeg processing policy in the retry
receipt. Raw cross-model consensus always wins. A processed candidate may rescue a span only
when raw audio has no valid consensus and both independent ASR models corroborate that same
candidate. If independently corroborated processed candidates disagree, the span stays
explicitly uncertain. Multiple language decodes of one channel remain one audio candidate and
cannot manufacture cross-channel corroboration. Broad FFT denoising and aggressive speech
expansion are intentionally excluded. The conservative normalization candidate also failed its
promotion loop: under identical beam-5 settings it reduced turbo accuracy from 57.58% to 48.48%
and full-large-v3 accuracy from 58.59% to 47.47% on 26 human-corrected difficult lesson spans.

Sparse-speech repair adds a semantic speech-region stage before re-decoding. The production
Gigabyte provider is the local Silero VAD already bundled with Faster-Whisper, so it needs no
new checkpoint or network access. A strict `0.2` pass supplies precise regions; a `0.1`
sensitive pass contributes only regions that do not overlap any strict region. Micro-regions
are capped at eight seconds and decoded separately by both ASR models. Every plan, pass,
source-time interval and candidate remains in the receipt. Uncovered intervals of at least one
second, exhausted regions and model disagreement stay explicit uncertainty; VAD is never
allowed to erase possibly quiet speech merely because it failed to detect it.
Text agreement alone is insufficient for micro-regions: when both providers expose acoustic
confidence, their lower confidence must be at least `0.35`. This prevents exact agreement on
low-confidence phonetic guesses such as `Oh` from becoming clean transcript text.
Local consensus must also contain at least two normalized tokens. A one-token utterance such as
Korean `어?`/`응?` is too vulnerable to phonetic-neighbor errors to become authoritative merely
because two related ASR models agree; it remains explicit uncertainty for later adjudication.

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

### Targeted cloud adjudication

Cloud adjudication is a separate, explicit stage after the local long-form run. The operator
does not cut audio manually. The command discovers only segments still marked uncertain,
merges and bounds them, extracts the minimum 20-second source context needed by the cloud
boundary, and never uploads the full recording. The source file, local transcript and
authorization must all carry the same SHA-256 digest.

The first live transport is the official OpenAI audio-transcriptions endpoint using
`gpt-4o-transcribe-diarize`. Its API key must already exist in `OPENAI_API_KEY`; the command
never accepts a credential argument. Returned segments must carry timestamps, and every request
produces a provider/model/configuration/span/candidate-hash receipt. The amended whole transcript
is evaluated again under the normal quality contract. Malformed output, an empty target, any
remaining uncertainty or a non-PASS quality result stays `FAIL_CLOSED`. The report always keeps
`giga_admission_allowed=false`; normal review and approval remain separate gates.

```bash
dubbing-adjudicate-cloud \
  /path/to/source.mov \
  --local-result /path/to/transcription-adaptive/result.json \
  --output /path/to/transcription-cloud \
  --recording-sha256 <exact-source-sha256> \
  --operator-authorization-id <recording-specific-id> \
  --provider openai \
  --cloud-allowed
```

Omitting `--cloud-allowed`, omitting the locally configured credential, changing the recording,
or presenting a transcript from another source terminates before network upload.

### Month-one cloud teacher programme

The full-recording cloud teacher is a separate, temporary learning programme. It does not
weaken the failed-span-only adjudication boundary above. During an explicitly dated period it
runs ElevenLabs Scribe v2 beside the completed local transcript, automatically constructs
bounded lossless FLAC speech packets, and checkpoints each provider response. The original
local transcript and the cloud transcript remain separate immutable candidates.

Cloud packets are silence-compacted before upload. The existing local Faster-Whisper Silero
detector supplies both strict and sensitive speech regions; compaction uses the inclusive union
of both passes plus every local transcript interval. Each retained interval receives configurable
context padding. Speech separated by at most 15 seconds remains one continuous source interval so
natural conversational pauses and speaker evidence are not cut. Longer gaps create separate
packets. The production policy requires exactly one contiguous source slice per packet and permits
at most one hour per packet; it never makes unrelated scenes artificially adjacent for provider
diarization.

Every retained slice records compact and original start/end times. Provider timestamps are
accepted only when the complete provider segment lies within its retained slice, then restored to
the untouched recording timeline. A provider timestamp outside the compacted interval fails
closed. Speaker labels are packet-namespaced; compaction never implies that provider speaker IDs
are stable across packets, while the original-audio local diarization remains separate.
If neither sensitive VAD nor the local transcript finds any speech, compaction fails open on cost
and retains the entire recording. It never turns an uncertain no-speech decision into a zero-byte
upload.

The programme policy binds the provider, 31-day maximum window, operator authorization,
privacy attestation, audio ceiling, estimated-cost ceiling, chunk size and training thresholds.
Before the first upload, the operator must disable ElevenLabs model-improvement data use in the
provider account, change `model_improvement_opt_out_attested` to `true`, and configure
`ELEVENLABS_API_KEY` locally. The key is never accepted as a command-line argument or written to
a receipt. The committed policy starts fail-closed until that account-side action is attested.

```bash
dubbing-cloud-teacher /path/to/full-recording.wav \
  --local-result /path/to/local/result.json \
  --policy deploy/cloud_teacher/month-one-2026-08.json \
  --programme-state /private/state/cloud-teacher-usage.json \
  --output /private/outputs/cloud-teacher/<recording-id>
```

The command is resumable and source-bound. It records provider/model/configuration, compact audio
hash, reversible mapping hash, request receipt, candidate provenance and estimated cost based on
uploaded—not original—duration. Replaying a completed packet does not upload or bill it again. A
policy or mapping change invalidates checkpoint reuse; exhausted programme audio or cost stops
before the next upload. `compaction-plan.json` reports retained, removed and uploaded durations,
estimated reduction, fallback reason and the known dual-miss limitation: sensitive VAD plus local
transcript evidence reduces but cannot mathematically eliminate missed quiet speech.

Cost enforcement is transactional per programme-state file. An exclusive POSIX lock serializes
budget decisions and uploads. Each packet's duration and estimated cost are reserved durably with
status `uploading` before network transmission, then promoted to `completed` only after a bound
provider checkpoint exists. A checkpoint that survived a ledger-write crash repairs the ledger on
restart. An `uploading` reservation without a checkpoint is treated as an unknown potentially paid
outcome: its cost remains counted and automatic re-upload is denied pending reconciliation.

Training data is deliberately stricter than transcript review. Cloud-only text is never a
label. A span becomes `CONSENSUS_SILVER` only when local and cloud text meet both the configured
WER and CER bounds, contains at least three tokens, is 0.4–30 seconds, contains no explicit
uncertainty or overlapping speakers, stays within English/Russian/Romanian/Korean, and does not
carry weak cloud acoustic confidence. All other spans are retained in `excluded.jsonl` as an
active-learning/evaluation set. Splits are recording-level, so adjacent spans from one recording
cannot leak across train, validation and locked test. These labels are silver evidence—not human
ground truth—and can never emit a GIGA event.

OpenAI remains the independent second opinion for disputed 20–60 second spans through
`dubbing-adjudicate-cloud`; the entire recording is not sent to a second provider by default.

## Diarization and identity

Diarization asks whether two regions contain the same voice. Identity asks whose voice that
is. They remain separate.

Long recordings are diarized in resumable chunks, but speaker labels are recording-global.
The preferred local reference backend is pyannote Community-1. Its runtime is selectable only
from an absolute local model path, so a missing gated model cannot silently trigger a download.
Sherpa remains the fully offline fallback and the independent TitaNet embedding provider. Its
automatic distance threshold is explicit deployment configuration rather than a hidden library
default.

Each chunk extracts one TitaNet embedding per anonymous speaker from speech that does not overlap
another detected speaker. An overlap-safe agglomerative complete-link clusterer reconciles those
embeddings across every chunk and may repair same-chunk fragmentation only when the underlying
turns never overlap. Complete-link scoring prevents a weak embedding bridge from joining two
voices; temporal overlap is a hard cannot-link constraint; an ambiguity margin applies when a
candidate competes with a temporally incompatible voice. Missing or ambiguous evidence stays as
a separate anonymous label. The result preserves overlapping turns and carries the full mapping,
merge evidence and ambiguity evidence in its provenance receipt. `CHUNK_000N_SPEAKER_XX` is
therefore checkpoint evidence only; review and transcript attribution receive stable
`SPEAKER_XX` labels across the recording.

This is anonymous consistency, not automatic identity. No person is named from embedding
similarity alone. The fallback's current evidence-selected settings are a Sherpa clustering
distance threshold of `0.85`, a TitaNet complete-link cosine threshold of `0.80`, and a `0.05`
incompatible-candidate margin. All are source-bound checkpoint configuration; changing them or
the embedding provider invalidates reuse.

The voiceprint registry accepts multiple consented reference clips per person, hashes every
clip, applies an explicit similarity threshold and inter-candidate margin, and returns
`UNKNOWN` rather than guessing. Manual corrections append lineage. DER, JER and
speaker-attributed WER evaluators run only when suitable reference labels exist.

The recording-global reconciliation gate is reproducible with
`scripts/verify_diarization_reconciliation.py`. It derives a multi-chunk fixture from one
source-bound real speaker interval, runs diarization twice, and emits zero-collar single-speaker
DER/JER, cross-chunk identity coverage, checkpoint replay equality, hashes, configuration and
provenance. This is an engineering regression gate; it does not substitute for natural
multi-speaker human labels.

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
