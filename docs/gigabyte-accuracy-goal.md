# Gigabyte local ASR accuracy goal

## Promotion gate

The target is at least 90% word accuracy (WER no greater than 10%) on timestamped,
human-corrected English, Russian, Romanian, and Korean evidence. Model agreement and
provisional transcripts are not ground truth. No result from this work may emit a GIGA
admission event until the gate passes.

## Claim-scoped replacement gate (Project 7)

The legacy tables below retain their historical point estimates, but their `PASS` labels are
superseded. An aggregate score above 90% is not an accuracy certificate. New evaluations use
`dubbing.accuracy-claim-gate.v1` and expose three non-interchangeable decisions:

1. `measurement_gate`: this exact fixture, every required language, WER, CER, timestamps,
   semantic quality and minimum sample size;
2. `benchmark_claim`: the measurement gate plus a deployable selector, human ground truth and
   an untouched holdout;
3. `production_claim`: a portfolio requiring clean holdout, noisy code-switch stress, complete
   natural long-form recordings, overlapping speech, speaker-attributed WER/DER/JER, speech
   coverage and uncertainty burden. A single fixture can never pass this claim.

The finite-sample guard reports a one-sided 95% Wilson error bound in addition to the observed
WER/CER. It is explicitly a conservative promotion guard, not a binomial confidence interval
for WER; insertions above the reference count force the maximum bound.

Re-evaluating the frozen reference-independent selector demonstrates the correction:

| Evidence | Aggregate word accuracy | Failing evidence | New status |
|---|---:|---|---|
| Validation | 92.59% | Romanian 89.98%; Romanian/Korean guarded lower bounds 87.74%/87.53% | `FAIL` |
| Untouched test | 91.96% | Korean 87.04%; Romanian/Korean guarded lower bounds 89.55%/83.83% | `FAIL` |

The source-bound receipt is
`benchmarks/fixtures/transcription-accuracy-portfolio-v1.json`. The production portfolio is
therefore `FAIL`, not “90% achieved,” and GIGA admission remains false.

## Hardware and model

- GPU: RTX 4060 Laptop GPU, 8 GB VRAM.
- Model: `deepdml/faster-whisper-large-v3-turbo-ct2`.
- Model binary SHA-256:
  `e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da`.
- Full-recording INT8 runtime: 120.047 seconds for 4,987.796 seconds of audio (RTF
  0.024068) without VAD; 81.687 seconds (RTF 0.016377) with VAD.

The hardware and speed gates pass. The strict per-language and production accuracy gates
do not.

## Human-ground-truth language fixture

A pinned `google/fleurs` validation fixture now supplies 100 complete, human-transcribed
utterances: 25 each in English, Russian, Romanian, and Korean. The dataset revision is
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`; every WAV has an individual SHA-256 and
exact sample-count timestamp, and the complete source binding is
`9a46879d4f0c80e63d478ccbb2ed2c9c6d5c6e94b762e7cdf942572323e53e59`.

This is defensible evidence for clean read-speech language coverage. It does not represent
long-form, noisy, conversational, or code-switched audio, so it cannot certify the production
pipeline by itself.

The best installed-model result is a frozen, reference-independent routing policy. It uses
the 200 ms VAD candidate by default, always retries detected Korean at threshold 0.35, and
uses maximum decoder log probability across the bounded candidate set for detected Romanian.
No selected clip required a non-zero-temperature fallback:

| Language | Reference words | Edits | WER | Word accuracy | CER | Gate |
|---|---:|---:|---:|---:|---:|---|
| English | 501 | 22 | 4.39% | 95.61% | 2.47% | PASS |
| Russian | 455 | 24 | 5.27% | 94.73% | 1.24% | PASS |
| Romanian | 579 | 56 | 9.67% | 90.33% | 4.18% | PASS |
| Korean | 409 | 40 | 9.78% | 90.22% | 7.77% | PASS |
| **Aggregate** | **1,944** | **142** | **7.30%** | **92.70%** | **3.39%** | **PASS: clean validation only** |

The four complete candidate passes took 426.36 seconds. The selected candidates contain no
blank hypotheses, timestamp disorder, adjacent duplicate segments, four-token repetition
runs, or exhausted
fallbacks. CER is micro-averaged across utterances; it is diagnostic and does not replace or
weaken the explicit WER gate. The exact machine-readable verdict is in
`benchmarks/fixtures/fleurs-validation-25x4/verdict.json`. Temperature-zero and monolingual
controls both scored 90.59% aggregate but left one Korean clip blank; disabling multilingual
mode made no accuracy difference. Disabling word timestamps produced byte-for-byte identical
recognized text and the same WER/CER, reducing runtime only from 110.047 to 107.484 seconds;
it is rejected because it removes required timing evidence without improving accuracy.
Enabling previous-text conditioning also produced identical recognized text and WER/CER
(107.0 seconds, no repetition issues), so decoder context within these complete utterances is
not the missing capacity either. A finite, predeclared VAD speech-pad study tested 100, 150,
200, 250, and the default 400 ms; 200 ms was the clear single-candidate optimum and is now
frozen. That candidate misses the Romanian gate by one edit and the Korean gate by three
edits. A merged
six-variant confidence selector reached 92.23% aggregate but still only 89.98% Romanian and
89.00% Korean. The human-reference oracle reached 93.62% and passed all four languages, but
it is explicitly non-deployable and is not treated as ground truth or promotion evidence.
The final language-conditioned policy passes all validation language gates without reference
access. Its exact policy and candidate hashes were frozen in
`frozen-language-retry-policy.json` before holdout access.

## Untouched holdout result

The frozen policy was run unchanged on 25 test clips per language from the same pinned
FLEURS revision. The dataset server could not serve test rows because its Parquet scan
exceeded the 300 MB service limit, so a selection-only amendment was committed before audio
extraction or inference: the first 25 WAV members in each pinned test archive were streamed
and joined to the pinned TSV. Model, decoding, routing, metrics, and thresholds remained
unchanged.

| Language | Reference words | Edits | WER | Word accuracy | CER | Gate |
|---|---:|---:|---:|---:|---:|---|
| English | 542 | 33 | 6.09% | 93.91% | 3.19% | PASS |
| Russian | 475 | 29 | 6.11% | 93.89% | 1.39% | PASS |
| Romanian | 617 | 52 | 8.43% | 91.57% | 2.86% | PASS |
| Korean | 355 | 46 | 12.96% | 87.04% | 3.86% | **FAIL** |
| **Aggregate** | **1,989** | **160** | **8.04%** | **91.96%** | **2.62%** | **FAIL: per-language gate** |

All structural, repetition, fallback, and word-timestamp checks pass, but aggregate accuracy
cannot override the Korean failure. The receipt is hash-bound in
`benchmarks/fixtures/fleurs-test-25x4/verdict.json`; GIGA admission remains false. This
holdout is now exposed and cannot be used for further policy tuning.

A reference-informed diagnostic, retained only as an error-analysis ceiling, found that 13
of the 25 Korean candidates already have exactly the reference non-whitespace character
sequence. Fourteen of the 46 Korean word edits on those clips are spacing-only; repairing
only those would produce 90.99% Korean word accuracy. This does not validate a spacing
provider and cannot be used as holdout promotion evidence. It only justifies evaluating the
guarded provider on validation before freezing a revised policy for a different holdout.

Two validation-only Korean controls have also been rejected. A fixed orthography/spacing
prompt reduced Korean accuracy from 90.22% to 87.29%. Beam 10 reached 89.73%, also below the
beam-5 baseline. Neither is eligible for a new holdout.

## Human evidence discovered

The operator review package contains 26 conservative corrected/no-speech clips after
excluding 13 punctuation-only, explicitly unclear, or truncated corrections. It has 99
reference word tokens: English 2 clips, Russian 12, Korean 5, mixed-language 6, and one
no-speech clip. Romanian is absent.

The correction text is human-verified. Its timestamps are inherited from the ASR output
and were not independently corrected. Several corrected texts imply impossible speaking
rates inside their nominal segment interval. The fixture is therefore useful for hard-clip
text-recovery development, but it cannot certify timestamp-bounded or full-recording
accuracy. The fixture and generated reports expose this limitation explicitly.

## Results on the hard corrected clips

These numbers are development diagnostics, not certification:

| Experiment | Best word accuracy | Interpretation |
|---|---:|---|
| Existing automatic fused transcript | 48.5% | Fails the 90% target |
| INT8, auto language, beam 1 | 54.5% | Best deployable short-clip baseline |
| INT8, maximum log-probability selection | 55.6% | Confidence is not a sufficient selector |
| INT8 oracle across language/beam candidates | 59.6% | Reference-leaking, non-deployable ceiling |
| FP16, all selectors | 51.5% | Quantization is not the bottleneck |
| 30-second context oracle | 62.6% | Context adds only about 3 points to the oracle |
| Fixed four-language initial prompt | 21.2% | Rejected; strong harmful conditioning |
| FFT denoise + loudness normalization | 46.5% | Rejected; removed useful speech evidence |
| Speech expansion + loudness normalization | 53.5% | Rejected; one point below raw baseline |
| Left channel only | 53.5% | Rejected; downmix remains better |
| Right channel only | 48.5% | Rejected; downmix remains better |
| 500 ms segment padding | 36.4% | Rejected; too little decoder context |
| 1,000 ms segment padding | 43.4% | Rejected; below the 1,500 ms baseline |
| Prior non-overlapping WhisperKit context | 50.5% | Rejected; independent context did not help |

Strict word-midpoint scoring produced only 24.2% because the inherited segment times do
not match the corrected text extent. Whole-clip scoring can count neighbouring speech as
errors. Both policies are retained and labelled; neither can be used for certification.

## Derived noisy code-switch development evidence

A deterministic 19.2-minute fixture now interleaves 25 complete FLEURS utterances per
language at exact sample-derived boundaries. Every source clip and generated WAV is
SHA-256-bound. Speech is normalized to -20 dBFS before deterministic uniform noise is
added. The inherited FLEURS text remains human ground truth, but this is still synthetic
read speech: it is explicitly ineligible for accuracy certification, production scope, or
GIGA admission.

At 20 dB SNR, the corrected adaptive tail planner reached 90.43% aggregate accuracy in
117.34 seconds, but Romanian reached only 85.32% and Korean 89.24%. Forced Romanian and
Korean re-decodes returned identical text, so they were rejected as no-op controls.

At the predeclared 30 dB SNR point, the original target-proximity planner reached 90.59%
aggregate but dropped a complete English utterance after Korean overlap contaminated the
start of its chunk. A 1.5-second hard-silence policy was also rejected after collapsing to
72.63%: real language-turn gaps were often only 0.75–1.35 seconds. The corrected 0.7-second
turn policy isolates those transitions and reached the following result:

| Language | Reference words | Edits | WER | Word accuracy | CER | Gate |
|---|---:|---:|---:|---:|---:|---|
| English | 501 | 24 | 4.79% | 95.21% | 2.60% | PASS |
| Russian | 455 | 29 | 6.37% | 93.63% | 1.43% | PASS |
| Romanian | 579 | 71 | 12.26% | 87.74% | 5.16% | **FAIL** |
| Korean | 409 | 53 | 12.96% | 87.04% | 8.85% | **FAIL** |
| **Aggregate** | **1,944** | **177** | **9.10%** | **90.90%** | **3.99%** | **FAIL: per-language gate** |

Segment-level Faster-Whisper multilingual detection produced the same metrics on this
turn-isolated fixture. It remains enabled because it makes language decisions at decoder
segment level when a real chunk contains multiple languages; it is not credited as an
accuracy improvement here. Disabling previous-text conditioning prevents language leakage
across decoder windows. INT8/FP16 comparison also failed to provide a deployable selector:
FP16 improved Romanian to 88.60% but reduced Korean to 85.57%, and a reference-only
cross-precision oracle still reached only 89.12% Romanian and 87.78% Korean.

The structural quality contract passed every selected run, with no repetition, exhausted
fallbacks, timestamp disorder, or uncertainty. That does not override semantic WER. Exact
development verdicts are stored beside the SNR-20 and SNR-30 manifests; both say
`FAIL_CLOSED`, `production_scope_certified=false`, and `giga_admission_emitted=false`.

A text-only Romanian repair audit was also rejected. The pinned 1,790,902-byte FLEURS
Romanian training TSV was used only as a lexicon, never as candidate ground truth. Across a
finite distance/frequency/diacritic grid, the safest unique-correction policy fixed only
three errors and moved Romanian from 87.74% to 88.26%. Removing the diacritic-equivalence
guard worsened WER by replacing valid rare words with common neighbours. Qwen3 8B was no
better: the strict token-preserving prompt made no changes, while stronger correction
instructions produced plausible but wrong substitutions. No LLM-corrected transcript was
admitted. The hash-bound diagnostic verdict concludes that Romanian now requires stronger
acoustic evidence, not more aggressive text guessing.

## Next gates

1. The deterministic Kiwi spacing stage was installed on Gigabyte WSL and validated. It
   preserved every non-whitespace character, but automatic Korean word accuracy reached only
   89.49% and Romanian 89.98%; it is therefore retained as a guarded postprocessor and was not
   promoted as an accuracy solution. The untouched test slice remains unused.
2. Keep the untouched test slice sealed until a reference-independent validation policy passes
   every language floor. Never use the exposed validation slice as promotion evidence.
3. Build or acquire a timestamp-exact, human-ground-truth natural noisy
   conversational/code-switched fixture. The derived noisy fixture is useful development
   evidence but is not a production proxy.
4. Promote only a reference-independent selector. Oracle and declared-reference-language
   selection remain diagnostic upper bounds.
5. Keep cloud disabled and GIGA admission false.

## Next model-capacity gate

No alternative Whisper checkpoint is already installed on the Gigabyte. The smallest
credible capacity comparison is the official CTranslate2
`Systran/faster-whisper-large-v3` repository pinned at
`edaa852ec7e145841d8ffdb056a99866b5f0a478`. Its complete repository is 3,090,839,273
bytes (2.88 GiB), including a 3,087,284,237-byte FP16 `model.bin`. It would be loaded with
`int8_float16` compute.

The operator authorized the pinned 3,090,839,273-byte download on 2026-08-19. Every file hash
matched, and `int8_float16` inference used about 2.2 GiB VRAM with more than 5.7 GiB free. The
capacity test nevertheless failed promotion: automatic FLEURS validation reached 92.28%
aggregate but only 86.80% Korean, and even the oracle selector missed the all-language gate.
On the 83-minute lesson it reached only 49.25% word accuracy against the provisional MacWhisper
reference and looped `nya` 222 times, producing `REPROCESS_REQUIRED`. Full large-v3 remains an
evaluation-only optional adjudicator; large-v3-turbo remains the production default.
