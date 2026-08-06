# Gigabyte local ASR accuracy goal

## Promotion gate

The target is at least 90% word accuracy (WER no greater than 10%) on timestamped,
human-corrected English, Russian, Romanian, and Korean evidence. Model agreement and
provisional transcripts are not ground truth. No result from this work may emit a GIGA
admission event until the gate passes.

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

## Next gates

1. Evaluate a deterministic Korean spacing stage on validation only. It must preserve every
   non-whitespace character, rebuild monotonic word timestamps, and record provenance. The
   official Kiwi runtime plus model is 91,590,807 bytes and requires separate download
   authorization before live evaluation.
2. If and only if that validation experiment passes, freeze it before selecting a different,
   untouched test slice. Never re-score the exposed holdout as promotion evidence.
3. Build or acquire a timestamp-exact, human-ground-truth noisy conversational/code-switched
   fixture after the clean per-language capacity gate passes. Clean FLEURS speech is not a
   production proxy.
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

Published Faster-Whisper measurements report roughly 2,926 MB VRAM for non-batched INT8
large-model inference. The Gigabyte had 3,320 MiB free at the preflight, leaving only about
394 MiB of indicative headroom; actual fit must be measured and an out-of-memory result must
fail safely. This download has not been authorized or started. The model is a capacity test,
not a promised route to 90%.
