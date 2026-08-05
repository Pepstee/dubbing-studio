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

The hardware and speed gates pass. The accuracy gate does not.

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

Strict word-midpoint scoring produced only 24.2% because the inherited segment times do
not match the corrected text extent. Whole-clip scoring can count neighbouring speech as
errors. Both policies are retained and labelled; neither can be used for certification.

## Next gates

1. Produce a small, timestamp-exact, stratified human reference set with full-clip text
   for all four languages. This is the minimum external evidence needed for a defensible
   90% claim.
2. Continue no-download work on deterministic speech enhancement and target-aware
   segmentation, evaluated as development signals until the exact fixture exists. The two
   initial speech-conditioning variants were rejected; do not promote them.
3. Promote only a reference-independent selector. Oracle and declared-reference-language
   selection remain diagnostic upper bounds.
4. Keep cloud disabled and GIGA admission false.
