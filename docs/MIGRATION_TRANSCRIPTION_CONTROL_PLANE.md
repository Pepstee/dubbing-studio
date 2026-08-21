# Migration and rollback: transcription control plane

## Baseline preserved

The implementation branched from `codex/personal-capture-v0` at commit `9847278` while PR #1
(`Certify personal capture audio pipeline`) remained open and mergeable. It did not merge or
push that PR. Four pre-existing uncommitted anti-repetition/CLI fixes were inspected, retained
and incorporated rather than reset:

- MLX fallback schedule and hallucination controls;
- optional CLI temperature override;
- deterministic sorting of provider timestamps;
- the corresponding MLX regression tests.

The existing inbox watcher, review application, capture ledger, explicit approval operation,
immutable outbox bundles and GIGA event schema remain the deployment substrate.

## What changed

1. A hash-bound 83-minute regression fixture and exact WER/CER evaluator were added.
2. Provider-neutral decoder diagnostics and a fail-closed transcript quality contract were
   added to the backward-compatible transcription document.
3. The official WhisperKit local-server backend and adaptive resumable coordinator were
   added. The server process persists across chunks.
4. Turn-level script/language consistency checks and targeted forced-language retries were
   added. Unresolved Latin-script ambiguity remains uncertain rather than being guessed, and a
   turn dominated by an unsupported alphabetic script now fails closed even when the decoder
   incorrectly labels it English. Unsupported-script turns are quarantined with source-time and
   original-text hash evidence without launching the expensive local retry matrix.
5. pyannote Community-1, optional Precision-2, consented voiceprint identity and DER/JER/
   speaker-attributed-WER boundaries were added without downloading gated models.
6. Optional cloud ASR adapters were added behind explicit per-recording authorization and
   offline mocks.
7. Translation rows now hash-bind their authoritative source segment.
8. New capture packages contain `quality-report.json`; `FAILED` and `REPROCESS_REQUIRED`
   packages cannot be approved or exported.
9. Automatic local multi-run adjudication was added as the default non-cloud accuracy-evidence
   path. Manual calibration remains available for formal human-ground-truth certification but is
   not required to operate the local consensus gate.
10. A standalone source-bound faster-whisper GPU runner and pathology-targeted local fusion path
    were added for the Gigabyte RTX 4060. The runner does not require deploying the full repository
    to that laptop and never uploads audio to a cloud provider.
11. Targeted cloud adjudication now has an executable OpenAI transport and automatic unresolved-
    interval packetization. It is deliberately not part of the unattended watcher: each recording
    still needs its exact SHA-256, an operator authorization ID, an explicit `--cloud-allowed`
    invocation and a locally configured credential. Rollback is simply to omit this separate
    stage; the certified local capture/review/outbox path is unchanged.
12. Long-form diarization now reconciles chunk-local Sherpa labels through overlap-free TitaNet
    embeddings and conservative recording-global clustering. Old v1 chunk-local diarization
    checkpoints are intentionally incompatible with this stronger semantic contract; existing
    review packages remain readable, while a re-run must use a fresh/global checkpoint. Rolling
    back to the previous commit restores chunk-local labels but does not change transcript text,
    source media, approval state or outbox records.
13. The first natural-audio verification exposed severe Sherpa default-threshold fragmentation:
    29 labels in ten minutes. The fallback now exposes and deploys Sherpa's documented distance
    threshold (`0.85`), while global reconciliation uses overlap-safe complete-link clustering
    (`0.80`) and can repair non-overlapping same-chunk fragments. Diarization checkpoint schema
    v3 binds both the diarizer and independent embedding provider, including exact SHA-256 and
    byte size for Sherpa segmentation and embedding weights; old checkpoints fail closed.
    pyannote Community-1 is wired
    as the preferred local backend but remains fail-closed until an absolute local gated-model
    path exists; no terms or private-audio upload are performed by migration.
14. Adaptive extraction now preserves every source audio stream as discrete lossless channels.
    Only rejected spans receive bounded raw/downmix/channel candidates. A speech-normalized
    candidate exists for experiments but is disabled after reducing accuracy on both local
    models. Raw two-model consensus has precedence and now terminates processed-audio escalation;
    skipped candidates and decode counts are receipt-bound. When raw consensus is absent, a
    processed rescue still needs independent agreement on the same candidate and divergent
    processed consensuses fail closed. Candidate hashes and exact processing policy are
    checkpoint-bound. A deterministic 64-decode chunk-wide repair budget prevents long
    music/noise tails from multiplying every language/channel combination indefinitely. Budget
    exhaustion is receipt-bound and emits explicit uncertainty; partial processed evidence is
    never promoted. Rollback restores first-stream-only extraction and
    removes processed retry candidates without modifying source recordings or old packages.
15. The ambiguous aggregate `target_passed`/`promotion_passed` evaluator output was replaced by
    claim-scoped measurement, held-out benchmark and production-portfolio gates. The new gate
    requires every English/Russian/Romanian/Korean WER and CER threshold, minimum sample counts,
    semantic/timestamp validity and a finite-sample guard. Production additionally requires
    natural long-form, noisy code-switch, overlap, uncertainty/coverage and speaker-attributed evidence.
    Historical v1 reports remain immutable evidence; rerunning the evaluator emits v2 reports.

Older already-certified packages remain readable because the added transcript fields are
optional and quality gating is activated by the package's quality manifest entry.

## Rollback

No database migration or destructive rewrite was performed. To return to the pushed capture
baseline after committing or stashing any later local work:

```bash
git switch codex/personal-capture-v0
git rev-parse HEAD
```

The expected pushed baseline is `9847278`. Do not delete new checkpoint/output directories;
they are source-bound and can be retained for audit or resumed from the control-plane branch.

To remove only the optional CLI runtime later:

```bash
brew uninstall whisperkit-cli
```

The existing 464 MiB WhisperKit-small asset was reused in place and was not copied into the
repository. No large model was downloaded by this migration.

## Remaining explicit gates

- Human-corrected, timestamped, language-labelled transcript turns remain necessary only for a
  formal ground-truth accuracy certificate, not for automatic local consensus operation.
- Consented voice reference clips and labelled diarization reference for identity/DER/JER.
- Acceptance of pyannote Community-1 terms plus a locally configured Hugging Face token or
  an already-downloaded local model.
- Separate explicit authorization, provider selection and locally configured credential for
  any live cloud adjudication test.

Until these gates pass, the system is implemented and testable but not production-certified.
