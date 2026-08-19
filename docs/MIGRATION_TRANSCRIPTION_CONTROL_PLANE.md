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
   added. Unresolved Latin-script ambiguity remains uncertain rather than being guessed.
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
