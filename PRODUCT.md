# Personal Capture — provisional product definition

## Status

This is a provisional, evidence-derived definition for private dogfooding. It must be
confirmed after Artiom has selected a microphone and reviewed real recordings.

## User and job

One operator uses a private browser over Tailscale to send completed recordings to his
Gigabyte, review multilingual transcripts and translations, correct speakers, and explicitly
approve evidence for GIGA. It must never silently convert speech into interpreted memory.

## Product principles

- Originals are immutable and remain in the inbox.
- Russian, Romanian, English and Korean source speech is preserved beside English translation.
- Uncertainty is visible. Missing confidence is never represented as confidence.
- Review precedes approval; approval creates a verified event, not a psychological conclusion.
- Every processing stage is local, resumable and idempotent.
- The product remains useful with an empty inbox and no recording hardware.

## Platform

Private responsive web application hosted on the Gigabyte under WSL, reachable only through
the operator's Tailscale network.
