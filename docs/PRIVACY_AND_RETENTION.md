# Personal Capture privacy and retention

## Data classification

Source recordings, transcripts, translations, speaker timestamps, review notes,
aliases and GIGA outbox bundles are private personal evidence. Treat them as
sensitive. Speaker embeddings are biometric-like inference material; the current
Sherpa backend keeps them in process and does not persist them.

## Conservative default

- Dubbing Studio never deletes or moves a source recording.
- No source, package, checkpoint or outbox directory is uploaded or backed up by
  this repository.
- Source and derived evidence remain operator-controlled until a separate,
  explicit retention decision is implemented.
- Do not place the workspace under OneDrive, Dropbox, iCloud or another automatic
  synchronization root.
- A rejected or failed capture is preserved. Retrying never requires deleting
  checkpoints.
- GIGA promotion remains a separate reviewed action; an outbox bundle is evidence,
  not interpreted autobiographical memory.

## Host release gate

Before admitting personal audio, the release receipt must record:

1. encryption-at-rest status for the Windows volume containing the WSL virtual disk;
2. workspace, token and outbox permissions;
3. whether any host backup/synchronization agent can read those paths;
4. the exact retention/deletion decision then in force.

If encryption or synchronization exposure is unknown, the host is not certified
for personal recordings. Preflight verifies Unix path permissions, but it cannot
prove Windows volume encryption or external backup policy.

## Deletion boundary

There is deliberately no automated deletion command. A safe deletion workflow
must account for the inbox source, resumable checkpoints, reviewed package,
outbox bundle, SQLite idempotency records and any already-consumed GIGA evidence.
Until that lifecycle is designed and explicitly authorized, deletion is a manual,
operator-owned action outside Dubbing Studio.
