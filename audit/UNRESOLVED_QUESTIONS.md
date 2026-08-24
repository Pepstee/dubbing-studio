# Unresolved questions

1. What exact contract should make outbox evidence resolvable: copy transcript/translation,
   embed content, or publish an immutable package URI?
2. Which package representation is authoritative after review, and should text/SRT be
   regenerated or removed?
3. Should translation be recomputed after transcript edits, or explicitly marked stale?
4. Should failures retry only on operator request, with exponential backoff, or on a
   bounded schedule?
5. What lease/ownership prevents CLI and watcher from processing the same capture?
6. Which ASR/NLLB model revisions and hashes are approved, and where must preflight find
   them without network access?
7. Should checkpoint identity include all transcription options, detector identity, model
   revisions, code/schema version, and media decoder version?
8. Is the unauthenticated legacy web product still required? If yes, where may it bind and
   what authentication/rate limits apply?
9. Is query-string token bootstrap acceptable given history/access-log exposure?
10. Should `capture scan` allow translation to be disabled, and should CLI approval accept
    deployment config so it can publish consistently?
11. What is the authoritative runtime root on the Gigabyte, and does it match the systemd
    `WorkingDirectory=/home/gutua/software-factory/projects/dubbing-studio`?
12. What GIGA consumer validates and consumes `giga.personal-capture-event.v1`?
13. Are ignored synthetic transcript/audio artifacts safe to retain, and are any derived
    from a real voice? Metadata suggests a synthetic Mac E2E run; this was not independently
    re-proven.
14. Is NLLB's non-commercial checkpoint acceptable for the intended lifecycle?
15. What retention, encryption-at-rest, backup, and deletion policy applies to source
    recordings and biometric-like speaker evidence?
