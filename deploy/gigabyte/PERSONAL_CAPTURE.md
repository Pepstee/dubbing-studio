# Gigabyte Personal Capture deployment

## Durable locations

- Inbox: `/mnt/c/Users/gutua/Documents/DubbingCapture/inbox`
- Workspace: `/home/gutua/.local/share/dubbing-studio/personal-capture`
- Config: `/home/gutua/.config/dubbing-studio/personal-capture.json`
- Secret: `/home/gutua/.config/dubbing-studio/capture.token` (`0600`)
- GIGA transactional outbox: `~/.local/share/dubbing-studio/personal-capture/giga-outbox`

## Operations

```bash
systemctl --user restart dubbing-capture-watch dubbing-capture-review
systemctl --user status dubbing-capture-watch dubbing-capture-review
journalctl --user -u dubbing-capture-watch -u dubbing-capture-review
```

Both units are enabled under a user manager with lingering. The watcher has a filesystem lock,
a 60-second stability window, bounded 15-second polling, durable SQLite state and a health
heartbeat. Failed captures remain retryable.

## Private network

The service listens on WSL port 7433, but no public firewall rule is created. Tailscale Serve
must be enabled for the tailnet once by its owner, then configured on Windows:

```powershell
tailscale serve --bg --https=443 http://localhost:7433
tailscale serve status
```

Open the returned HTTPS URL with `?token=` followed by the contents of `capture.token`.
The token is moved into an authenticated session and all modifying forms also require CSRF.

## GIGA boundary

Approval writes `giga-event.json` inside the derived package. The outbox exporter recomputes
source-audio and transcript hashes and records the event id in its own SQLite ledger before
publishing exactly one JSON event. A separate GIGA consumer is intentionally required to map
this reviewed evidence into TwinStore; there is no automatic interpreted-memory write.
