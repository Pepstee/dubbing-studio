from __future__ import annotations

import hmac
import json
import os
import re
import secrets
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from dubbing.capture.runtime import build_service, load_config

_ALLOWED = {
    ".aac", ".aiff", ".avi", ".flac", ".m4a", ".m4v", ".mkv", ".mov",
    ".mp3", ".mp4", ".mpeg", ".mpg", ".oga", ".ogg", ".opus", ".wav",
    ".weba", ".webm", ".wmv",
}
_CAPTURE_ID = re.compile(r"^[a-f0-9]{64}$")

_BASE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }} · Personal Capture</title><style>
:root{--bg:oklch(.975 .006 150);--surface:oklch(.995 .002 150);--ink:oklch(.22 .025 150);
--muted:oklch(.43 .025 150);--line:oklch(.87 .018 150);--moss:oklch(.400 .106 150);
--danger:oklch(.46 .15 25);--warn:oklch(.53 .105 80)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5
system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--surface);border-bottom:1px solid var(--line);z-index:2}
nav,main{max-width:1120px;margin:auto;padding:16px 22px}nav{display:flex;align-items:center;gap:14px}
nav a{color:var(--ink);font-weight:720;text-decoration:none}nav span{color:var(--muted)}
h1{font-size:1.7rem;letter-spacing:-.02em;margin:12px 0 4px}h2{font-size:1.1rem;margin:28px 0 10px}
p{max-width:72ch}.muted{color:var(--muted)}.flash{padding:10px 12px;background:var(--surface);
border:1px solid var(--line);margin:0 0 12px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button,.button,input,textarea{font:inherit;border-radius:8px}button,.button{border:0;background:var(--moss);
color:white;padding:9px 13px;font-weight:700;cursor:pointer;text-decoration:none}
button.secondary,.button.secondary{background:var(--surface);color:var(--ink);border:1px solid var(--line)}
button.danger{background:var(--danger)}button:focus-visible,a:focus-visible,input:focus-visible,
textarea:focus-visible{outline:3px solid oklch(.72 .12 150);outline-offset:2px}
table{width:100%;border-collapse:collapse;background:var(--surface)}th,td{padding:11px;text-align:left;
border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-size:.85rem}
.state{font-weight:750}.segments{border-top:1px solid var(--line)}
.segment{display:grid;grid-template-columns:110px minmax(0,1fr) minmax(0,1fr);gap:14px;
padding:16px 0;border-bottom:1px solid var(--line)}label{font-weight:650}textarea,input[type=text]{
width:100%;border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:9px}
textarea{min-height:88px;resize:vertical}.meta{font-size:.83rem;color:var(--muted)}
audio{width:100%;margin:14px 0}.upload{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
@media(max-width:760px){.segment{grid-template-columns:1fr}.segments{border-top:0}table thead{display:none}
table,tr,td{display:block}tr{border-bottom:1px solid var(--line)}td{border:0;padding:5px 11px}
nav,main{padding:13px 15px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style></head><body><header><nav><a href="{{ url_for('index') }}">Personal Capture</a>
<span>Private review queue</span></nav></header><main>
{% for message in get_flashed_messages() %}<div class="flash" role="status">{{ message }}</div>{% endfor %}
{{ body|safe }}</main></body></html>"""


def _safe_source(inbox: Path, name: str) -> Path:
    root = inbox.resolve()
    path = (root / name).resolve()
    if path.parent != root or path.is_symlink() or not path.is_file():
        abort(404)
    return path


def create_capture_app(config_path: str | Path) -> Flask:
    config = load_config(config_path)
    token_path = Path(config["network"]["token_file"])
    token = token_path.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise RuntimeError("capture token must contain at least 32 characters")
    service = build_service(config)
    inbox = Path(config["landing_inbox"]["wsl_path"])
    app = Flask(__name__)
    app.secret_key = bytes.fromhex(token[:64]) if re.fullmatch(r"[a-f0-9]{64,}", token) else token
    app.config["MAX_CONTENT_LENGTH"] = int(config["network"].get("max_upload_bytes", 2e9))

    @app.before_request
    def authenticate():
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        supplied = supplied or request.args.get("token", "")
        if not hmac.compare_digest(supplied, token) and not session.get("authenticated"):
            abort(401)
        session["authenticated"] = True
        session.setdefault("csrf", secrets.token_urlsafe(24))
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            csrf = request.form.get("csrf") or request.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(csrf, session["csrf"]):
                abort(403)

    @app.after_request
    def security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'unsafe-inline'; media-src 'self'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    def page(title: str, body: str, **context):
        inner = render_template_string(body, csrf=session["csrf"], **context)
        return render_template_string(_BASE, title=title, body=inner)

    @app.get("/")
    def index():
        records = tuple(reversed(tuple(service.records())))
        return page(
            "Queue",
            """<h1>Review queue</h1><p class="muted">Original recordings remain in the inbox.
            Nothing enters GIGA until you approve it.</p>
            <form class="upload" method="post" action="{{ url_for('upload') }}"
              enctype="multipart/form-data"><input type="hidden" name="csrf" value="{{ csrf }}">
              <input type="file" name="media" accept="audio/*,video/*" required>
              <button>Upload recording</button></form><h2>Captures</h2>
            {% if records %}<table><thead><tr><th>Recording</th><th>State</th><th>Updated</th>
            <th>Attempts</th></tr></thead><tbody>{% for record in records %}<tr>
            <td><a href="{{ url_for('detail', capture_id=record.capture_id) }}">
            {{ record.source_name }}</a></td><td class="state">{{ record.state }}</td>
            <td>{{ record.updated_at }}</td><td>{{ record.attempt_count }}</td></tr>{% endfor %}
            </tbody></table>{% else %}<p>No recordings yet. Uploading a completed recording
            places it into the same permanent inbox watched by the background worker.</p>{% endif %}""",
            records=records,
        )

    @app.post("/upload")
    def upload():
        item = request.files.get("media")
        if item is None or not item.filename:
            abort(400)
        name = secure_filename(item.filename)
        if not name or Path(name).suffix.lower() not in _ALLOWED:
            abort(415)
        destination = inbox / name
        if destination.exists():
            destination = inbox / f"{Path(name).stem}-{secrets.token_hex(4)}{Path(name).suffix}"
        partial = inbox / f".{destination.name}.{secrets.token_hex(8)}.partial"
        try:
            item.save(partial)
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
        flash(f"Uploaded {destination.name}. Processing begins after the stability window.")
        return redirect(url_for("index"))

    @app.get("/capture/<capture_id>")
    def detail(capture_id: str):
        if not _CAPTURE_ID.fullmatch(capture_id):
            abort(404)
        record = service.store.get(capture_id)
        if record is None:
            abort(404)
        transcript = translation = manifest = None
        if record.package_path:
            package = Path(record.package_path)
            transcript = json.loads((package / "transcript.json").read_text(encoding="utf-8"))
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            if (package / "translation.json").is_file():
                translation = json.loads((package / "translation.json").read_text(encoding="utf-8"))
        return page(
            record.source_name,
            """<a href="{{ url_for('index') }}">← Queue</a><h1>{{ record.source_name }}</h1>
            <p><strong>{{ record.state }}</strong> · {{ record.source_size }} bytes ·
            attempt {{ record.attempt_count }}</p>{% if record.error %}<p>{{ record.error }}</p>{% endif %}
            <audio controls preload="metadata" src="{{ url_for('audio', capture_id=record.capture_id) }}"></audio>
            {% if transcript %}<form method="post" action="{{ url_for('save', capture_id=record.capture_id) }}">
            <input type="hidden" name="csrf" value="{{ csrf }}"><div class="segments">
            {% for seg in transcript.segments %}<div class="segment"><div class="meta">
            {{ '%.1f'|format(seg.start_ms/1000) }}–{{ '%.1f'|format(seg.end_ms/1000) }}s<br>
            {{ seg.speaker or 'Unassigned' }}<br>confidence {{ seg.confidence if seg.confidence is not none else 'n/a' }}
            </div><label>Source<textarea name="source_{{ loop.index0 }}">{{ seg.text }}</textarea></label>
            <label>English translation<textarea name="target_{{ loop.index0 }}">{% if translation and loop.index0 < translation.segments|length %}{{ translation.segments[loop.index0].target_text or '' }}{% endif %}</textarea>
            {% if translation and loop.index0 < translation.segments|length %}<span class="meta">
            {{ translation.segments[loop.index0].source_language or 'uncertain' }}
            · {{ translation.segments[loop.index0].language_confidence or 'n/a' }}
            · {{ translation.segments[loop.index0].status }}</span>{% endif %}</label></div>{% endfor %}
            </div><h2>Review</h2><label>Speaker aliases (one <code>SPEAKER_00=Name</code> per line)
            <textarea name="aliases">{% if manifest %}{% for key,value in manifest.review.speaker_aliases.items() %}{{ key }}={{ value }}
{% endfor %}{% endif %}</textarea></label><label>Notes<textarea name="notes">{{ manifest.review.notes if manifest else '' }}</textarea></label>
            <div class="toolbar"><button name="action" value="save">Save edits</button>
            <button name="action" value="approve">Approve for GIGA outbox</button>
            <button class="danger" name="action" value="reject">Reject</button></div></form>
            {% elif record.state in ('failed','rejected') %}<form method="post"
            action="{{ url_for('retry', capture_id=record.capture_id) }}"><input type="hidden"
            name="csrf" value="{{ csrf }}"><button>Retry processing</button></form>{% endif %}""",
            record=record,
            transcript=transcript,
            translation=translation,
            manifest=manifest,
        )

    @app.get("/capture/<capture_id>/audio")
    def audio(capture_id: str):
        record = service.store.get(capture_id)
        if record is None:
            abort(404)
        return send_file(_safe_source(inbox, record.source_name), conditional=True)

    def aliases_from_form() -> dict[str, str]:
        aliases = {}
        for line in request.form.get("aliases", "").splitlines():
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key and value:
                aliases[key[:100]] = value[:200]
        return aliases

    @app.post("/capture/<capture_id>/save")
    def save(capture_id: str):
        record = service.store.get(capture_id)
        if record is None or not record.package_path:
            abort(404)
        transcript = json.loads(
            (Path(record.package_path) / "transcript.json").read_text(encoding="utf-8")
        )
        source = [{"text": request.form.get(f"source_{i}", "")}
                  for i in range(len(transcript["segments"]))]
        translation_path = Path(record.package_path) / "translation.json"
        targets = None
        if translation_path.is_file():
            translation = json.loads(translation_path.read_text(encoding="utf-8"))
            targets = [{"target_text": request.form.get(f"target_{i}", "")}
                       for i in range(len(translation["segments"]))]
        aliases = aliases_from_form()
        notes = request.form.get("notes", "")
        action = request.form.get("action", "save")
        if action == "reject":
            service.reject(capture_id, notes=notes)
        else:
            service.update_review(
                capture_id, transcript_segments=source, translation_segments=targets,
                speaker_aliases=aliases, notes=notes
            )
            if action == "approve":
                service.approve(capture_id, speaker_aliases=aliases, notes=notes)
        flash(f"Capture {action}d.")
        return redirect(url_for("detail", capture_id=capture_id))

    @app.post("/capture/<capture_id>/retry")
    def retry(capture_id: str):
        service.store.retry(capture_id)
        flash("Retry queued for the background worker.")
        return redirect(url_for("detail", capture_id=capture_id))

    @app.get("/health")
    def health():
        path = Path(config["workspace"]["wsl_path"]) / "health.json"
        return (path.read_text(encoding="utf-8"), 200, {"Content-Type": "application/json"})

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Private Personal Capture review service")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    app = create_capture_app(args.config)
    app.run(
        host=config["network"].get("bind_host", "127.0.0.1"),
        port=int(config["network"].get("port", 7433)),
    )


if __name__ == "__main__":
    main()
