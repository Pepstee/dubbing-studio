from __future__ import annotations

import hmac
import ipaddress
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
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from dubbing.apps.personal_capture.config import SUPPORTED_MEDIA_SUFFIXES, load_config
from dubbing.apps.personal_capture.runtime import build_control_service

_CAPTURE_ID = re.compile(r"^[a-f0-9]{64}$")

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
    service = build_control_service(config)
    inbox = Path(config["landing_inbox"]["wsl_path"])
    app = Flask(__name__)
    app.secret_key = bytes.fromhex(token[:64]) if re.fullmatch(r"[a-f0-9]{64,}", token) else token
    app.config["MAX_CONTENT_LENGTH"] = int(config["network"].get("max_upload_bytes", 2e9))
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=bool(config["network"].get("secure_cookie", True)),
    )

    @app.before_request
    def authenticate():
        if request.endpoint in {"login", "static"}:
            return None
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if supplied and hmac.compare_digest(supplied, token):
            session["authenticated"] = True
        if not session.get("authenticated"):
            abort(401)
        session.setdefault("csrf", secrets.token_urlsafe(24))
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            csrf = request.form.get("csrf") or request.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(csrf, session["csrf"]):
                abort(403)
        return None

    @app.after_request
    def security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'self'; media-src 'self'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            supplied = request.form.get("token", "")
            if not hmac.compare_digest(supplied, token):
                abort(401)
            session.clear()
            session["authenticated"] = True
            session["csrf"] = secrets.token_urlsafe(24)
            return redirect(url_for("index"))
        return render_template("login.html", title="Sign in")

    @app.get("/")
    def index():
        records = tuple(reversed(tuple(service.records())))
        return render_template(
            "index.html",
            title="Queue",
            csrf=session["csrf"],
            records=records,
        )

    @app.post("/upload")
    def upload():
        item = request.files.get("media")
        if item is None or not item.filename:
            abort(400)
        name = secure_filename(item.filename)
        if not name or Path(name).suffix.lower() not in SUPPORTED_MEDIA_SUFFIXES:
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
        return render_template(
            "detail.html",
            title=record.source_name,
            csrf=session["csrf"],
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
                try:
                    service.approve(
                        capture_id,
                        speaker_aliases=aliases,
                        notes=notes,
                        diarization_review_acknowledged=(
                            request.form.get("diarization_review_acknowledged")
                            == "yes"
                        ),
                    )
                except ValueError as exc:
                    flash(f"Edits saved, but approval is blocked: {exc}")
                    return redirect(url_for("detail", capture_id=capture_id))
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
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="acknowledge a non-loopback bind (normally use Tailscale Serve instead)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    host = args.host or config["network"].get("bind_host", "127.0.0.1")
    port = args.port or int(config["network"].get("port", 7433))
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not args.allow_remote:
        parser.error("non-loopback binds require --allow-remote")
    app = create_capture_app(args.config)
    try:
        from waitress import serve
    except ImportError as exc:
        raise RuntimeError(
            "Waitress is required for the Personal Capture review service"
        ) from exc
    serve(
        app,
        host=host,
        port=port,
        threads=4,
        channel_timeout=int(config["network"].get("upload_timeout_seconds", 3600)),
        clear_untrusted_proxy_headers=True,
    )


if __name__ == "__main__":
    main()
