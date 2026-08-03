from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, session

from dubbing.transcription.adaptive import detect_silence_intervals
from dubbing.transcription.job import source_sha256
from dubbing.transcription.models import transcription_result_from_dict
from dubbing.transcription.quality import evaluate_transcript_quality

_STATUSES = {"pending", "approved", "corrected", "unclear"}
_LANGUAGES = {"en", "ru", "ro", "ko", "mixed", "unknown"}


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _progress(decisions: dict) -> dict:
    values = tuple(decisions["items"].values())
    counts = {status: sum(item["status"] == status for item in values) for status in _STATUSES}
    counts["reviewed"] = len(values) - counts["pending"]
    counts["total"] = len(values)
    counts["export_ready"] = counts["pending"] == 0
    return counts


def create_review_app(package_dir: str | Path) -> Flask:
    package = Path(package_dir).resolve()
    manifest_path = package / "manifest.json"
    decisions_path = package / "decisions.json"
    if not manifest_path.is_file() or not decisions_path.is_file():
        raise FileNotFoundError("review package requires manifest.json and decisions.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "dubbing.uncertain-review-package.v1":
        raise ValueError("unsupported review package schema")
    source = Path(manifest["source"]["path"])
    transcript_path = Path(manifest["transcript"]["path"])
    if source_sha256(source) != manifest["source"]["sha256"]:
        raise ValueError("review source hash mismatch")
    if source_sha256(transcript_path) != manifest["transcript"]["sha256"]:
        raise ValueError("review transcript hash mismatch")
    item_by_id = {item["id"]: item for item in manifest["items"]}
    lock = threading.Lock()
    result_path = package / "reviewed-result.json"
    quality_path = package / "reviewed-quality-report.json"

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = secrets.token_bytes(32)

    def load_decisions() -> dict:
        document = json.loads(decisions_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != "dubbing.uncertain-review-decisions.v1":
            raise ValueError("unsupported review decisions schema")
        if document.get("manifest_sha256") != source_sha256(manifest_path):
            raise ValueError("review decisions are not bound to this manifest")
        if set(document.get("items", {})) != set(item_by_id):
            raise ValueError("review decisions do not match manifest items")
        if any(
            not isinstance(decision, dict)
            or decision.get("status") not in _STATUSES
            or not isinstance(decision.get("text"), str)
            or decision.get("language") not in _LANGUAGES
            or not isinstance(decision.get("notes"), str)
            for decision in document["items"].values()
        ):
            raise ValueError("review decisions contain malformed entries")
        return document

    def export_state() -> dict | None:
        if not result_path.is_file() or not quality_path.is_file():
            return None
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        return {
            "quality_status": quality["status"],
            "approval_allowed": quality["approval_allowed"],
            "exported_at": datetime.fromtimestamp(
                result_path.stat().st_mtime, timezone.utc
            ).isoformat(),
            "result": str(result_path),
            "quality_report": str(quality_path),
            "giga_admission_emitted": False,
        }

    @app.before_request
    def csrf_session():
        session.setdefault("csrf", secrets.token_urlsafe(24))
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            supplied = request.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(supplied, session["csrf"]):
                abort(403)

    @app.after_request
    def security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'self'; script-src 'self'; media-src 'self'; "
            "connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index():
        decisions = load_decisions()
        return render_template(
            "review.html",
            title="Uncertain span review",
            csrf=session["csrf"],
            manifest=manifest,
            decisions=decisions,
            progress=_progress(decisions),
        )

    @app.get("/api/state")
    def state():
        decisions = load_decisions()
        return jsonify(
            {
                "source_name": manifest["source"]["name"],
                "items": manifest["items"],
                "decisions": decisions["items"],
                "progress": _progress(decisions),
                "uncertain_audio_duration_ms": manifest["uncertain_audio_duration_ms"],
                "export": export_state(),
            }
        )

    @app.get("/clips/<item_id>.wav")
    def clip(item_id: str):
        item = item_by_id.get(item_id)
        if item is None:
            abort(404)
        clip_path = (package / item["clip_path"]).resolve()
        if clip_path.parent != (package / "clips").resolve() or not clip_path.is_file():
            abort(404)
        return send_file(clip_path, mimetype="audio/wav", conditional=True)

    @app.post("/api/decision/<item_id>")
    def decision(item_id: str):
        item = item_by_id.get(item_id)
        if item is None:
            abort(404)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400)
        status = payload.get("status")
        text = str(payload.get("text", "")).strip()
        language = str(payload.get("language", "")).strip()
        notes = str(payload.get("notes", "")).strip()
        if status not in _STATUSES - {"pending"}:
            abort(400)
        if status == "approved":
            text = item["proposed_text"]
            language = item["proposed_language"]
        if not text or language not in _LANGUAGES or len(text) > 20_000 or len(notes) > 4000:
            abort(400)
        with lock:
            decisions = load_decisions()
            decisions["items"][item_id] = {
                "status": status,
                "text": text,
                "language": language,
                "notes": notes,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(decisions_path, decisions)
        return jsonify({"ok": True, "progress": _progress(decisions)})

    @app.post("/api/export")
    def export():
        with lock:
            decisions = load_decisions()
            progress = _progress(decisions)
            if not progress["export_ready"]:
                return jsonify({"ok": False, "error": "review_incomplete", "progress": progress}), 409
            transcript_document = json.loads(transcript_path.read_text(encoding="utf-8"))
            transcript = transcription_result_from_dict(transcript_document)
            segments = list(transcript.segments)
            lineage = []
            for item in manifest["items"]:
                selected = decisions["items"][item["id"]]
                before = segments[item["segment_index"]]
                corrected = selected["status"] == "corrected"
                unresolved = selected["status"] == "unclear"
                after = replace(
                    before,
                    text=selected["text"],
                    language=selected["language"],
                    uncertain=unresolved,
                    words=() if corrected else before.words,
                )
                segments[item["segment_index"]] = after
                lineage.append(
                    {
                        "item_id": item["id"],
                        "segment_index": item["segment_index"],
                        "decision": selected["status"],
                        "before_text_sha256": _text_sha256(before.text),
                        "after_text_sha256": _text_sha256(after.text),
                        "before_language": before.language,
                        "after_language": after.language,
                    }
                )
            provenance = dict(transcript.provenance or {})
            provenance["uncertain_span_review"] = {
                "manifest_sha256": source_sha256(manifest_path),
                "decisions_sha256": source_sha256(decisions_path),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "reviewed_items": len(lineage),
                "lineage": lineage,
                "giga_admission_authorized": False,
            }
            reviewed = replace(
                transcript,
                segments=tuple(segments),
                text=" ".join(segment.text for segment in segments),
                provenance=provenance,
            )
            quality = evaluate_transcript_quality(
                reviewed,
                expected_duration_ms=reviewed.duration_ms,
                known_silence_intervals=detect_silence_intervals(source),
            )
            _atomic_json(result_path, reviewed.to_dict())
            _atomic_json(quality_path, quality.to_dict())
            exported_at = datetime.fromtimestamp(
                result_path.stat().st_mtime, timezone.utc
            ).isoformat()
        return jsonify(
            {
                "ok": True,
                "quality_status": quality.status.value,
                "approval_allowed": quality.approval_allowed,
                "giga_admission_emitted": False,
                "exported_at": exported_at,
                "result": str(result_path),
                "quality_report": str(quality_path),
            }
        )

    @app.get("/health")
    def health():
        decisions = load_decisions()
        return jsonify({"status": "ok", "progress": _progress(decisions)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Private local transcript span reviewer")
    parser.add_argument("--package", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7444)
    args = parser.parse_args()
    try:
        loopback = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        parser.error("transcript review binds to loopback only")
    from waitress import serve

    serve(
        create_review_app(args.package),
        host=args.host,
        port=args.port,
        threads=4,
        channel_timeout=120,
        clear_untrusted_proxy_headers=True,
    )


if __name__ == "__main__":
    main()
