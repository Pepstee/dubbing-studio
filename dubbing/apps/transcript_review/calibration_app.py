from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, session

from dubbing.apps.transcript_review.package import _atomic_json
from dubbing.transcription.job import source_sha256

_STATUSES = {"pending", "complete", "no_speech"}
_LANGUAGES = {"en", "ru", "ro", "ko", "mixed", "unknown"}
_SPEAKER_PATTERN = re.compile(r"(?:Speaker [A-Z0-9]+|UNKNOWN|OVERLAP)\Z")


def _progress(decisions: dict) -> dict:
    values = tuple(decisions["items"].values())
    counts = {status: sum(row["status"] == status for row in values) for status in _STATUSES}
    counts["reviewed"] = len(values) - counts["pending"]
    counts["total"] = len(values)
    counts["export_ready"] = counts["pending"] == 0
    return counts


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_calibration_app(package_dir: str | Path) -> Flask:
    package = Path(package_dir).resolve()
    manifest_path = package / "manifest.json"
    decisions_path = package / "decisions.json"
    if not manifest_path.is_file() or not decisions_path.is_file():
        raise FileNotFoundError("calibration package requires manifest.json and decisions.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "dubbing.ground-truth-calibration-package.v1":
        raise ValueError("unsupported calibration package schema")
    source = Path(manifest["source"]["path"])
    candidate = Path(manifest["candidate"]["path"])
    if source_sha256(source) != manifest["source"]["sha256"]:
        raise ValueError("calibration source hash mismatch")
    if source_sha256(candidate) != manifest["candidate"]["sha256"]:
        raise ValueError("calibration candidate hash mismatch")
    item_by_id = {item["id"]: item for item in manifest["items"]}
    ground_truth_path = package / "ground-truth.json"
    lock = threading.Lock()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = secrets.token_bytes(32)

    def load_decisions() -> dict:
        document = json.loads(decisions_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != "dubbing.ground-truth-calibration-decisions.v1":
            raise ValueError("unsupported calibration decisions schema")
        if document.get("manifest_sha256") != source_sha256(manifest_path):
            raise ValueError("calibration decisions are not bound to this manifest")
        if set(document.get("items", {})) != set(item_by_id):
            raise ValueError("calibration decisions do not match manifest items")
        return document

    def export_state() -> dict | None:
        if not ground_truth_path.is_file():
            return None
        document = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        return {
            "exported_at": document["reviewed_at"],
            "turn_count": len(document["turns"]),
            "sha256": source_sha256(ground_truth_path),
            "path": str(ground_truth_path),
        }

    def validate_turns(item: dict, turns) -> list[dict]:
        if not isinstance(turns, list) or not 1 <= len(turns) <= 250:
            raise ValueError("a completed clip requires between 1 and 250 turns")
        clean = []
        for position, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError("turn must be an object")
            start_ms = int(turn.get("start_ms", -1))
            end_ms = int(turn.get("end_ms", -1))
            text = str(turn.get("text", "")).strip()
            language = str(turn.get("language", "")).strip()
            speaker = str(turn.get("speaker", "")).strip()
            if not 0 <= start_ms < end_ms <= item["duration_ms"]:
                raise ValueError("turn timestamps must stay inside the clip")
            if not text or len(text) > 20_000:
                raise ValueError("turn text cannot be blank")
            if language not in _LANGUAGES:
                raise ValueError("turn language is unsupported")
            if not _SPEAKER_PATTERN.fullmatch(speaker):
                raise ValueError("speaker must be anonymous or UNKNOWN")
            if speaker == "UNKNOWN" and text.casefold() != "[unclear]":
                raise ValueError("transcribable speech requires an anonymous speaker label")
            clean.append(
                {
                    "id": str(turn.get("id") or f"{item['id']}-human-{position:03d}"),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker": speaker,
                    "language": language,
                    "text": text,
                }
            )
        return clean

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
            "calibration.html",
            title="Ground-truth calibration",
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
                "coverage_gaps": manifest["coverage_gaps"],
                "total_audio_duration_ms": manifest["total_audio_duration_ms"],
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
        notes = str(payload.get("notes", "")).strip()
        if status not in _STATUSES - {"pending"} or len(notes) > 4000:
            abort(400)
        try:
            turns = [] if status == "no_speech" else validate_turns(item, payload.get("turns"))
        except (TypeError, ValueError):
            abort(400)
        with lock:
            decisions = load_decisions()
            decisions["items"][item_id] = {
                "status": status,
                "turns": turns,
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
            turns = []
            lineage = []
            for item in manifest["items"]:
                selected = decisions["items"][item["id"]]
                for turn in selected["turns"]:
                    turns.append(
                        {
                            **turn,
                            "clip_id": item["id"],
                            "start_ms": item["start_ms"] + turn["start_ms"],
                            "end_ms": item["start_ms"] + turn["end_ms"],
                        }
                    )
                lineage.append(
                    {
                        "clip_id": item["id"],
                        "decision": selected["status"],
                        "candidate_turns_sha256": _text_sha256(
                            json.dumps(item["candidate_turns"], ensure_ascii=False, sort_keys=True)
                        ),
                        "reviewed_turns_sha256": _text_sha256(
                            json.dumps(selected["turns"], ensure_ascii=False, sort_keys=True)
                        ),
                    }
                )
            turns.sort(key=lambda turn: (turn["start_ms"], turn["end_ms"], turn["id"]))
            document = {
                "schema_version": "dubbing.timestamped-ground-truth.v1",
                "source_sha256": manifest["source"]["sha256"],
                "review_manifest_sha256": source_sha256(manifest_path),
                "review_decisions_sha256": source_sha256(decisions_path),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "human_ground_truth": True,
                "speaker_labels_are_anonymous": True,
                "coverage_gaps": manifest["coverage_gaps"],
                "turns": turns,
                "lineage": lineage,
                "giga_admission_authorized": False,
            }
            _atomic_json(ground_truth_path, document)
        return jsonify(
            {
                "ok": True,
                "path": str(ground_truth_path),
                "sha256": source_sha256(ground_truth_path),
                "turn_count": len(turns),
                "giga_admission_emitted": False,
            }
        )

    @app.get("/health")
    def health():
        decisions = load_decisions()
        return jsonify({"status": "ok", "progress": _progress(decisions)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Private local ground-truth calibration reviewer")
    parser.add_argument("--package", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7445)
    args = parser.parse_args()
    try:
        loopback = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        parser.error("calibration review binds to loopback only")
    from waitress import serve

    serve(
        create_calibration_app(args.package),
        host=args.host,
        port=args.port,
        threads=4,
        channel_timeout=120,
        clear_untrusted_proxy_headers=True,
    )


if __name__ == "__main__":
    main()
