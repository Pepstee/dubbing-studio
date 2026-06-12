from __future__ import annotations

import io
import subprocess
import uuid

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
_jobs: dict[str, bytes] = {}

MAX_UPLOAD_BYTES = 1_048_576  # 1 MiB

_UPLOAD_FORM = """\
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Dubbing Studio</title></head>
<body>
<h1>Dubbing Studio</h1>
<p>Upload an SRT subtitle file to synthesise dubbed audio.</p>
<form method="post" action="/dub" enctype="multipart/form-data">
  <label>SRT file: <input type="file" name="srt" accept=".srt" required></label><br><br>
  <label>Language (optional, e.g. en, es, fr): <input type="text" name="lang"></label><br><br>
  <button type="submit">Dub</button>
</form>
</body>
</html>"""


@app.route("/")
def index():
    return _UPLOAD_FORM


@app.route("/dub", methods=["POST"])
def dub():
    srt_file = request.files.get("srt")
    if srt_file is None:
        return jsonify({"error": "No SRT file provided"}), 400

    # Cap the read to MAX_UPLOAD_BYTES + 1; if we get more, reject before decode.
    chunk = srt_file.read(MAX_UPLOAD_BYTES + 1)
    if len(chunk) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Upload exceeds maximum allowed size of 1 MB"}), 413

    try:
        srt_text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        # Binary or wrongly-encoded upload: a client error, not a server crash.
        return jsonify({"error": "SRT file is not valid UTF-8 text"}), 400
    language = (request.form.get("lang") or "").strip()

    from dubbing.assembler import assemble_timeline
    from dubbing.backends.say import SayTTSBackend
    from dubbing.pipeline import DubbingPipeline

    backend = SayTTSBackend()
    pipeline = DubbingPipeline(backend)
    try:
        timed, results = pipeline.run_full(srt_text, language=language)
        combined = assemble_timeline(timed, results)
    except (RuntimeError, ValueError) as exc:
        # Synthesis genuinely failed — report it; never substitute silence.
        return jsonify({"error": str(exc)}), 502
    except subprocess.SubprocessError as exc:
        # Covers CalledProcessError AND TimeoutExpired (which is NOT a
        # CalledProcessError): a synthesis that hangs past its timeout must
        # surface as a 502, never an unhandled 500.
        return jsonify({"error": f"TTS engine failed: {exc}"}), 502

    job_id = str(uuid.uuid4())
    _jobs[job_id] = combined
    return jsonify({"id": job_id, "download": f"/download/{job_id}"})


@app.route("/download/<job_id>")
def download(job_id: str):
    if job_id not in _jobs:
        return "Not found", 404
    return send_file(
        io.BytesIO(_jobs[job_id]),
        mimetype="audio/wav",
        download_name="dubbed.wav",
        as_attachment=True,
    )


def main() -> None:
    app.run(host="0.0.0.0", port=7432)


if __name__ == "__main__":
    main()
