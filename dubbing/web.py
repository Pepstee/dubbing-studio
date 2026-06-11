from __future__ import annotations

import io
import uuid
import wave

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
_jobs: dict[str, bytes] = {}

_UPLOAD_FORM = """\
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Dubbing Studio</title></head>
<body>
<h1>Dubbing Studio</h1>
<p>Upload an SRT subtitle file to synthesise dubbed audio.</p>
<form method="post" action="/dub" enctype="multipart/form-data">
  <label>SRT file: <input type="file" name="srt" accept=".srt" required></label><br><br>
  <button type="submit">Dub</button>
</form>
</body>
</html>"""


def _silence_wav(duration_ms: int) -> bytes:
    sample_rate = 22050
    num_samples = max(1, int(sample_rate * duration_ms / 1000))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * num_samples * 2)
    return buf.getvalue()


def _combine_wav(wav_list: list[bytes]) -> bytes:
    pcm_chunks: list[bytes] = []
    params = None
    for wav_data in wav_list:
        try:
            with wave.open(io.BytesIO(wav_data)) as wf:
                if params is None:
                    params = wf.getparams()
                pcm_chunks.append(wf.readframes(wf.getnframes()))
        except Exception:
            pass
    if not pcm_chunks or params is None:
        return _silence_wav(1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(b"".join(pcm_chunks))
    return buf.getvalue()


@app.route("/")
def index():
    return _UPLOAD_FORM


@app.route("/dub", methods=["POST"])
def dub():
    srt_file = request.files.get("srt")
    if srt_file is None:
        return jsonify({"error": "No SRT file provided"}), 400

    srt_text = srt_file.read().decode("utf-8")

    from dubbing.backends.say import SayTTSBackend
    from dubbing.pipeline import DubbingPipeline

    backend = SayTTSBackend()
    pipeline = DubbingPipeline(backend)
    _, results = pipeline.run_full(srt_text)

    wav_list = [r.audio_bytes for r in results if r.audio_bytes]
    combined = _combine_wav(wav_list) if wav_list else _silence_wav(1000)

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
