from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from dubbing.apps.personal_capture.outbox import export_approved, verify_outbox_bundle
from dubbing.apps.personal_capture.config import load_config
from dubbing.apps.personal_capture.runtime import build_service

EXPECTED = {
    "en": "Today I am testing my private multilingual memory system.",
    "ru": "Сегодня я проверяю свою личную многоязычную систему памяти.",
    "ro": "Astăzi testez sistemul meu privat de memorie multilingvă.",
    "ko": "오늘 저는 개인 다국어 기억 시스템을 테스트하고 있습니다.",
}


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def run_benchmark(config_path: str | Path, audio_dir: str | Path, output_dir: str | Path) -> dict:
    config = load_config(config_path)
    service = build_service(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for language, expected in EXPECTED.items():
        source = Path(audio_dir) / f"{language}.wav"
        outcome = service.process(source)
        row = {
            "capture_id": outcome.capture_id,
            "language": language,
            "state": outcome.state,
            "source": source.name,
        }
        if outcome.package_path:
            transcript = json.loads(
                (outcome.package_path / "transcript.json").read_text(encoding="utf-8")
            )
            translation_path = outcome.package_path / "translation.json"
            translation = (
                json.loads(translation_path.read_text(encoding="utf-8"))
                if translation_path.is_file()
                else {"segments": []}
            )
            actual = transcript["text"]
            detections = [item.get("source_language") for item in translation["segments"]]
            row.update(
                {
                    "expected": expected,
                    "transcript": actual,
                    "normalized_similarity": round(
                        SequenceMatcher(None, _normalize(expected), _normalize(actual)).ratio(),
                        4,
                    ),
                    "detected_languages": detections,
                    "language_detection_pass": language in detections,
                    "translation_statuses": [
                        item.get("status") for item in translation["segments"]
                    ],
                    "translation_output_pass": all(
                        item.get("target_text") for item in translation["segments"]
                    ),
                    "speakers": sorted(
                        {
                            item.get("speaker")
                            for item in transcript["segments"]
                            if item.get("speaker")
                        }
                    ),
                }
            )
            service.approve(outcome.capture_id, notes="synthetic benchmark fixture")
            row["approval_pass"] = True
        rows.append(row)
    delivered = export_approved(
        config["workspace"]["wsl_path"],
        config["giga_outbox"]["path"],
        # Benchmark fixtures deliberately live outside the permanent inbox, but the
        # exporter still verifies their source bytes against this explicit root.
        inbox=audio_dir,
    )
    outbox = Path(config["giga_outbox"]["path"])
    handoff_pass = all(row.get("approval_pass") for row in rows) and all(
        verify_outbox_bundle(outbox / row["capture_id"]) for row in rows
    )
    document = {
        "schema_version": "dubbing.personal-capture-benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True,
        "generator": "espeak-ng",
        "results": rows,
        "outbox_events_delivered_this_run": delivered,
        "handoff_pass": handoff_pass,
    }
    (output / "benchmark.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Four-language Personal Capture benchmark",
        "",
        "Synthetic espeak-ng audio only; no private recordings were used.",
        "",
        "| Language | State | Similarity | Detection | Translation | Speakers |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['language']} | {row['state']} | "
            f"{row.get('normalized_similarity', 0):.4f} | "
            f"{'pass' if row.get('language_detection_pass') else 'fail'} | "
            f"{'pass' if row.get('translation_output_pass') else 'fail'} | "
            f"{', '.join(row.get('speakers', [])) or 'none'} |"
        )
    (output / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_benchmark(args.config, args.audio_dir, args.output_dir)


if __name__ == "__main__":
    main()
