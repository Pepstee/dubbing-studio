from __future__ import annotations

import argparse
import json
from pathlib import Path

from dubbing.transcription.models import TranscriptionError, transcription_result_from_dict
from dubbing.transcription.teacher import CloudTeacherRunner, load_cloud_teacher_policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authorized month-one cloud teacher beside a completed local "
            "transcript and build only high-agreement silver ASR examples"
        )
    )
    parser.add_argument("media")
    parser.add_argument("--local-result", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--programme-state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        local_document = json.loads(
            Path(args.local_result).read_text(encoding="utf-8")
        )
        local = transcription_result_from_dict(local_document)
        policy = load_cloud_teacher_policy(args.policy)
        report = CloudTeacherRunner(
            policy,
            args.output,
            args.programme_state,
        ).run(args.media, local)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"could not start cloud teacher programme: {exc}") from exc
    except TranscriptionError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
