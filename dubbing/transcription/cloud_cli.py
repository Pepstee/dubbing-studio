from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dubbing.transcription.cloud import (
    CloudAdjudicator,
    CloudAuthorization,
    OpenAIHTTPTransport,
    TargetedCloudAdjudication,
)
from dubbing.transcription.models import (
    TranscriptionError,
    transcription_result_from_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload only unresolved 20-60 second context packets for fail-closed "
            "cloud transcription adjudication"
        )
    )
    parser.add_argument("media")
    parser.add_argument("--local-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recording-sha256", required=True)
    parser.add_argument("--operator-authorization-id", required=True)
    parser.add_argument("--provider", choices=("openai",), required=True)
    parser.add_argument(
        "--cloud-allowed",
        action="store_true",
        help="Required explicit per-recording upload authorization.",
    )
    args = parser.parse_args()

    try:
        result_document = json.loads(
            Path(args.local_result).read_text(encoding="utf-8")
        )
        local_result = transcription_result_from_dict(result_document)
        output = Path(args.output).resolve()
        authorization = CloudAuthorization(
            recording_sha256=args.recording_sha256,
            provider=args.provider,
            operator_authorization_id=args.operator_authorization_id,
            cloud_allowed=args.cloud_allowed,
            failed_spans_only=True,
        )
        adjudicator = CloudAdjudicator(
            output / "receipts",
            transport=OpenAIHTTPTransport(),
        )
        _, report = TargetedCloudAdjudication(adjudicator, output).run(
            args.media,
            local_result,
            authorization,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"could not load local transcript: {exc}") from exc
    except TranscriptionError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["admission_status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
