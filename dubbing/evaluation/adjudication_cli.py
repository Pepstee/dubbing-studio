from __future__ import annotations

import argparse
import json

from dubbing.evaluation.adjudication import adjudicate_transcripts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed automatic local transcript adjudication"
    )
    parser.add_argument("--primary", required=True)
    parser.add_argument("--source")
    parser.add_argument("--comparator", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--low-agreement", type=float, default=0.55)
    parser.add_argument("--consensus", type=float, default=0.75)
    args = parser.parse_args()
    report = adjudicate_transcripts(
        args.primary,
        args.comparator,
        output_path=args.output,
        source_path=args.source,
        window_ms=args.window_seconds * 1000,
        low_agreement_threshold=args.low_agreement,
        consensus_threshold=args.consensus,
    )
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))
    if report["verdict"]["status"] != "AUTOMATED_CONSENSUS_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
