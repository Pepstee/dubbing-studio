from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dubbing.transcription.models import TranscriptionError, transcription_result_from_dict
from dubbing.transcription.teacher import CloudTeacherRunner, load_cloud_teacher_policy


def _macos_keychain_credential(service: str, account: str) -> str:
    service = service.strip()
    account = account.strip()
    if not service or not account:
        raise ValueError("Keychain service and account must both be non-empty")
    if sys.platform != "darwin":
        raise ValueError("macOS Keychain credential lookup requires macOS")
    child_environment = os.environ.copy()
    child_environment.pop("ELEVENLABS_API_KEY", None)
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-w",
                "-s",
                service,
                "-a",
                account,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=child_environment,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("could not read the ElevenLabs credential from Keychain") from None
    if result.returncode != 0:
        raise ValueError("could not read the ElevenLabs credential from Keychain")
    try:
        credential = result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise ValueError("the Keychain credential is not valid UTF-8") from None
    if (
        not credential
        or len(credential) > 16_384
        or any(character.isspace() for character in credential)
    ):
        raise ValueError("the Keychain credential is malformed")
    return credential


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
    parser.add_argument(
        "--keychain-service",
        help=(
            "Read the ElevenLabs API key from this macOS Keychain generic-password "
            "service; requires --keychain-account"
        ),
    )
    parser.add_argument(
        "--keychain-account",
        help=(
            "macOS Keychain account for --keychain-service; neither value contains "
            "the secret"
        ),
    )
    args = parser.parse_args()

    try:
        local_document = json.loads(
            Path(args.local_result).read_text(encoding="utf-8")
        )
        local = transcription_result_from_dict(local_document)
        policy = load_cloud_teacher_policy(args.policy)
        if bool(args.keychain_service) != bool(args.keychain_account):
            raise ValueError(
                "--keychain-service and --keychain-account must be supplied together"
            )
        credential = (
            _macos_keychain_credential(
                args.keychain_service,
                args.keychain_account,
            )
            if args.keychain_service
            else None
        )
        report = CloudTeacherRunner(
            policy,
            args.output,
            args.programme_state,
            credential=credential,
        ).run(args.media, local)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"could not start cloud teacher programme: {exc}") from exc
    except TranscriptionError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
