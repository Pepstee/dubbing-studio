from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def nvidia_library_directories() -> tuple[Path, ...]:
    """Return CUDA runtime directories installed by NVIDIA's Python wheels."""
    directories: list[Path] = []
    for base in site.getsitepackages():
        nvidia = Path(base) / "nvidia"
        for component in ("cublas", "cudnn", "cuda_nvrtc"):
            candidate = nvidia / component / "lib"
            if candidate.is_dir():
                directories.append(candidate)
    return tuple(directories)


def main() -> None:
    """Re-exec Dubbing Studio with venv-local NVIDIA libraries discoverable."""
    arguments = [sys.executable, "-m", "dubbing", *sys.argv[1:]]
    if any(item in {"-h", "--help"} for item in sys.argv[1:]):
        os.execve(sys.executable, arguments, os.environ.copy())
        return
    directories = nvidia_library_directories()
    if not directories:
        raise SystemExit(
            "error: NVIDIA runtime libraries were not found. Install "
            "`dubbing-studio[transcription-faster]` in this environment."
        )
    env = os.environ.copy()
    current = env.get("LD_LIBRARY_PATH")
    values = [str(path) for path in directories]
    if current:
        values.append(current)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(values)
    os.execve(
        sys.executable,
        arguments,
        env,
    )


if __name__ == "__main__":
    main()
