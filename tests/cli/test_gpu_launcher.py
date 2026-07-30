from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from dubbing import gpu_launcher


def test_library_directories_find_installed_components(tmp_path):
    cublas = tmp_path / "nvidia" / "cublas" / "lib"
    cudnn = tmp_path / "nvidia" / "cudnn" / "lib"
    cublas.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    with patch.object(gpu_launcher.site, "getsitepackages", return_value=[str(tmp_path)]):
        assert gpu_launcher.nvidia_library_directories() == (cublas, cudnn)


def test_launcher_reexecs_dubbing_with_library_path(tmp_path):
    library = tmp_path / "nvidia" / "cublas" / "lib"
    library.mkdir(parents=True)
    with (
        patch.object(gpu_launcher, "nvidia_library_directories", return_value=(library,)),
        patch.object(gpu_launcher.sys, "argv", ["dubbing-gpu", "transcribe", "audio.wav"]),
        patch.object(gpu_launcher.os, "execve") as execute,
        patch.dict(os.environ, {"LD_LIBRARY_PATH": "/existing"}),
    ):
        gpu_launcher.main()
    executable, arguments, environment = execute.call_args.args
    assert executable == gpu_launcher.sys.executable
    assert arguments == [
        gpu_launcher.sys.executable,
        "-m",
        "dubbing",
        "transcribe",
        "audio.wav",
    ]
    assert environment["LD_LIBRARY_PATH"] == f"{library}{os.pathsep}/existing"


def test_launcher_fails_without_runtime_libraries():
    with patch.object(gpu_launcher, "nvidia_library_directories", return_value=()):
        with pytest.raises(SystemExit, match="transcription-faster"):
            gpu_launcher.main()


def test_launcher_help_does_not_require_nvidia_runtime():
    with (
        patch.object(gpu_launcher.sys, "argv", ["dubbing-gpu", "--help"]),
        patch.object(gpu_launcher, "nvidia_library_directories") as discover,
        patch.object(gpu_launcher.os, "execve") as execute,
    ):
        gpu_launcher.main()

    discover.assert_not_called()
    assert execute.call_args.args[1] == [
        gpu_launcher.sys.executable,
        "-m",
        "dubbing",
        "--help",
    ]
