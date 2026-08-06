from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES: list[object] = []


def configure_nvidia_dlls() -> list[str]:
    """Admit pip-installed NVIDIA DLL directories before CTranslate2 import."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    candidates = (
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
    )
    admitted = []
    for path in candidates:
        if path.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
            admitted.append(str(path))
    if admitted:
        os.environ["PATH"] = os.pathsep.join(admitted + [os.environ.get("PATH", "")])
    return admitted
