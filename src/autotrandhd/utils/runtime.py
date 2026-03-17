from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

import torch


def get_package_version() -> str:
    try:
        return version("autotrandhd")
    except PackageNotFoundError:
        return "0.1.0"


def build_runtime_snapshot() -> dict:
    return {
        "version": get_package_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "os": platform.system().lower(),
    }
