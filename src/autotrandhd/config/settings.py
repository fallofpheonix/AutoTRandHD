from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VOCAB = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:-'\"()ñçáéíóú"


@dataclass(frozen=True)
class AppSettings:
    model_path: Path = Path("artifacts/checkpoints/best.pt")
    device: str = "cpu"
    scratch_dir: Path = Path("artifacts/runtime")
    vocab: str = DEFAULT_VOCAB
    api_title: str = "AutoTRandHD API"
    benchmark_batch_size: int = 16


def load_settings() -> AppSettings:
    model_path = Path(os.getenv("AUTOTRANDHD_MODEL", "artifacts/checkpoints/best.pt"))
    device = os.getenv("AUTOTRANDHD_DEVICE", "cpu")
    scratch_dir = Path(os.getenv("AUTOTRANDHD_SCRATCH_DIR", "artifacts/runtime"))
    api_title = os.getenv("AUTOTRANDHD_API_TITLE", "AutoTRandHD API")
    benchmark_batch_size = int(os.getenv("AUTOTRANDHD_BENCHMARK_BATCH", "16"))
    return AppSettings(
        model_path=model_path,
        device=device,
        scratch_dir=scratch_dir,
        vocab=os.getenv("AUTOTRANDHD_VOCAB", DEFAULT_VOCAB),
        api_title=api_title,
        benchmark_batch_size=benchmark_batch_size,
    )
