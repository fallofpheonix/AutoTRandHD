from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def decode_image_bytes(payload: bytes) -> np.ndarray:
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported or corrupted image payload")
    return image


def read_grayscale_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def collect_image_paths(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_paths: list[Path] = []
    for child in sorted(folder.iterdir()):
        if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            image_paths.append(child)
    return image_paths
