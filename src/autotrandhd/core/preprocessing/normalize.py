"""src.preprocess.normalize — grayscale conversion, denoising, and contrast normalisation.

The normalisation pipeline is fully deterministic: given the same input image
and the same configuration values, it always produces the same output.

Pipeline steps
--------------
1. Convert to grayscale (if the image is colour).
2. Apply fast non-local means denoising (``cv2.fastNlMeansDenoising``).
3. Enhance local contrast with CLAHE.

The raw input file is never modified.

Typical usage::

    import cv2
    from autotrandhd.core.preprocessing.normalize import normalize_page

    img = cv2.imread("artifacts/pages/source_001_p0000.png")
    gray = normalize_page(img)
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_page(
    image: np.ndarray,
    denoise_h: float = 10.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Normalise a page image.

    Parameters
    ----------
    image:
        Input image as a NumPy array.  May be BGR, RGB, or already grayscale
        (``H × W`` or ``H × W × 1``).
    denoise_h:
        Filter strength for ``cv2.fastNlMeansDenoising``.  Larger values
        remove more noise but may blur fine details.  Set to ``0`` to skip.
    clahe_clip_limit:
        Contrast limit for CLAHE.  Set to ``0`` to skip contrast normalisation.
    clahe_tile_grid:
        Tile grid size for CLAHE.

    Returns
    -------
    np.ndarray
        Normalised grayscale image (``H × W``, dtype ``uint8``).
    """
    gray = _to_gray(image)

    if denoise_h > 0:
        gray = cv2.fastNlMeansDenoising(gray, h=denoise_h)

    if clahe_clip_limit > 0:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )
        gray = clahe.apply(gray)

    return gray


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert *image* to a 2-D uint8 grayscale array."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].astype(np.uint8)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(
        f"Unsupported image shape: {image.shape}. "
        "Expected 2-D or 3-D (HWC) NumPy array."
    )
