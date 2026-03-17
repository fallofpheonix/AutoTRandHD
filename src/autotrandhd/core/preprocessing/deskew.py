"""src.preprocess.deskew — detect and correct page skew.

Scanned pages are frequently tilted by a small angle.  This module estimates
the dominant skew from the foreground text mass and rotates the image to
straighten it.

Algorithm
---------
1. Threshold the grayscale image to isolate ink pixels.
2. Compute the minimum area bounding rectangle of all foreground pixels.
3. If the detected angle is within *max_angle_deg*, rotate the image.

The rotation is applied around the image center with a white (255) background
fill, expanding the canvas as needed to preserve the full page content. As a
result, the output image dimensions may differ from the input.

Typical usage::

    from autotrandhd.core.preprocessing.deskew import deskew_image
    import cv2

    gray = cv2.imread("page.png", cv2.IMREAD_GRAYSCALE)
    corrected, angle = deskew_image(gray)
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def deskew_image(
    image: np.ndarray,
    max_angle_deg: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """Detect and correct the skew of *image*.

    Parameters
    ----------
    image:
        Grayscale page image (``H × W``, dtype ``uint8``).
    max_angle_deg:
        Maximum skew angle (in degrees, absolute value) that the function
        will attempt to correct.  If the detected angle exceeds this limit
        the original image is returned unchanged.

    Returns
    -------
    corrected : np.ndarray
        Deskewed grayscale image.
    angle_deg : float
        Detected skew angle in degrees.  Positive values indicate a
        counter-clockwise tilt.  Returns ``0.0`` if correction was skipped.
    """
    angle = _estimate_skew(image)
    logger.debug("Detected skew angle: %.2f°", angle)

    if abs(angle) > max_angle_deg:
        logger.info(
            "Skew angle %.2f° exceeds max_angle_deg=%.1f°; skipping correction.",
            angle,
            max_angle_deg,
        )
        return image, 0.0

    if abs(angle) < 0.1:
        return image, angle

    corrected = _rotate(image, angle)
    return corrected, angle


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _estimate_skew(gray: np.ndarray) -> float:
    """Return the estimated skew angle of *gray* in degrees."""
    # Invert so that text pixels are white (foreground) on black.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]  # angle in (-90, 0]

    # OpenCV convention: -90 < angle <= 0.
    # We convert to a signed angle in (-45, 45].
    if angle < -45:
        angle += 90
    return angle


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *image* by *angle* degrees about its centre.

    The canvas is expanded to prevent clipping, and the background is filled
    with white (255).
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    rotation_matrix[0, 2] += (new_w / 2) - cx
    rotation_matrix[1, 2] += (new_h / 2) - cy

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated
