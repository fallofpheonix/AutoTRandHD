"""src.preprocess.text_region — isolate the main-text block from a page image.

Historical printed pages typically contain:

- a central main-text column
- marginal notes, catchwords, and page numbers
- decorative borders and ornaments

This module attempts to locate the bounding rectangle of the main-text
block so that downstream line segmentation operates on clean content only.

Algorithm (heuristic, deterministic)
-------------------------------------
1. Threshold to isolate ink.
2. Dilate horizontally to merge characters within words.
3. Find connected components; keep components whose width and height exceed
   minimum thresholds.
4. Cluster the kept components vertically; select the largest cluster as the
   main-text block.
5. Return the bounding box of that cluster, padded slightly.

Typical usage::

    from src.preprocess.text_region import extract_text_region
    import cv2

    gray = cv2.imread("page.png", cv2.IMREAD_GRAYSCALE)
    region, bbox = extract_text_region(gray)
    # bbox = (x, y, w, h) in page-pixel coordinates
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


def extract_text_region(
    gray: np.ndarray,
    min_component_width: int = 20,
    min_component_height: int = 10,
    padding: int = 5,
) -> Tuple[np.ndarray, BBox]:
    """Extract the main-text region from *gray*.

    Parameters
    ----------
    gray:
        Normalised grayscale page image (``H × W``, ``uint8``).
    min_component_width:
        Minimum pixel width of a connected component to be considered text.
    min_component_height:
        Minimum pixel height of a connected component to be considered text.
    padding:
        Extra pixels added to each side of the detected bounding box.

    Returns
    -------
    region : np.ndarray
        Cropped grayscale image containing the main-text block.
    bbox : (x, y, w, h)
        Bounding box of the extracted region in the original page coordinate
        system.  All values are in pixels.

    Notes
    -----
    When no significant text components are found, the entire image is
    returned with ``bbox = (0, 0, W, H)``.
    """
    h, w = gray.shape[:2]

    # --- binarise ---
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- horizontal dilation to merge characters into word blobs ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # --- find connected components ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    # Filter: skip background (label 0); keep components above size thresholds.
    valid_stats = []
    for lbl in range(1, num_labels):
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if cw >= min_component_width and ch >= min_component_height:
            valid_stats.append(stats[lbl])

    if not valid_stats:
        logger.warning("No text components found; returning full page.")
        return gray, (0, 0, w, h)

    # --- bounding box of all valid components ---
    xs = [s[cv2.CC_STAT_LEFT] for s in valid_stats]
    ys = [s[cv2.CC_STAT_TOP] for s in valid_stats]
    x2s = [s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] for s in valid_stats]
    y2s = [s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] for s in valid_stats]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(x2s) + padding)
    y2 = min(h, max(y2s) + padding)

    bbox: BBox = (x1, y1, x2 - x1, y2 - y1)
    region = gray[y1:y2, x1:x2]

    logger.debug("Text region bbox: %s (page size: %dx%d)", bbox, w, h)
    return region, bbox
