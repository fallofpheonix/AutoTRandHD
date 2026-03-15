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
4. Cluster the kept components vertically: sort by y-centre; start a new
   cluster when the gap between consecutive y-centres exceeds
   ``cluster_gap_ratio × median_component_height``.
5. Select the largest cluster (by total component area) as the main-text
   block.  This discards marginal notes, catchwords, and ornaments that form
   smaller, isolated vertical clusters.
6. Return the bounding box of that cluster, padded slightly.

Typical usage::

    from src.preprocess.text_region import extract_text_region
    import cv2

    gray = cv2.imread("page.png", cv2.IMREAD_GRAYSCALE)
    region, bbox = extract_text_region(gray)
    # bbox = (x, y, w, h) in page-pixel coordinates
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # (x, y, w, h)

# Each element is a row from cv2.connectedComponentsWithStats.
_StatRow = np.ndarray  # shape (5,) — LEFT, TOP, WIDTH, HEIGHT, AREA


def _cluster_vertically(
    valid_stats: List[_StatRow],
    cluster_gap_ratio: float,
) -> List[List[_StatRow]]:
    """Group *valid_stats* into vertical clusters.

    Components are sorted by their y-centre.  A new cluster is started
    whenever the gap between consecutive y-centres exceeds
    ``cluster_gap_ratio × median_component_height``.

    Parameters
    ----------
    valid_stats:
        List of stat rows (each a length-5 array) for all accepted components.
    cluster_gap_ratio:
        Multiplier applied to the median component height to determine the
        minimum vertical gap that separates two distinct clusters.

    Returns
    -------
    List of clusters, each cluster being a list of stat rows.
    """
    heights = [int(s[cv2.CC_STAT_HEIGHT]) for s in valid_stats]
    median_h = float(np.median(heights)) if heights else 1.0
    gap_threshold = cluster_gap_ratio * median_h

    # Sort by vertical centre of each component.
    sorted_stats = sorted(
        valid_stats,
        key=lambda s: int(s[cv2.CC_STAT_TOP]) + int(s[cv2.CC_STAT_HEIGHT]) / 2.0,
    )

    clusters: List[List[_StatRow]] = [[sorted_stats[0]]]
    for prev, curr in zip(sorted_stats, sorted_stats[1:]):
        prev_cy = int(prev[cv2.CC_STAT_TOP]) + int(prev[cv2.CC_STAT_HEIGHT]) / 2.0
        curr_cy = int(curr[cv2.CC_STAT_TOP]) + int(curr[cv2.CC_STAT_HEIGHT]) / 2.0
        if curr_cy - prev_cy > gap_threshold:
            clusters.append([])
        clusters[-1].append(curr)

    return clusters


def extract_text_region(
    gray: np.ndarray,
    min_component_width: int = 20,
    min_component_height: int = 10,
    padding: int = 5,
    cluster_gap_ratio: float = 2.0,
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
    cluster_gap_ratio:
        Controls sensitivity of vertical cluster splitting.  A gap larger
        than ``cluster_gap_ratio × median_component_height`` starts a new
        cluster.  Increase this value to be more lenient (fewer clusters);
        decrease it to split marginal annotations into their own cluster more
        aggressively.  Default ``2.0`` works well for typical book pages.

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
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    # Filter: skip background (label 0); keep components above size thresholds.
    valid_stats: List[_StatRow] = []
    for lbl in range(1, num_labels):
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if cw >= min_component_width and ch >= min_component_height:
            valid_stats.append(stats[lbl])

    if not valid_stats:
        logger.warning("No text components found; returning full page.")
        return gray, (0, 0, w, h)

    # --- vertical clustering: select the largest cluster ---
    clusters = _cluster_vertically(valid_stats, cluster_gap_ratio)

    # "Largest" is the cluster with the greatest total component area.
    best_cluster = max(
        clusters,
        key=lambda cl: sum(int(s[cv2.CC_STAT_AREA]) for s in cl),
    )

    logger.debug(
        "Clusters found: %d; selected cluster size: %d components",
        len(clusters),
        len(best_cluster),
    )

    # --- bounding box of the selected cluster ---
    xs  = [int(s[cv2.CC_STAT_LEFT])                            for s in best_cluster]
    ys  = [int(s[cv2.CC_STAT_TOP])                             for s in best_cluster]
    x2s = [int(s[cv2.CC_STAT_LEFT]) + int(s[cv2.CC_STAT_WIDTH])  for s in best_cluster]
    y2s = [int(s[cv2.CC_STAT_TOP])  + int(s[cv2.CC_STAT_HEIGHT]) for s in best_cluster]

    x1 = max(0, min(xs)  - padding)
    y1 = max(0, min(ys)  - padding)
    x2 = min(w, max(x2s) + padding)
    y2 = min(h, max(y2s) + padding)

    bbox: BBox = (x1, y1, x2 - x1, y2 - y1)
    region = gray[y1:y2, x1:x2]

    logger.debug("Text region bbox: %s (page size: %dx%d)", bbox, w, h)
    return region, bbox
