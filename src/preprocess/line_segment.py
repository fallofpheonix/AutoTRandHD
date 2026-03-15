"""src.preprocess.line_segment — segment a text-region image into line crops.

Each detected text line is exported as a cropped grayscale image.  A metadata
record is produced for every line so that it can be traced back to its
source-page coordinates.

Algorithm
---------
1. Threshold the (already normalised) image.
2. Compute a horizontal projection profile: sum of foreground pixels per row.
3. Identify runs of rows with non-zero pixel density as line bands.
4. Merge bands that are too close together; drop bands that are too small.
5. Crop each band and save the line image.

Typical usage::

    from src.preprocess.line_segment import segment_lines

    line_records = segment_lines(
        image=text_region_gray,
        source_id="source_001",
        page_id=0,
        output_dir="artifacts/line_crops/source_001",
        page_bbox=(x, y, w, h),   # offset of text_region in the full page
    )
    # Each record: {line_id, bbox, image_path, transcript}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


def segment_lines(
    image: np.ndarray,
    source_id: str,
    page_id: int,
    output_dir: str | Path,
    page_bbox: BBox = (0, 0, 0, 0),
    min_line_height: int = 10,
    min_line_width: int = 20,
    merge_gap: int = 3,
    pad_vertical: int = 2,
    target_height: Optional[int] = None,
) -> List[dict]:
    """Segment *image* into text-line crops.

    Parameters
    ----------
    image:
        Grayscale text-region image (``H × W``, ``uint8``).
    source_id:
        Source identifier, used to build crop filenames.
    page_id:
        Zero-based page index within the source.
    output_dir:
        Directory where line-crop images are written.
    page_bbox:
        Bounding box ``(x, y, w, h)`` of the text region within the original
        full page image.  Used to record bounding boxes in page coordinates.
        Pass ``(0, 0, 0, 0)`` if the image is already the full page.
    min_line_height:
        Minimum height in pixels for a detected line band to be kept.
    min_line_width:
        Minimum non-zero column span for a detected line band to be kept.
    merge_gap:
        Row-gap threshold; adjacent bands closer than this are merged.
    pad_vertical:
        Extra rows added above and below each detected band before cropping.
    target_height:
        If provided, each line crop is resized to this height while preserving
        the aspect ratio.

    Returns
    -------
    List[dict]
        One dict per line with keys:
        ``line_id``, ``bbox``, ``image_path``, ``transcript`` (empty).

    Notes
    -----
    The ``transcript`` field is always empty at segmentation time.  It is
    populated later from a ground-truth transcription file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h_img, w_img = image.shape[:2]
    region_x, region_y, _, _ = page_bbox

    # --- binarise ---
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- horizontal projection ---
    h_proj = np.sum(thresh > 0, axis=1)  # (H,)

    bands = _projection_bands(h_proj, merge_gap=merge_gap)

    records: List[dict] = []
    line_id = 0

    for row_start, row_end in bands:
        row_start = max(0, row_start - pad_vertical)
        row_end = min(h_img, row_end + pad_vertical)

        band_h = row_end - row_start
        if band_h < min_line_height:
            continue

        crop = image[row_start:row_end, :]

        # Check minimum non-zero column span.
        col_has_content = np.any(thresh[row_start:row_end, :] > 0, axis=0)
        col_span = int(col_has_content.sum())
        if col_span < min_line_width:
            continue

        if target_height is not None and band_h != target_height:
            scale = target_height / band_h
            new_w = max(1, int(w_img * scale))
            crop = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

        filename = f"{source_id}_p{page_id:04d}_l{line_id:04d}.png"
        crop_path = output_dir / filename
        cv2.imwrite(str(crop_path), crop)

        # Bbox in full-page coordinates.
        bbox: BBox = (
            region_x,
            region_y + row_start,
            w_img,
            row_end - row_start,
        )

        records.append(
            {
                "line_id": line_id,
                "bbox": list(bbox),
                "image_path": str(crop_path),
                "transcript": "",
            }
        )
        line_id += 1

    logger.info(
        "Segmented %d lines for %s page %d → %s",
        len(records),
        source_id,
        page_id,
        output_dir,
    )
    return records


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _projection_bands(
    h_proj: np.ndarray,
    merge_gap: int = 3,
) -> List[Tuple[int, int]]:
    """Return (row_start, row_end) tuples of non-zero projection runs.

    Adjacent runs separated by at most *merge_gap* empty rows are merged.
    """
    in_band = False
    bands: List[Tuple[int, int]] = []
    start = 0

    for row, val in enumerate(h_proj):
        if val > 0 and not in_band:
            in_band = True
            start = row
        elif val == 0 and in_band:
            in_band = False
            bands.append((start, row))

    if in_band:
        bands.append((start, len(h_proj)))

    # Merge bands that are too close.
    if merge_gap <= 0 or len(bands) < 2:
        return bands

    merged: List[Tuple[int, int]] = [bands[0]]
    for b_start, b_end in bands[1:]:
        prev_start, prev_end = merged[-1]
        if b_start - prev_end <= merge_gap:
            merged[-1] = (prev_start, b_end)
        else:
            merged.append((b_start, b_end))

    return merged
