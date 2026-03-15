"""src.preprocess — deterministic image preprocessing for the OCR pipeline.

Public API
----------
- normalize.normalize_page    : grayscale conversion, denoising, contrast normalisation
- deskew.deskew_image         : skew detection and correction
- text_region.extract_text_region : isolate main-text block from page image
- line_segment.segment_lines  : segment a page into line-crop images with metadata
"""

from .normalize import normalize_page
from .deskew import deskew_image
from .text_region import extract_text_region
from .line_segment import segment_lines

__all__ = [
    "normalize_page",
    "deskew_image",
    "extract_text_region",
    "segment_lines",
]
