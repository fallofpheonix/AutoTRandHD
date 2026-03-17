"""src.data — PDF ingestion, manifest generation, and split handling.

Public API
----------
- pdf_loader.export_pages       : convert PDF pages → image files
- manifest.build_manifest       : build a page-level manifest DataFrame
- manifest.load_manifest        : load a persisted manifest from disk
- splits.generate_splits        : assign train/val/test labels to manifest rows
"""

from .pdf_loader import export_pages
from .manifest import build_manifest, load_manifest
from .splits import generate_splits

__all__ = [
    "export_pages",
    "build_manifest",
    "load_manifest",
    "generate_splits",
]
