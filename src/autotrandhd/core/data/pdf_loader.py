"""src.data.pdf_loader — convert PDF pages to image files.

Each page is exported at a fixed DPI and saved as a lossless PNG or TIFF.
The loader never mutates the source PDF.

Typical usage::

    from autotrandhd.core.data.pdf_loader import export_pages

    image_paths = export_pages(
        pdf_path="data/raw/source_001.pdf",
        output_dir="artifacts/pages/source_001",
        dpi=300,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def export_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
    image_format: str = "png",
    first_page: int | None = None,
    last_page: int | None = None,
) -> List[Path]:
    """Export each page of *pdf_path* as an image file.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    output_dir:
        Directory where exported images are written.  Created if absent.
    dpi:
        Rendering resolution.  300 DPI is the pipeline default.
    image_format:
        Output format — ``"png"`` (lossless) or ``"tiff"``.
    first_page:
        First 0-indexed page to export (inclusive).  Defaults to page 0.
    last_page:
        Last 0-indexed page to export (inclusive).  Defaults to last page.

    Returns
    -------
    List[Path]
        Sorted list of exported image paths in page order.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    ValueError
        If *image_format* is not ``"png"`` or ``"tiff"``.
    ImportError
        If PyMuPDF (fitz) is not installed.
    """
    try:
        import fitz  # type: ignore[import-untyped]  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyMuPDF is required for PDF export.  "
            "Install it with:  pip install PyMuPDF"
        ) from exc

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    allowed_formats = {"png", "tiff"}
    if image_format not in allowed_formats:
        raise ValueError(
            f"image_format must be one of {allowed_formats}, got {image_format!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    p_first = first_page if first_page is not None else 0
    p_last = last_page if last_page is not None else total_pages - 1
    p_first = max(0, p_first)
    p_last = min(total_pages - 1, p_last)

    # Derive a short source_id from the PDF stem for use in filenames.
    source_id = _pdf_source_id(pdf_path)

    exported: List[Path] = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)  # 72 pt = 1 inch in PDF coordinates

    for page_idx in range(p_first, p_last + 1):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
        filename = f"{source_id}_p{page_idx:04d}.{image_format}"
        out_path = output_dir / filename
        pix.save(str(out_path))
        exported.append(out_path)
        logger.debug("Exported page %d → %s", page_idx, out_path)

    doc.close()
    logger.info(
        "Exported %d page(s) from %s at %d DPI → %s",
        len(exported),
        pdf_path.name,
        dpi,
        output_dir,
    )
    return sorted(exported)


def _pdf_source_id(pdf_path: Path) -> str:
    """Return a short, filesystem-safe identifier derived from *pdf_path*.

    Uses the stem of the filename.  If two PDFs share the same stem the caller
    must place them in separate output directories.
    """
    return pdf_path.stem
