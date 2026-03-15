"""src.data.manifest — page manifest construction and persistence.

A *page manifest* is a pandas DataFrame (and its CSV serialisation) that
records every exported page image together with the metadata required by
downstream pipeline stages.

Minimum required columns (per docs/SYSTEM_ARCHITECTURE.md):

    source_id   — identifier of the originating PDF
    page_id     — zero-based page index within that PDF
    image_path  — absolute path to the exported page image
    dpi         — resolution used during export
    split       — train | val | test  (populated by splits.generate_splits)

Typical usage::

    from src.data.manifest import build_manifest, load_manifest, save_manifest

    manifest = build_manifest(
        source_id="source_001",
        image_paths=exported_paths,
        dpi=300,
    )
    save_manifest(manifest, "artifacts/manifests/source_001.csv")

    manifest = load_manifest("artifacts/manifests/source_001.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must be present in a valid manifest.
REQUIRED_COLUMNS: List[str] = [
    "source_id",
    "page_id",
    "image_path",
    "dpi",
    "split",
]


def build_manifest(
    source_id: str,
    image_paths: Sequence[Path | str],
    dpi: int,
    transcription_paths: Sequence[Path | str] | None = None,
) -> pd.DataFrame:
    """Build a page manifest for a single source.

    Parameters
    ----------
    source_id:
        Identifier for the originating PDF / source document.
    image_paths:
        Ordered sequence of exported page image paths.
    dpi:
        Resolution at which the images were exported.
    transcription_paths:
        Optional sequence of transcription file paths, one per page.
        When provided the list length must equal *image_paths*.

    Returns
    -------
    pd.DataFrame
        Manifest with columns ``source_id``, ``page_id``, ``image_path``,
        ``dpi``, and ``split`` (initially empty).  An optional
        ``transcript_path`` column is appended when *transcription_paths*
        is supplied.

    Raises
    ------
    ValueError
        If *transcription_paths* is provided but its length differs from
        *image_paths*.
    """
    image_paths = [str(p) for p in image_paths]

    if transcription_paths is not None:
        if len(transcription_paths) != len(image_paths):
            raise ValueError(
                f"transcription_paths length ({len(transcription_paths)}) "
                f"must match image_paths length ({len(image_paths)})"
            )

    records = []
    for page_id, image_path in enumerate(image_paths):
        rec: dict = {
            "source_id": source_id,
            "page_id": page_id,
            "image_path": image_path,
            "dpi": dpi,
            "split": "",  # populated later by generate_splits
        }
        if transcription_paths is not None:
            rec["transcript_path"] = str(transcription_paths[page_id])
        records.append(rec)

    df = pd.DataFrame(records)
    logger.info("Built manifest: %d pages for source %r", len(df), source_id)
    return df


def merge_manifests(manifests: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple per-source manifests into one.

    Parameters
    ----------
    manifests:
        Iterable of per-source DataFrames produced by :func:`build_manifest`.

    Returns
    -------
    pd.DataFrame
        Combined manifest with reset index.

    Raises
    ------
    ValueError
        If *manifests* is empty.
    """
    if not manifests:
        raise ValueError("manifests sequence must not be empty")
    combined = pd.concat(list(manifests), ignore_index=True)
    _validate_manifest(combined)
    return combined


def save_manifest(manifest: pd.DataFrame, path: str | Path) -> None:
    """Persist *manifest* to a CSV file.

    Parameters
    ----------
    manifest:
        DataFrame to save.
    path:
        Destination CSV path.  Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    logger.info("Saved manifest (%d rows) → %s", len(manifest), path)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a previously saved manifest from *path*.

    Parameters
    ----------
    path:
        Path to a CSV file produced by :func:`save_manifest`.

    Returns
    -------
    pd.DataFrame
        Loaded manifest.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the loaded CSV is missing required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path, dtype={"page_id": int, "dpi": int, "split": str})
    _validate_manifest(df)
    # Empty strings in the split column are read as NaN by pandas; restore them.
    df["split"] = df["split"].fillna("")
    logger.info("Loaded manifest (%d rows) from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Line-metadata helpers
# ---------------------------------------------------------------------------

LINE_REQUIRED_COLUMNS: List[str] = [
    "source_id",
    "page_id",
    "line_id",
    "bbox",
    "image_path",
    "transcript",
]


def build_line_metadata(
    source_id: str,
    page_id: int,
    line_records: Sequence[dict],
) -> pd.DataFrame:
    """Build a line-level metadata DataFrame for one page.

    Each record in *line_records* must contain at minimum:

    - ``line_id``    — integer line index within the page
    - ``bbox``       — bounding box as ``[x, y, w, h]`` in page-pixel coordinates
    - ``image_path`` — path to the exported line-crop image
    - ``transcript`` — ground-truth text (empty string if unavailable)

    Parameters
    ----------
    source_id:
        Identifier of the originating source.
    page_id:
        Zero-based page index.
    line_records:
        Sequence of dicts, one per segmented line.

    Returns
    -------
    pd.DataFrame
        Line metadata with ``source_id`` and ``page_id`` prepended.
    """
    records = [
        {"source_id": source_id, "page_id": page_id, **rec} for rec in line_records
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
