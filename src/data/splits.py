"""src.data.splits — reproducible train / val / test split assignment.

Splits are assigned deterministically using a fixed random seed so that
re-running the pipeline always produces the same partition.

Two strategies are supported (see ``docs/DATASET_AND_EVALUATION.md``):

``"page"``
    Shuffle individual pages and divide by ratio.  Simple, but adjacent pages
    from the same source may land in different splits.

``"source"``
    Assign complete sources to splits.  Prevents leakage from near-identical
    neighbouring pages; preferred when multiple sources are available.

Typical usage::

    from src.data.splits import generate_splits

    manifest = generate_splits(
        manifest=manifest,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        strategy="source",
        seed=42,
    )
    print(manifest["split"].value_counts())
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SplitStrategy = Literal["page", "source"]


def generate_splits(
    manifest: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    strategy: SplitStrategy = "page",
    seed: int = 42,
) -> pd.DataFrame:
    """Assign ``split`` labels to every row of *manifest*.

    The function never mutates the input; it returns a copy with the
    ``split`` column populated.

    Parameters
    ----------
    manifest:
        Page manifest produced by :func:`src.data.manifest.build_manifest`.
    train_ratio:
        Fraction of data for training.
    val_ratio:
        Fraction for validation.
    test_ratio:
        Fraction for test.  Must satisfy
        ``train_ratio + val_ratio + test_ratio ≈ 1.0``.
    strategy:
        ``"page"`` or ``"source"`` (see module docstring).
    seed:
        Random seed ensuring reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *manifest* with the ``split`` column set to
        ``"train"``, ``"val"``, or ``"test"``.

    Raises
    ------
    ValueError
        If the ratios do not approximately sum to 1.0, or if *strategy*
        is unrecognised.
    """
    _check_ratios(train_ratio, val_ratio, test_ratio)

    manifest = manifest.copy()

    if strategy == "page":
        manifest = _split_by_page(manifest, train_ratio, val_ratio, seed)
    elif strategy == "source":
        manifest = _split_by_source(manifest, train_ratio, val_ratio, seed)
    else:
        raise ValueError(
            f"Unknown split strategy {strategy!r}. Choose 'page' or 'source'."
        )

    counts = manifest["split"].value_counts().to_dict()
    logger.info("Split counts: %s", counts)
    return manifest


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.6f}"
        )
    for name, r in [("train_ratio", train), ("val_ratio", val), ("test_ratio", test)]:
        if not 0.0 < r < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {r}")


def _assign_labels(
    indices: np.ndarray, train_ratio: float, val_ratio: float
) -> list[str]:
    """Return a list of split labels in the same order as *indices*."""
    n = len(indices)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    # The remainder goes to test.
    n_test = n - n_train - n_val
    if n_test < 1:
        # Edge case: not enough items for all three splits.
        n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
    return (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * n_test
    )


def _split_by_page(
    manifest: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int
) -> pd.DataFrame:
    """Assign splits at the individual page level."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(manifest))
    rng.shuffle(indices)
    labels = _assign_labels(indices, train_ratio, val_ratio)

    split_col = [""] * len(manifest)
    for shuffled_pos, original_idx in enumerate(indices):
        split_col[original_idx] = labels[shuffled_pos]

    manifest["split"] = split_col
    return manifest


def _split_by_source(
    manifest: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int
) -> pd.DataFrame:
    """Assign splits at the source level (all pages of a source share a split)."""
    sources = manifest["source_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled_sources = rng.permutation(sources)
    labels = _assign_labels(shuffled_sources, train_ratio, val_ratio)
    source_to_split = dict(zip(shuffled_sources, labels))
    manifest["split"] = manifest["source_id"].map(source_to_split)
    return manifest
