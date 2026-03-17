"""src.eval.metrics — CER, WER, and exact-line-match computation.

All metrics are computed from plain Python strings.  The implementation uses
``jiwer`` for WER/CER to ensure compatibility with the community standard and
to avoid reinventing normalisation logic.

Typical usage::

    from autotrandhd.core.evaluation.metrics import compute_cer, compute_wer, compute_exact_match

    predictions = ["eñ la çibdad", "El rey don Felipe"]
    references  = ["en la çibdad",  "El Rey Don Felipe"]

    print(compute_cer(predictions, references))        # e.g. 0.04
    print(compute_wer(predictions, references))        # e.g. 0.25
    print(compute_exact_match(predictions, references))# e.g. 0.0
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


def compute_cer(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute Character Error Rate (CER).

    CER = (substitutions + deletions + insertions) / total_reference_characters

    Parameters
    ----------
    predictions:
        List of predicted transcriptions.
    references:
        List of ground-truth transcriptions.

    Returns
    -------
    float
        CER in ``[0, ∞)``.  Values above 1.0 indicate more errors than
        reference characters (possible but unusual).

    Raises
    ------
    ValueError
        If *predictions* and *references* have different lengths.
    ImportError
        If ``jiwer`` is not installed.
    """
    _check_lengths(predictions, references)
    try:
        import jiwer  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("jiwer is required: pip install jiwer") from exc

    # jiwer.cer expects lists of strings.
    cer = jiwer.cer(references, predictions)
    return float(cer)


def compute_wer(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute Word Error Rate (WER).

    WER = (substitutions + deletions + insertions) / total_reference_words

    Parameters
    ----------
    predictions:
        List of predicted transcriptions.
    references:
        List of ground-truth transcriptions.

    Returns
    -------
    float
        WER in ``[0, ∞)``.

    Raises
    ------
    ValueError
        If *predictions* and *references* have different lengths.
    ImportError
        If ``jiwer`` is not installed.
    """
    _check_lengths(predictions, references)
    try:
        import jiwer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("jiwer is required: pip install jiwer") from exc

    wer = jiwer.wer(references, predictions)
    return float(wer)


def compute_exact_match(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute the exact line match rate.

    A line is considered an exact match only if the prediction equals the
    reference after Unicode normalisation (NFC).

    Parameters
    ----------
    predictions:
        List of predicted transcriptions.
    references:
        List of ground-truth transcriptions.

    Returns
    -------
    float
        Fraction of lines that are exact matches, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input lists have different lengths.
    """
    import unicodedata

    _check_lengths(predictions, references)
    if not predictions:
        return 0.0

    matches = sum(
        unicodedata.normalize("NFC", p) == unicodedata.normalize("NFC", r)
        for p, r in zip(predictions, references)
    )
    return matches / len(predictions)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check_lengths(predictions: List[str], references: List[str]) -> None:
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions length ({len(predictions)}) must equal "
            f"references length ({len(references)})"
        )
