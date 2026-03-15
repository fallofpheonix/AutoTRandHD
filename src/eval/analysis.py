"""src.eval.analysis — character-level error analysis.

Functions
---------
- rare_char_metrics  : Precision, Recall, F1 for rare characters
- top_confusions     : Most frequent character substitutions across the corpus

These functions operate on aligned prediction/reference pairs.  Alignment is
done at the character level using the standard edit distance algorithm.

Typical usage::

    from src.eval.analysis import rare_char_metrics, top_confusions

    preds = ["eñ la çibdad", "El rey don Felipe"]
    refs  = ["en la çibdad",  "El Rey Don Felipe"]

    metrics = rare_char_metrics(preds, refs, min_freq=1)
    confusions = top_confusions(preds, refs, topk=10)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def rare_char_metrics(
    predictions: List[str],
    references: List[str],
    min_freq: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Compute per-character Precision, Recall, and F1 for rare characters.

    A character is *rare* if it appears fewer than *min_freq* times across
    all reference strings.

    Parameters
    ----------
    predictions:
        List of predicted transcriptions.
    references:
        List of ground-truth transcriptions.
    min_freq:
        Characters appearing fewer times in references are treated as rare.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping ``{char: {"precision": float, "recall": float, "f1": float}}``.
        Only rare characters that appear in at least one reference or prediction
        are included.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    # Count character frequencies in references.
    ref_freq: Counter = Counter("".join(references))
    rare_chars = {c for c, n in ref_freq.items() if n < min_freq}

    if not rare_chars:
        logger.info("No rare characters found with min_freq=%d", min_freq)
        return {}

    # True positives, false positives, false negatives per rare character.
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for pred, ref in zip(predictions, references):
        pred_counts = Counter(pred)
        ref_counts = Counter(ref)
        for c in rare_chars:
            pc = pred_counts.get(c, 0)
            rc = ref_counts.get(c, 0)
            hit = min(pc, rc)
            tp[c] += hit
            fp[c] += max(0, pc - rc)
            fn[c] += max(0, rc - pc)

    results: Dict[str, Dict[str, float]] = {}
    for c in rare_chars:
        if tp[c] + fp[c] + fn[c] == 0:
            continue
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
        results[c] = {"precision": precision, "recall": recall, "f1": f1}

    return results


def top_confusions(
    predictions: List[str],
    references: List[str],
    topk: int = 20,
) -> List[Tuple[str, str, int]]:
    """Return the most frequent character-level substitutions.

    Uses a simple character-level Levenshtein alignment to identify
    substitution pairs.

    Parameters
    ----------
    predictions:
        List of predicted transcriptions.
    references:
        List of ground-truth transcriptions.
    topk:
        Number of top confusion pairs to return.

    Returns
    -------
    List[Tuple[str, str, int]]
        List of ``(predicted_char, reference_char, count)`` tuples, sorted by
        descending count.  ``(predicted_char, reference_char)`` represents the
        substitution of *reference_char* with *predicted_char*.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    confusion: Counter = Counter()

    for pred, ref in zip(predictions, references):
        subs = _char_substitutions(pred, ref)
        confusion.update(subs)

    return [
        (p, r, count)
        for (p, r), count in confusion.most_common(topk)
        if p != r
    ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _char_substitutions(pred: str, ref: str) -> List[Tuple[str, str]]:
    """Return list of ``(pred_char, ref_char)`` substitution pairs."""
    # Standard DP edit distance with backtracking.
    m, n = len(pred), len(ref)
    # dp[i][j] = edit distance between pred[:i] and ref[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find substitutions.
    subs: List[Tuple[str, str]] = []
    i, j = m, n
    while i > 0 and j > 0:
        if pred[i - 1] == ref[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            subs.append((pred[i - 1], ref[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1  # deletion
        else:
            j -= 1  # insertion

    return subs
