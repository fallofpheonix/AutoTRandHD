"""src.postprocess.confidence_gate — identify OCR lines that require LLM review.

Lines whose sequence confidence falls below a threshold are flagged for
optional LLM correction.  Only flagged lines incur an LLM API call.

Typical usage::

    from autotrandhd.core.decoding.confidence import sequence_confidence, token_confidence
    from autotrandhd.core.postprocessing.confidence_gate import filter_low_confidence

    decoder_outputs = [...]   # list of decoder result dicts
    flagged, accepted = filter_low_confidence(
        decoder_outputs,
        threshold=0.7,
        max_flags=200,
    )
"""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def filter_low_confidence(
    decoder_outputs: List[dict],
    threshold: float = 0.7,
    max_flags: int | None = None,
) -> Tuple[List[dict], List[dict]]:
    """Split *decoder_outputs* into flagged (low confidence) and accepted.

    Each dict in *decoder_outputs* must contain at minimum:

    - ``line_id``            — unique line identifier
    - ``top1_text``          — best hypothesis text
    - ``sequence_confidence``— scalar confidence in ``[0, 1]``

    Parameters
    ----------
    decoder_outputs:
        List of decoder result dicts.
    threshold:
        Lines with ``sequence_confidence < threshold`` are flagged.
    max_flags:
        Hard cap on the number of flagged lines per run.  Lines beyond the
        cap are accepted as-is to avoid excessive LLM cost.

    Returns
    -------
    flagged : List[dict]
        Lines below the confidence threshold.
    accepted : List[dict]
        Lines at or above the threshold (or beyond the flag cap).
    """
    flagged: List[dict] = []
    accepted: List[dict] = []

    for rec in decoder_outputs:
        conf = float(rec.get("sequence_confidence", 1.0))
        if conf < threshold:
            flagged.append(rec)
        else:
            accepted.append(rec)

    if max_flags is not None and len(flagged) > max_flags:
        overflow = flagged[max_flags:]
        flagged = flagged[:max_flags]
        accepted.extend(overflow)
        logger.info(
            "Flag cap (%d) reached; %d lines accepted without LLM review.",
            max_flags,
            len(overflow),
        )

    logger.info(
        "Confidence gate: %d flagged (threshold=%.2f), %d accepted.",
        len(flagged),
        threshold,
        len(accepted),
    )
    return flagged, accepted
