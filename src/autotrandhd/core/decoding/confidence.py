"""src.decode.confidence — per-token and sequence-level confidence scores.

Confidence is estimated from the CTC log-probabilities:

* **Token confidence** — for each emitted token, the maximum log-probability
  at the corresponding time step, exponentiated to [0, 1].

* **Sequence confidence** — geometric mean of individual token confidences
  (equivalent to the average log-confidence), providing a single scalar
  summary for the whole line.

These scores are used downstream by :mod:`src.postprocess.confidence_gate`
to decide whether to invoke the LLM correction step.

Typical usage::

    import numpy as np
    from autotrandhd.core.decoding.confidence import token_confidence, sequence_confidence
    from autotrandhd.core.decoding.greedy_decoder import greedy_decode

    logits = np.load("line_logits.npy")   # (T, C)
    result = greedy_decode(logits, vocab)
    tok_conf = token_confidence(logits, result["raw_indices"], blank_idx=0)
    seq_conf = sequence_confidence(tok_conf)
"""

from __future__ import annotations

from typing import List

import numpy as np


def token_confidence(
    logits: np.ndarray,
    raw_indices: List[int],
    blank_idx: int = 0,
) -> List[float]:
    """Compute per-token confidence from CTC log-probabilities.

    For each non-blank, non-repeated token in the greedy decoding, the
    confidence is the maximum probability observed across the time steps
    that are attributed to that token.

    Parameters
    ----------
    logits:
        Log-probability array of shape ``(T, C)``.
    raw_indices:
        List of per-step argmax indices (output of
        :func:`greedy_decode` under key ``"raw_indices"``).
    blank_idx:
        CTC blank token index.

    Returns
    -------
    List[float]
        Confidence value in ``[0, 1]`` for each emitted (collapsed) token.
        Empty list if no tokens were emitted.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (T, C), got {logits.shape}")

    T = logits.shape[0]
    if len(raw_indices) != T:
        raise ValueError(
            f"len(raw_indices)={len(raw_indices)} must equal T={T}"
        )

    # Group time steps by emitted token, collapsing repeats and blanks.
    token_probs: List[List[float]] = []
    prev_idx: int | None = None
    current_probs: List[float] = []

    for t, idx in enumerate(raw_indices):
        prob = float(np.exp(logits[t, idx]))  # convert log-prob to prob
        if idx == blank_idx:
            if current_probs:
                token_probs.append(current_probs)
                current_probs = []
            prev_idx = None
        elif idx == prev_idx:
            # Repeated token — attribute to the same emission.
            current_probs.append(prob)
        else:
            if current_probs:
                token_probs.append(current_probs)
            current_probs = [prob]
            prev_idx = idx

    if current_probs:
        token_probs.append(current_probs)

    # Confidence for each token = max prob across its time steps.
    return [max(probs) for probs in token_probs]


def sequence_confidence(token_confs: List[float]) -> float:
    """Geometric mean of *token_confs*.

    Parameters
    ----------
    token_confs:
        List of per-token confidence values in ``[0, 1]``.

    Returns
    -------
    float
        Sequence-level confidence in ``[0, 1]``.
        Returns ``0.0`` for an empty list.
    """
    if not token_confs:
        return 0.0

    log_sum = sum(np.log(max(c, 1e-12)) for c in token_confs)
    return float(np.exp(log_sum / len(token_confs)))
