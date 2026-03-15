"""src.decode.greedy_decoder — fast greedy CTC decoding.

Greedy decoding selects the highest-probability token at each time step and
then collapses repeated tokens and removes blanks to produce a text sequence.

The decoder operates on stored logits (numpy arrays) so it can be replayed
without running the model again.

Typical usage::

    import numpy as np
    from src.decode.greedy_decoder import greedy_decode

    logits = np.load("artifacts/logits/source_001_p0000_l0000_logits.npy")
    # logits: (T, C) — log-probabilities
    result = greedy_decode(logits, vocab=" abcdefghijklmnopqrstuvwxyz...")
    print(result["text"])
"""

from __future__ import annotations

from typing import List

import numpy as np


def greedy_decode(
    logits: np.ndarray,
    vocab: str | List[str],
    blank_idx: int = 0,
) -> dict:
    """Decode a single logit sequence with the greedy CTC algorithm.

    Parameters
    ----------
    logits:
        Log-probability array of shape ``(T, C)``.
    vocab:
        Ordered character vocabulary.  ``vocab[i]`` is the character for
        class index ``i``.  Index ``blank_idx`` is the CTC blank and is
        omitted from the output.
    blank_idx:
        Index of the CTC blank token (default: 0).

    Returns
    -------
    dict
        Keys:

        - ``text`` — decoded string
        - ``token_indices`` — list of (collapsed) token indices
        - ``raw_indices`` — argmax indices before collapsing
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (T, C), got shape {logits.shape}")

    T, C = logits.shape
    if isinstance(vocab, str):
        vocab_list: List[str] = list(vocab)
    else:
        vocab_list = list(vocab)

    if len(vocab_list) != C:
        raise ValueError(
            f"vocab length ({len(vocab_list)}) must match logit classes ({C})"
        )

    # Argmax at each time step.
    raw_indices: List[int] = logits.argmax(axis=1).tolist()

    # Collapse repeats and remove blanks.
    token_indices: List[int] = []
    prev: int | None = None
    for idx in raw_indices:
        if idx != blank_idx and idx != prev:
            token_indices.append(idx)
        prev = idx

    text = "".join(vocab_list[i] for i in token_indices)

    return {
        "text": text,
        "token_indices": token_indices,
        "raw_indices": raw_indices,
    }


def greedy_decode_batch(
    logits_batch: np.ndarray,
    vocab: str | List[str],
    blank_idx: int = 0,
) -> List[dict]:
    """Decode a batch of logit sequences.

    Parameters
    ----------
    logits_batch:
        Array of shape ``(B, T, C)``.
    vocab:
        Character vocabulary.
    blank_idx:
        Blank token index.

    Returns
    -------
    List[dict]
        One result dict per sequence in the batch.
    """
    if logits_batch.ndim != 3:
        raise ValueError(
            f"logits_batch must be 3-D (B, T, C), got shape {logits_batch.shape}"
        )
    return [greedy_decode(logits_batch[i], vocab, blank_idx) for i in range(len(logits_batch))]
