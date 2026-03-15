"""src.decode.beam_search — lexicon-constrained beam search decoder.

The beam search explores multiple hypotheses at each step and ranks them by a
combined acoustic score (from the CTC logits) and optional lexicon/language
prior.

This implementation follows the prefix-tree beam search for CTC described in
Hannun et al. (2014).  It is intentionally kept simple and readable.

Typical usage::

    import numpy as np
    from src.decode.beam_search import beam_decode
    from src.decode.lexicon import Lexicon

    logits = np.load("artifacts/logits/line_logits.npy")  # (T, C)
    lexicon = Lexicon.from_file("configs/lexicon.txt", vocab=vocab_str)
    results = beam_decode(logits, vocab=vocab_str, lexicon=lexicon, beam_width=10)
    for hyp in results:
        print(hyp["text"], hyp["score"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lexicon import Lexicon


@dataclass
class _Beam:
    """A single beam hypothesis."""
    token_indices: List[int] = field(default_factory=list)
    score: float = 0.0            # log-probability accumulator
    last_token: Optional[int] = None  # last emitted token (for blank/repeat logic)
    ended_blank: bool = True      # whether the previous step was blank


def beam_decode(
    logits: np.ndarray,
    vocab: str | List[str],
    blank_idx: int = 0,
    beam_width: int = 10,
    lexicon: Optional[Lexicon] = None,
    lm_alpha: float = 0.0,
    lm_beta: float = 0.0,
    max_candidates: int = 10,
) -> List[dict]:
    """Beam search CTC decoding.

    Parameters
    ----------
    logits:
        Log-probability array of shape ``(T, C)``.
    vocab:
        Ordered character vocabulary.
    blank_idx:
        CTC blank index.
    beam_width:
        Maximum number of active beams at each step.
    lexicon:
        Optional :class:`Lexicon` for rescoring hypotheses at word boundaries.
    lm_alpha:
        Language model weight.  Ignored when ``lexicon`` is ``None`` or when
        ``lm_alpha == 0``.
    lm_beta:
        Word insertion bonus.
    max_candidates:
        Maximum n-best hypotheses to return.

    Returns
    -------
    List[dict]
        Up to *max_candidates* hypotheses, sorted by descending score.
        Each dict has keys: ``text``, ``score``, ``token_indices``.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (T, C), got {logits.shape}")

    T, C = logits.shape
    if isinstance(vocab, str):
        vocab_list: List[str] = list(vocab)
    else:
        vocab_list = list(vocab)

    if len(vocab_list) != C:
        raise ValueError(
            f"vocab length ({len(vocab_list)}) must match logit classes ({C})"
        )

    # Initialise with a single empty beam.
    beams: List[_Beam] = [_Beam()]

    for t in range(T):
        log_probs = logits[t]  # (C,)
        new_beams: Dict[Tuple[int, ...], _Beam] = {}

        for beam in beams:
            # --- extend with blank ---
            blank_score = beam.score + log_probs[blank_idx]
            key = tuple(beam.token_indices)
            _merge_beam(
                new_beams, key,
                _Beam(
                    token_indices=list(beam.token_indices),
                    score=blank_score,
                    last_token=beam.last_token,
                    ended_blank=True,
                ),
            )

            # --- extend with non-blank tokens ---
            for c in range(C):
                if c == blank_idx:
                    continue
                token_score = beam.score + log_probs[c]

                # CTC rule: repeated token after non-blank is only emitted if
                # there was an intervening blank.
                if c == beam.last_token and not beam.ended_blank:
                    # Extend the existing last token (no new character added).
                    key = tuple(beam.token_indices)
                    _merge_beam(
                        new_beams, key,
                        _Beam(
                            token_indices=list(beam.token_indices),
                            score=token_score,
                            last_token=c,
                            ended_blank=False,
                        ),
                    )
                else:
                    # Emit a new character.
                    new_tokens = beam.token_indices + [c]
                    key = tuple(new_tokens)
                    extended = _Beam(
                        token_indices=new_tokens,
                        score=token_score,
                        last_token=c,
                        ended_blank=False,
                    )
                    # Optional lexicon rescoring at a newly created word boundary.
                    # Only apply when we just transitioned from a non-space to a space,
                    # to avoid scoring the same boundary multiple times on consecutive spaces.
                    if (
                        lexicon is not None
                        and lm_alpha > 0
                        and vocab_list[c] == " "
                        and (not beam.token_indices or vocab_list[beam.token_indices[-1]] != " ")
                    ):
                        extended.score += _lexicon_score(
                            new_tokens, vocab_list, lexicon, lm_alpha, lm_beta
                        )
                    _merge_beam(new_beams, key, extended)

        # Prune to beam_width.
        beams = sorted(new_beams.values(), key=lambda b: b.score, reverse=True)[
            :beam_width
        ]

    # Build output list.
    candidates = []
    for beam in sorted(beams, key=lambda b: b.score, reverse=True)[:max_candidates]:
        text = "".join(vocab_list[i] for i in beam.token_indices)
        candidates.append(
            {
                "text": text,
                "score": float(beam.score),
                "token_indices": beam.token_indices,
            }
        )

    return candidates


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_beam(
    beams_dict: Dict[Tuple[int, ...], "_Beam"],
    key: Tuple[int, ...],
    new_beam: "_Beam",
) -> None:
    """Insert *new_beam* into *beams_dict*, merging scores for identical keys."""
    if key in beams_dict:
        existing = beams_dict[key]
        # Log-sum-exp merge.
        merged_score = _log_sum_exp(existing.score, new_beam.score)
        existing.score = merged_score
    else:
        beams_dict[key] = new_beam


def _log_sum_exp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _lexicon_score(
    token_indices: List[int],
    vocab: List[str],
    lexicon: "Lexicon",
    alpha: float,
    beta: float,
) -> float:
    """Return a lexicon bonus for the most recently completed word."""
    text = "".join(vocab[i] for i in token_indices)
    # Detect word boundary by checking for trailing space.
    if text and text[-1] == " ":
        word = text.rstrip(" ").split(" ")[-1] if " " in text.rstrip(" ") else text.rstrip(" ")
        if lexicon.contains(word):
            return alpha * math.log(lexicon.score(word) + 1e-10) + beta
    return 0.0
