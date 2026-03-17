"""src.postprocess.validator — validate and accept/reject LLM correction candidates.

Validation rules
----------------
A correction is **rejected** if any of the following conditions hold:

1. **Empty** — the candidate is empty or whitespace-only.
2. **Length ratio** — the candidate length differs from the original by more
   than a configurable factor (guards against hallucinated expansions).
3. **Character set** — the candidate introduces characters not present in the
   allowed character set (guards against accidental script mixing).
4. **Historical spelling** — if a ``Lexicon`` is provided and the *original*
   hypothesis contains at least one word that is in the lexicon, but the
   *candidate* removes all such words, the correction is rejected.

Every call returns an updated audit entry with ``decision="accepted"`` or
``"rejected"`` and a human-readable ``decision_reason``.

Typical usage::

    from autotrandhd.core.postprocessing.validator import validate_correction

    audit_entry = validate_correction(
        audit_entry=audit_entry,
        allowed_chars=set("abcdefghijklmnopqrstuvwxyz "),
    )
    print(audit_entry["decision"])  # "accepted" or "rejected"
"""

from __future__ import annotations

import logging
from typing import Optional

from autotrandhd.core.decoding.lexicon import Lexicon

logger = logging.getLogger(__name__)

# Decisions.
ACCEPTED = "accepted"
REJECTED = "rejected"


def validate_correction(
    audit_entry: dict,
    allowed_chars: Optional[set[str]] = None,
    max_length_ratio: float = 2.0,
    lexicon: Optional[Lexicon] = None,
) -> dict:
    """Validate an LLM correction candidate and update *audit_entry*.

    Parameters
    ----------
    audit_entry:
        Dict produced by :func:`src.postprocess.llm_correction.request_correction`.
        Must contain ``input_text`` and ``candidate_text``.
    allowed_chars:
        Set of permitted characters.  ``None`` skips this check.
    max_length_ratio:
        Reject if ``len(candidate) / len(original) > max_length_ratio`` or
        the inverse.
    lexicon:
        Historical word lexicon used to protect valid spellings.

    Returns
    -------
    dict
        *audit_entry* with ``decision`` and ``decision_reason`` populated.
    """
    entry = dict(audit_entry)  # do not mutate caller's dict
    original = entry.get("input_text", "")
    candidate = entry.get("candidate_text", "")

    reason = _check_empty(candidate)
    if reason:
        return _reject(entry, reason)

    reason = _check_length_ratio(original, candidate, max_length_ratio)
    if reason:
        return _reject(entry, reason)

    if allowed_chars is not None:
        reason = _check_char_set(candidate, allowed_chars)
        if reason:
            return _reject(entry, reason)

    if lexicon is not None:
        reason = _check_lexicon_coverage(original, candidate, lexicon)
        if reason:
            return _reject(entry, reason)

    entry["decision"] = ACCEPTED
    entry["decision_reason"] = "all checks passed"
    logger.debug("Correction ACCEPTED for line %r", entry.get("line_id"))
    return entry


# ---------------------------------------------------------------------------
# Private validation helpers
# ---------------------------------------------------------------------------

def _check_empty(candidate: str) -> str:
    if not candidate or not candidate.strip():
        return "candidate is empty"
    return ""


def _check_length_ratio(original: str, candidate: str, max_ratio: float) -> str:
    orig_len = max(len(original), 1)
    cand_len = max(len(candidate), 1)
    ratio = cand_len / orig_len
    if ratio > max_ratio or ratio < (1.0 / max_ratio):
        return (
            f"length ratio {ratio:.2f} outside allowed range "
            f"[{1.0 / max_ratio:.2f}, {max_ratio:.2f}]"
        )
    return ""


def _check_char_set(candidate: str, allowed_chars: set[str]) -> str:
    illegal = {c for c in candidate if c not in allowed_chars}
    if illegal:
        return f"candidate contains disallowed characters: {sorted(illegal)}"
    return ""


def _check_lexicon_coverage(original: str, candidate: str, lexicon: Lexicon) -> str:
    orig_words = set(original.split())
    cand_words = set(candidate.split())

    orig_lex_hits = {w for w in orig_words if lexicon.contains(w)}
    if not orig_lex_hits:
        return ""  # original had no lexicon words; nothing to protect

    cand_lex_hits = {w for w in cand_words if lexicon.contains(w)}
    if not cand_lex_hits:
        return (
            "correction removes all historically-valid lexicon words "
            f"({orig_lex_hits}) from the original"
        )
    return ""


def _reject(entry: dict, reason: str) -> dict:
    entry["decision"] = REJECTED
    entry["decision_reason"] = reason
    logger.debug(
        "Correction REJECTED for line %r: %s", entry.get("line_id"), reason
    )
    return entry
