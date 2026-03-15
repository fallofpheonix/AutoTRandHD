"""src.postprocess.llm_correction — construct prompts and query the LLM.

The LLM is invoked only as a bounded late-stage correction step for
low-confidence OCR lines.  It is **not** permitted to rewrite arbitrary
content; correction candidates are always validated by
:mod:`src.postprocess.validator` before acceptance.

This module is intentionally decoupled from any particular LLM SDK.
It exposes a ``LLMClient`` protocol that callers may satisfy with a mock or
a real HTTP client.

Every correction call produces an audit entry (see
``docs/SYSTEM_ARCHITECTURE.md — LLM Correction Audit``).

Typical usage::

    from src.postprocess.llm_correction import request_correction, MockLLMClient

    client = MockLLMClient()
    audit_entry = request_correction(
        line_id="source_001_p0000_l0003",
        ocr_text="eñ la çibdad",
        nbest=["eñ la çibdad", "en la ciudad"],
        client=client,
    )
    print(audit_entry["candidate_text"])
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client protocol (dependency inversion)
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Minimal protocol that any LLM backend must satisfy."""

    def complete(self, prompt: str) -> str:
        """Return a completion for *prompt*."""
        ...


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "You are correcting OCR output from a seventeenth-century Spanish printed source.\n"
    "Preserve all historical spellings, glyph variants, and orthographic conventions.\n"
    "Do NOT modernise spelling or punctuation.\n\n"
    "OCR output: {ocr_text}\n"
    "Alternative readings: {alternatives}\n\n"
    "Return only the corrected text, nothing else."
)


def build_prompt(ocr_text: str, nbest: List[str]) -> str:
    """Build a correction prompt.

    Parameters
    ----------
    ocr_text:
        The top-1 OCR hypothesis.
    nbest:
        Up to *k* alternative hypotheses from the decoder.

    Returns
    -------
    str
        Formatted prompt string.
    """
    alternatives = " | ".join(nbest[1:]) if len(nbest) > 1 else "(none)"
    return _PROMPT_TEMPLATE.format(ocr_text=ocr_text, alternatives=alternatives)


def request_correction(
    line_id: str,
    ocr_text: str,
    nbest: List[str],
    client: LLMClient,
    model: str = "gpt-4o-mini",
) -> dict:
    """Query the LLM and return an audit entry.

    The audit entry schema follows ``docs/SYSTEM_ARCHITECTURE.md``:

    - ``line_id``
    - ``input_text``
    - ``candidate_text``
    - ``decision``        — ``"pending"`` (validation happens in validator.py)
    - ``decision_reason`` — empty until :func:`validate_correction` is called

    Parameters
    ----------
    line_id:
        Unique identifier for the line.
    ocr_text:
        Original OCR hypothesis.
    nbest:
        N-best decoder candidates.
    client:
        LLM client implementing the :class:`LLMClient` protocol.
    model:
        Model identifier (logged in the audit entry for traceability).

    Returns
    -------
    dict
        Audit entry with ``decision="pending"``.
    """
    prompt = build_prompt(ocr_text, nbest)

    try:
        candidate_text = client.complete(prompt).strip()
    except Exception as exc:
        logger.warning("LLM call failed for line %r: %s", line_id, exc)
        candidate_text = ocr_text  # fall back to original

    audit_entry = {
        "line_id": line_id,
        "input_text": ocr_text,
        "candidate_text": candidate_text,
        "decision": "pending",
        "decision_reason": "",
        "model": model,
        "prompt": prompt,
    }
    logger.debug("LLM correction candidate for %r: %r", line_id, candidate_text)
    return audit_entry


# ---------------------------------------------------------------------------
# Mock client for testing
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Returns the input text unchanged.  Useful for testing and dry runs."""

    def complete(self, prompt: str) -> str:  # noqa: D401
        """Return the OCR text from *prompt* unchanged."""
        # Extract the OCR text from the prompt template.
        try:
            line = [l for l in prompt.splitlines() if l.startswith("OCR output: ")][0]
            return line.removeprefix("OCR output: ")
        except IndexError:
            return ""
