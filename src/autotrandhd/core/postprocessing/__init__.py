"""src.postprocess — confidence-gated LLM correction and validation.

Public API
----------
- confidence_gate.filter_low_confidence : identify lines that need LLM review
- llm_correction.request_correction     : build prompts and query the LLM
- validator.validate_correction         : accept or reject proposed corrections
"""

from .confidence_gate import filter_low_confidence
from .llm_correction import request_correction
from .validator import validate_correction

__all__ = [
    "filter_low_confidence",
    "request_correction",
    "validate_correction",
]
