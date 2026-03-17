"""src.decode — greedy and beam-search decoders, lexicon scoring, and confidence.

Public API
----------
- greedy_decoder.greedy_decode   : fast greedy CTC decoding from logits
- beam_search.beam_decode        : lexicon-constrained beam search
- lexicon.Lexicon                 : token lookup and scoring helper
- confidence.token_confidence    : per-token confidence scores
- confidence.sequence_confidence : aggregate sequence confidence score
"""

from .greedy_decoder import greedy_decode
from .beam_search import beam_decode
from .lexicon import Lexicon
from .confidence import token_confidence, sequence_confidence

__all__ = [
    "greedy_decode",
    "beam_decode",
    "Lexicon",
    "token_confidence",
    "sequence_confidence",
]
