"""src.eval — OCR evaluation: CER/WER metrics, error analysis, and benchmarking.

Public API
----------
- metrics.compute_cer        : Character Error Rate
- metrics.compute_wer        : Word Error Rate
- metrics.compute_exact_match: Exact line match rate
- analysis.rare_char_metrics : Precision / Recall / F1 for rare characters
- analysis.top_confusions    : Most frequent character substitutions
- benchmark.run_benchmark    : Full evaluation pipeline with report generation
"""

from .metrics import compute_cer, compute_wer, compute_exact_match
from .analysis import rare_char_metrics, top_confusions
from .benchmark import run_benchmark

__all__ = [
    "compute_cer",
    "compute_wer",
    "compute_exact_match",
    "rare_char_metrics",
    "top_confusions",
    "run_benchmark",
]
