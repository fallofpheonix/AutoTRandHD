"""src.eval.benchmark — full evaluation pipeline with report generation.

A valid benchmark report must state (per ``docs/DATASET_AND_EVALUATION.md``):

- model checkpoint identifier
- decoder configuration
- lexicon version
- preprocessing configuration
- evaluation split identifier

:func:`run_benchmark` collects all required metadata, computes all metrics,
and writes a JSON report to ``output_dir``.  The report is deterministic given
the same inputs.

Typical usage::

    from autotrandhd.core.evaluation.benchmark import run_benchmark

    report = run_benchmark(
        predictions=preds,
        references=refs,
        checkpoint_id="checkpoint_epoch0050",
        decoder_config={"beam_width": 10, "blank_idx": 0},
        split_id="test",
        preprocess_config={"dpi": 300, "target_height": 64},
        output_dir="artifacts/reports",
    )
    print(report["cer"], report["wer"])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import compute_cer, compute_wer, compute_exact_match
from .analysis import rare_char_metrics, top_confusions

logger = logging.getLogger(__name__)


def run_benchmark(
    predictions: List[str],
    references: List[str],
    checkpoint_id: str,
    decoder_config: Dict[str, Any],
    split_id: str,
    preprocess_config: Optional[Dict[str, Any]] = None,
    lexicon_version: Optional[str] = None,
    rare_char_min_freq: int = 5,
    top_confusion_k: int = 20,
    output_dir: str | Path = "artifacts/reports",
    report_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics and save a benchmark report.

    Parameters
    ----------
    predictions:
        Ordered list of model output transcriptions.
    references:
        Ordered list of ground-truth transcriptions.
    checkpoint_id:
        Human-readable identifier for the model checkpoint used.
    decoder_config:
        Dict describing the decoder settings (beam width, blank idx, etc.).
    split_id:
        Name of the evaluation split (e.g. ``"test"``).
    preprocess_config:
        Dict describing preprocessing settings used to produce the inputs.
    lexicon_version:
        Identifier of the lexicon used during decoding (or ``None``).
    rare_char_min_freq:
        Characters appearing fewer times in references are treated as rare.
    top_confusion_k:
        Number of top character confusions to include in the report.
    output_dir:
        Directory where the JSON report is written.
    report_filename:
        Override the auto-generated report filename.

    Returns
    -------
    Dict[str, Any]
        Full benchmark report.

    Raises
    ------
    ValueError
        If *predictions* and *references* have different lengths.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions length ({len(predictions)}) must equal "
            f"references length ({len(references)})"
        )

    # --- compute metrics ---
    cer = compute_cer(predictions, references)
    wer = compute_wer(predictions, references)
    exact = compute_exact_match(predictions, references)
    rare = rare_char_metrics(predictions, references, min_freq=rare_char_min_freq)
    confusions = top_confusions(predictions, references, topk=top_confusion_k)

    report: Dict[str, Any] = {
        # Required provenance fields.
        "checkpoint_id": checkpoint_id,
        "decoder_config": decoder_config,
        "lexicon_version": lexicon_version,
        "preprocess_config": preprocess_config or {},
        "split_id": split_id,
        "num_lines": len(predictions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Primary metrics.
        "cer": cer,
        "wer": wer,
        "exact_match": exact,
        # Secondary metrics.
        "rare_char_metrics": rare,
        "top_confusions": [
            {"predicted": p, "reference": r, "count": c}
            for p, r, c in confusions
        ],
    }

    # --- save report ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if report_filename is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_filename = f"benchmark_{split_id}_{checkpoint_id}_{ts}.json"

    report_path = output_dir / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(
        "Benchmark report → %s  (CER=%.4f, WER=%.4f, EM=%.4f)",
        report_path,
        cer,
        wer,
        exact,
    )
    return report
