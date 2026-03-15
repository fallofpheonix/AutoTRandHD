"""AutoTRandHD — historical OCR pipeline for seventeenth-century Spanish print.

Package layout::

    src/
        data/        PDF ingestion, manifest generation, split handling
        preprocess/  Image normalization, deskew, text-region extraction, line segmentation
        models/      CRNN architecture, CTC loss, trainer, inference
        decode/      Greedy and beam-search decoders, lexicon scoring, confidence
        postprocess/ Confidence gating, LLM correction, correction validation
        eval/        CER/WER metrics, error analysis, benchmark reporting
"""
