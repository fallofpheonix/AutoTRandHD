# Project Scope

## Objective

Build a reproducible OCR and transcription pipeline for seventeenth-century Spanish printed sources using:

- weighted CNN-RNN recognition
- lexicon-constrained decoding
- selective LLM-based post-correction

## Problem

Generic OCR systems underperform on early modern Spanish print because they are not designed for:

- historical glyph variants
- degraded scans and bleed-through
- inconsistent spacing and orthography
- decorative page elements that disrupt segmentation

The project must produce a practical baseline that is technically credible and extensible.

## Primary Goals

- detect and isolate main text from historical scanned PDFs
- segment pages into trainable and inferable line images
- train a CRNN recognizer with CTC alignment
- improve rare glyph recognition through weighted learning
- reduce hallucinated outputs through lexicon-constrained beam search
- apply an LLM only as a bounded late-stage correction step
- evaluate quality with explicit OCR metrics

## Non-Goals

- handwritten manuscript recognition
- UI or productization
- cloud-scale serving
- generalized OCR for arbitrary languages and centuries
- unrestricted LLM rewriting of transcriptions

## Constraints

- total project budget: 175 hours
- implementation language: Python
- limited labeled data from project sources
- main text only; marginalia may be ignored
- inference speed is secondary to transcription accuracy
- all pipeline stages must remain debuggable and individually measurable

## Operating Assumptions

- PDFs can be converted to images without unacceptable loss
- partial transcriptions are sufficient for line-level supervised learning after alignment work
- historical spelling variation is valid and must not be indiscriminately normalized
- external public datasets may be used only if license-compatible and domain-relevant

## Acceptance Criteria

The initial baseline is acceptable only if it provides:

- reproducible training and inference commands
- deterministic preprocessing and segmentation outputs on a fixed sample
- reported CER and WER on a held-out evaluation split
- measured impact of weighted training vs unweighted baseline
- measured impact of constrained decoding vs greedy decoding
- auditable LLM correction logic with acceptance and rejection rules

## Success Metrics

Primary:

- Character Error Rate
- Word Error Rate

Secondary:

- exact line match rate
- rare-character precision, recall, and F1
- per-source performance variance
- percentage of LLM corrections accepted after validation

## Failure Conditions

The baseline should be treated as incomplete if:

- preprocessing is source-specific and non-reproducible
- evaluation is reported without clear splits
- decoder improvements are not isolated from model improvements
- LLM correction silently modernizes valid historical spelling
- metric claims cannot be reproduced from committed scripts and manifests
