# GSoC 2026 Proposal: HumanAI Task 2

## 1. Project Title

**Hybrid CNN-RNN OCR with Lexicon-Constrained Decoding and LLM Post-Correction for Seventeenth-Century Spanish Print**

## 2. Abstract / Overview

Current OCR systems are optimized for modern print and fail on seventeenth-century Spanish sources because of degraded scans, historical glyph variants, irregular orthography, decorative layouts, and non-standard spacing. The proposed project will build a practical transcription pipeline for early modern printed Spanish using a weighted convolutional-recurrent recognizer, constrained decoding with a Renaissance Spanish lexicon, and a late-stage LLM correction module.

The pipeline will begin with PDF ingestion, page preprocessing, and main-text extraction to suppress marginalia and ornaments. A CRNN model trained with CTC loss will perform line-level recognition. Weighted learning will improve recognition of rare letterforms, ligatures, and diacritics that are systematically underrepresented in training data. Decoding will use beam search constrained by a historical lexicon to reduce hallucinated outputs and improve word-level consistency. An LLM such as Gemini will then revise low-confidence spans while respecting OCR confidence and historical spelling constraints.

The deliverable is a reproducible open-source baseline for printed historical Spanish OCR, with evaluation on held-out pages using CER, WER, and rare-character metrics. The target is a usable transcription system approaching or exceeding 90% accuracy on the provided sources.

## 3. Problem Statement

Historical printed Spanish is a difficult OCR setting because errors are driven by both document degradation and domain mismatch.

- early modern typefaces contain long-s forms, ligatures, uncommon abbreviations, and inconsistent punctuation
- scans may contain skew, bleed-through, low contrast, ink spread, and decorative initials
- spelling is historically variable, so naive language correction can destroy valid forms
- marginalia, page numbers, headers, and ornaments interfere with line detection and recognition

Current OCR tools such as generic PDF OCR engines are insufficient for this task because they assume modern typography, clean layout structure, and standard vocabulary. Their language priors bias outputs toward contemporary spellings, which is especially harmful when visual evidence is weak. Rare graphemes are often collapsed into frequent modern substitutes. The result is not just noisy transcription, but systematically incorrect text that is difficult to post-correct reliably.

This matters to RenAIssance because transcription quality is the entry point for all downstream tasks: indexing, search, corpus analysis, philological study, and model extension to more difficult handwritten materials. A domain-adapted recognizer for printed sources provides the lowest-risk path to an operational historical text recognition pipeline.

## 4. Project Constraints and Assumptions

- total implementation budget: 175 hours
- language and tooling: Python-based ML stack
- focus restricted to printed sources, not handwritten manuscripts
- main text only; marginalia can be ignored
- labeled data is limited to partial transcriptions from the supplied PDFs, optionally supplemented by compatible public historical OCR data
- inference speed is secondary to accuracy and reproducibility
- LLM integration must be optional and isolated behind a stable interface
- evaluation must distinguish between visually wrong characters and historically valid spelling variants
- system should support per-source tuning without source-specific rewrites

Assumptions:

- PDFs can be converted into page images without significant information loss
- enough transcription alignment can be produced at line level for supervised training
- the historical lexicon can be constructed from transcriptions plus external early modern Spanish sources
- target accuracy will be measured explicitly using CER, WER, and exact match statistics rather than an undefined single score

## 5. Proposed Solution

### Core Components

- document ingestion and page normalization
- main-text detection and line segmentation
- weighted CNN-RNN recognizer trained with CTC
- constrained beam search decoder with historical lexicon
- confidence-aware LLM post-correction
- evaluation and error-analysis toolkit

### System Architecture

```text
PDF -> page images -> preprocess -> main-text regions -> line crops
    -> CRNN recognizer -> character probabilities
    -> constrained beam search -> n-best hypotheses + confidence
    -> selective LLM correction -> validated transcription
    -> evaluation reports
```

### Data Flow

1. Convert each PDF into page images at a fixed DPI.
2. Normalize pages using grayscale conversion, denoising, skew correction, and contrast adjustment.
3. Detect and retain only the main text region using morphology, projection profiles, and connected-component filtering.
4. Segment retained blocks into line images.
5. Train a CRNN model:
   - CNN encoder extracts visual features
   - bidirectional recurrent layers model sequence context
   - CTC loss aligns logits to transcriptions without explicit character segmentation
6. Apply weighted loss or class reweighting to emphasize rare glyphs and diacritics.
7. Decode model outputs with beam search constrained by a Renaissance Spanish lexicon and optional character-level language prior.
8. For low-confidence lines, pass top-k hypotheses and confidence maps to an LLM for bounded correction.
9. Reject LLM outputs that violate validation rules such as excessive edit distance or unsupported modernization.
10. Export line-level and document-level transcriptions with metrics.

### Algorithms and Libraries

- PyTorch for model training
- OpenCV and Pillow for image preprocessing
- jiwer or equivalent for CER/WER evaluation
- PyMuPDF or pdf2image for PDF ingestion
- optional KenLM-style character language model for decoder scoring
- Gemini API or equivalent for post-correction

### Why This Architecture

- CRNN+CTC is data-efficient relative to full transformer OCR on small labeled corpora
- weighting directly addresses rare-class imbalance instead of hoping the decoder corrects it
- lexicon-constrained decoding improves word plausibility before LLM intervention
- LLM is used narrowly where it is most useful: ambiguity resolution, not raw visual transcription

## 6. Technical Methodology

### System Modules

- `data`
  - PDF conversion, dataset manifests, split generation, transcription alignment
- `preprocess`
  - binarization, denoising, skew correction, text-region extraction, line segmentation
- `models`
  - CRNN encoder-decoder, loss functions, checkpointing
- `decode`
  - greedy and beam decoders, lexicon scorer, confidence estimation
- `postprocess`
  - prompt builder, LLM client, output validator
- `eval`
  - CER/WER, exact line match, rare-character confusion analysis

### Implementation Strategy

- establish a deterministic baseline before introducing learned postprocessing
- treat preprocessing, recognition, decoding, and correction as independently testable modules
- build ablations to measure incremental gain from:
  - weighted training
  - constrained decoding
  - LLM correction
- keep the LLM path optional so the non-LLM OCR baseline remains usable

### Testing Approach

- unit tests for dataset parsing, lexicon loading, and decoding invariants
- integration tests for page-to-line and line-to-text stages
- fixed validation subset for regression tracking
- manual error analysis focused on frequent confusion pairs and rare glyphs

### Evaluation Metrics

- Character Error Rate (primary)
- Word Error Rate
- exact line match rate
- rare-character precision, recall, and F1
- per-source performance variance
- decoder gain over greedy baseline
- accepted vs rejected LLM corrections

## 7. Implementation Plan

### Phase 1 – Research and System Design

- inspect source PDFs and transcription format
- define transcription policy and normalization boundaries
- build dataset manifest and split strategy
- prototype preprocessing and line extraction

### Phase 2 – Core Implementation

- implement CRNN baseline with CTC loss
- train first recognizer on segmented line images
- create evaluation scripts and baseline reports

### Phase 3 – Feature Integration

- add weighted learning for rare classes
- implement lexicon-constrained beam search
- generate n-best hypotheses and confidence scores

### Phase 4 – Optimization

- integrate LLM post-correction for low-confidence spans
- add validation guards to prevent over-correction
- tune thresholds and decoder weights through ablation

### Phase 5 – Testing and Documentation

- add unit and integration tests
- prepare reproducible training/inference commands
- finalize notebook, benchmark report, and documentation

## 8. Project Roadmap (GSoC Timeline)

### Community Bonding

**Week 1**
- review project material and evaluation expectations
- confirm dataset organization and required deliverables

**Week 2**
- finalize technical design
- set up environment, manifests, and experiment logging

### Coding Period

**Week 3**
- implement PDF ingestion and dataset loader
- deliverable: reproducible data preparation pipeline

**Week 4**
- implement preprocessing, text extraction, and line segmentation
- deliverable: segmented line dataset from source PDFs

**Week 5**
- implement CRNN training loop and baseline inference
- deliverable: first end-to-end recognizer

**Week 6**
- baseline evaluation and error analysis
- checkpoint: pre-midterm CER/WER baseline

**Week 7**
- implement weighted loss and rare-character handling
- deliverable: improved rare-glyph metrics

**Week 8**
- implement constrained beam search with historical lexicon
- checkpoint: midterm evaluation and ablation report

**Week 9**
- add confidence estimation and n-best export
- deliverable: uncertainty-aware decoding interface

**Week 10**
- integrate Gemini-based post-correction with validation
- deliverable: full hybrid OCR pipeline

**Week 11**
- optimize thresholds and run held-out benchmark
- checkpoint: final metrics and source-level analysis

**Week 12**
- documentation, tests, notebook export, and final report
- deliverable: submission-ready repository and reproducible outputs

## 9. Repository README Draft

### Project Description

Historical OCR pipeline for seventeenth-century Spanish printed sources using a weighted CRNN recognizer, lexicon-constrained decoding, and late-stage LLM correction.

### Key Features

- main-text extraction from scanned PDFs
- CRNN-based line recognition
- rare-glyph weighting
- constrained beam search with historical lexicon
- optional Gemini-based post-correction
- CER/WER evaluation and source-level reports

### Installation Instructions

```bash
git clone https://github.com/<user>/renaissance-historical-ocr.git
cd renaissance-historical-ocr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage Guide

```bash
python -m src.data.build_manifest
python -m src.preprocess.segment_pages --input data/raw_pdfs --output data/lines
python -m src.train --config configs/crnn_base.yaml
python -m src.infer --checkpoint checkpoints/best.pt --input data/raw_pdfs/source_01.pdf
python -m src.eval.run --pred outputs/source_01.txt --gt data/transcripts/source_01_gt.txt
```

### Project Architecture

```text
src/
  data/
  preprocess/
  models/
  decode/
  postprocess/
  eval/
tests/
configs/
notebooks/
```

### Contribution Guidelines

- open issues for bugs or feature requests
- keep modules small and testable
- document assumptions about transcription policy and historical spelling
- add tests for new preprocessing, decoding, or evaluation logic

### License

- code: MIT
- datasets: subject to source licensing terms

## 10. Expected Outcomes and Impact

- a reproducible OCR baseline specialized for early modern Spanish print
- measurable improvement over generic OCR tools on provided sources
- evidence-backed analysis of how weighting, constrained decoding, and LLM correction affect accuracy
- reduced manual transcription effort for RenAIssance collaborators
- a modular base for later manuscript and multilingual extensions

## 11. Risks and Mitigation Strategies

### Risk: Limited labeled data
- mitigation: transfer learning, augmentation, conservative model size, strict validation splits

### Risk: Layout variability across sources
- mitigation: modular preprocessing with per-source configuration and manual ROI fallback

### Risk: Rare-glyph collapse
- mitigation: weighted loss, targeted augmentation, confusion-matrix-driven tuning

### Risk: LLM hallucination or over-normalization
- mitigation: confidence-triggered usage, edit-distance caps, lexicon and policy checks, rejection of unsafe edits

### Risk: Ground-truth alignment noise
- mitigation: manifest validation scripts and sampled manual verification

### Risk: External API dependency
- mitigation: keep LLM module optional and maintain a complete non-LLM baseline

## 12. Future Work

- extension to handwritten manuscripts
- active learning with human correction loops
- TEI/XML export for digital humanities workflows
- multilingual early modern OCR support
- source-adaptive fine-tuning for new printers and typefaces
