# AutoTRandHD

Historical OCR and transcription system for seventeenth-century Spanish printed sources.

This repository is documentation-first. The Markdown files listed below are the canonical specification for future implementation.

## Canonical Documents

- [README.md](README.md): repository entry point and document index
- [CONTRIBUTING.md](CONTRIBUTING.md): development workflow, review policy, and contribution rules
- [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md): scope, constraints, goals, non-goals, acceptance criteria
- [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md): data flow, module boundaries, interfaces, and invariants
- [docs/DATASET_AND_EVALUATION.md](docs/DATASET_AND_EVALUATION.md): dataset assumptions, preprocessing contract, metrics, and benchmark policy
- [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md): phased delivery plan and 12-week execution schedule

## Project Definition

AutoTRandHD targets OCR for early modern Spanish print where generic OCR fails due to:

- degraded scans
- non-standard glyphs and ligatures
- historical spelling variation
- decorative layouts and marginal noise

The planned system combines:

- deterministic PDF and image preprocessing
- main-text extraction and line segmentation
- weighted CNN-RNN text recognition with CTC training
- lexicon-constrained beam search decoding
- selective late-stage LLM correction for low-confidence outputs

## Repository Status

Current state:

- documentation baseline defined
- implementation not yet started
- repository structure prepared conceptually, not materially

Planned source tree:

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

## Development Priorities

1. Build deterministic data ingestion and segmentation.
2. Establish CRNN baseline with reproducible CER/WER reporting.
3. Add weighted training for rare glyphs and diacritics.
4. Add constrained beam search with historical lexicon support.
5. Integrate confidence-gated LLM correction.

## Technology Baseline

- Python 3.10+
- PyTorch
- OpenCV
- Pillow
- NumPy
- pandas
- jiwer
- PyMuPDF or pdf2image
- optional KenLM or equivalent for decoder language priors

## Execution Policy

- The documents in this repository are the primary source of truth.
- Code, configs, experiments, and tests must conform to the architecture and acceptance criteria defined in `docs/`.
- Any material design change must update the relevant Markdown document before or with the implementation change.

## License

Code: MIT, to be added when source code is introduced.

Dataset usage remains subject to the original source licenses and access restrictions.
