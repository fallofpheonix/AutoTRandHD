# AutoTRandHD

Hybrid OCR and transcription pipeline for seventeenth-century Spanish printed sources using a weighted CNN-RNN recognizer, lexicon-constrained decoding, and late-stage LLM correction.

## Project Scope

This repository contains the technical proposal and implementation plan for a HumanAI / RenAIssance GSoC project focused on historical text recognition from early modern Spanish print.

Primary goals:

- detect and isolate main text from scanned historical pages
- train a domain-adapted recognizer for line-level transcription
- improve rare glyph recognition with weighted learning
- reduce decoder hallucinations using a Renaissance Spanish lexicon
- integrate an LLM as a constrained post-correction stage

## Planned Features

- PDF-to-image conversion and dataset manifest generation
- main-text extraction with marginalia suppression
- line segmentation for OCR training and inference
- CRNN recognizer with CTC training
- weighted loss for rare glyphs, ligatures, and diacritics
- constrained beam search decoding
- optional Gemini-based post-correction for low-confidence spans
- evaluation with CER, WER, exact line match, and rare-character recall

## Repository Layout

```text
.
├── README.md
└── docs/
    └── GSoC_2026_HumanAI_Proposal.md
```

Planned implementation layout:

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

## Installation

Planned Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected core dependencies:

- Python 3.10+
- PyTorch
- OpenCV
- Pillow
- NumPy
- pandas
- jiwer
- PyMuPDF or pdf2image

## Intended Workflow

1. Convert source PDFs into page images.
2. Extract main text blocks and segment pages into line crops.
3. Train a CRNN recognizer on aligned line/transcription pairs.
4. Decode with beam search constrained by a Renaissance Spanish lexicon.
5. Apply LLM correction only on uncertain outputs.
6. Evaluate against held-out transcriptions.

## Usage

This repository currently stores the proposal and project plan. Code modules listed above are planned deliverables for the GSoC implementation period.

## Contribution Guidelines

- keep preprocessing and decoding modules deterministic
- document dataset assumptions and transcription policy
- add tests for all non-trivial pipeline stages
- report metrics with source-level breakdowns

## License

Code: MIT License.

Datasets and source scans remain subject to their original licenses and usage restrictions.
