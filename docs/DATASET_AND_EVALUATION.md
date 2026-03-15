# Dataset and Evaluation

## Supported Data Scope

Initial scope is limited to scanned printed Spanish sources from the seventeenth century or closely related early modern print with compatible typography.

Primary project data:

- scanned PDF sources
- partial transcriptions used as ground truth

Optional augmentation sources:

- public historical OCR datasets with compatible license and typography

## Data Policy

- raw project PDFs are authoritative inputs
- transcriptions must remain versioned and traceable to source pages
- all derived manifests must be reproducible from scripts
- no benchmark result is valid without an explicit split definition

## Preprocessing Contract

The preprocessing stage must produce:

- page image at fixed target DPI
- normalized grayscale output
- metadata describing transforms applied
- line bounding boxes in source-page coordinates

Ignored content:

- marginalia
- decorative borders
- isolated ornaments
- page furniture not part of main text

## Label Policy

Ground truth should preserve historical spelling unless a documented normalization policy says otherwise.

Rules:

- keep glyph distinctions that are visible and semantically relevant
- avoid silently modernizing spelling
- document any collapsed classes explicitly
- note ambiguous regions instead of inventing normalized text

## Split Strategy

Minimum split requirement:

- training split
- validation split
- held-out test split

Preferred strategy:

- split by source or by non-overlapping page ranges to avoid leakage from near-identical neighboring pages

The repository must document:

- number of pages per split
- number of line samples per split
- source IDs included in each split

## Metrics

### Primary Metrics

- Character Error Rate (CER)
- Word Error Rate (WER)

### Secondary Metrics

- exact line match rate
- rare-character precision
- rare-character recall
- rare-character F1
- source-level variance

### Decoder and Postprocessing Metrics

- greedy vs beam CER/WER delta
- lexicon hit rate
- LLM invocation rate
- LLM acceptance rate
- rejected-correction count by reason

## Benchmark Policy

Every benchmark report must state:

- model checkpoint identifier
- decoder configuration
- lexicon version
- preprocessing configuration
- evaluation split identifier

No benchmark claim is valid if any of the above is missing.

## Error Analysis Requirements

At minimum, each major experiment should include:

- top character confusions
- rare-glyph error summary
- source-level failure cases
- qualitative examples where lexicon constraints help
- qualitative examples where LLM correction helps or fails

## Target Quality Threshold

Project target:

- approach or exceed 90% transcription accuracy on held-out data

Operational interpretation:

- report CER/WER explicitly
- if a single “accuracy” number is used, define the formula in the report

## Reproducibility Requirements

Must be possible to regenerate:

- manifests
- splits
- line crops
- training metrics
- evaluation reports

from committed scripts plus documented external data inputs.
