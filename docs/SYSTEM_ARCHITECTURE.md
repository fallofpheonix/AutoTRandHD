# System Architecture

## Architecture Summary

```text
PDF
  -> page images
  -> page normalization
  -> main-text extraction
  -> line segmentation
  -> CRNN recognizer
  -> constrained beam search
  -> confidence filter
  -> optional LLM correction
  -> validated transcription
  -> evaluation report
```

## Module Boundaries

### `src/data`

Responsibilities:

- PDF ingestion
- page-image export
- manifest generation
- transcription loading
- split definition

Inputs:

- raw PDFs
- transcription files

Outputs:

- page manifests
- line metadata
- train/validation/test splits

### `src/preprocess`

Responsibilities:

- grayscale conversion
- denoising
- contrast normalization
- skew correction
- main-text extraction
- line segmentation

Inputs:

- page images

Outputs:

- normalized page images
- line crops
- segmentation metadata

### `src/models`

Responsibilities:

- CNN encoder
- recurrent sequence model
- CTC training loop
- checkpoint save/load

Inputs:

- line crops
- aligned transcripts

Outputs:

- logits
- checkpoints
- training summaries

### `src/decode`

Responsibilities:

- greedy decoding
- beam search
- lexicon scoring
- optional character language prior integration
- confidence estimation

Inputs:

- model logits
- vocabulary
- lexicon

Outputs:

- decoded text
- n-best hypotheses
- confidence metadata

### `src/postprocess`

Responsibilities:

- LLM request construction
- low-confidence span selection
- validation of candidate corrections

Inputs:

- n-best OCR hypotheses
- confidence signals
- lexicon and policy constraints

Outputs:

- corrected text
- audit trail of accepted and rejected edits

### `src/eval`

Responsibilities:

- CER and WER computation
- exact-match metrics
- rare-character analysis
- per-source benchmark reports

Inputs:

- predictions
- ground truth

Outputs:

- metric reports
- error summaries

## Interface Contracts

### Page Manifest

Minimum fields:

- `source_id`
- `page_id`
- `image_path`
- `dpi`
- `split`

### Line Metadata

Minimum fields:

- `source_id`
- `page_id`
- `line_id`
- `bbox`
- `image_path`
- `transcript`

### Decoder Output

Minimum fields:

- `line_id`
- `top1_text`
- `topk_candidates`
- `token_confidence`
- `sequence_confidence`

### LLM Correction Audit

Minimum fields:

- `line_id`
- `input_text`
- `candidate_text`
- `decision`
- `decision_reason`

## Invariants

- preprocessing must not mutate raw input files
- line segmentation outputs must be traceable back to source page coordinates
- decoder output must be reproducible from saved logits and decoding parameters
- LLM corrections must be optional and auditable
- evaluation must operate on immutable prediction and ground-truth snapshots

## Failure Isolation

To keep debugging bounded:

- preprocessing failures must be diagnosable without model execution
- decoder failures must be reproducible from stored logits
- postprocessing failures must be reproducible from stored candidate sets
- each stage must expose intermediate artifacts for inspection

## Initial Directory Plan

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
artifacts/
notebooks/
```

`artifacts/` should store derived outputs only when reproducibility or review requires them. Large generated files should remain excluded from version control.
