# Implementation Roadmap

## Execution Strategy

Development proceeds in dependency order:

1. data ingestion
2. preprocessing and segmentation
3. baseline recognition
4. weighted training
5. constrained decoding
6. LLM post-correction
7. benchmark consolidation

This ordering minimizes hidden coupling and keeps regressions attributable.

## Phase Breakdown

### Phase 1: Repository and Data Foundations

- define manifests and metadata schemas
- implement PDF-to-image conversion
- establish split generation and baseline dataset checks

Deliverables:

- manifest format
- page export script
- split definition artifact

### Phase 2: Preprocessing and Segmentation

- implement normalization and skew correction
- implement main-text extraction
- implement line segmentation
- build visual inspection samples for failure review

Deliverables:

- deterministic preprocessing pipeline
- segmentation metadata
- sample inspection outputs

### Phase 3: Baseline OCR Model

- implement CRNN with CTC
- train baseline model
- produce baseline CER/WER report

Deliverables:

- train and infer commands
- baseline checkpoint
- first benchmark report

### Phase 4: Weighted Learning

- define rare-character weighting policy
- integrate weighted loss or equivalent rebalancing
- compare against baseline with fixed splits

Deliverables:

- weighting configuration
- ablation report

### Phase 5: Constrained Decoding

- implement beam search
- integrate Renaissance Spanish lexicon
- add confidence scoring and n-best outputs

Deliverables:

- decoder module
- lexicon format
- greedy vs beam comparison report

### Phase 6: LLM Post-Correction

- implement low-confidence gating
- define prompt and correction policy
- add validation and rejection rules

Deliverables:

- postprocessing module
- correction audit output
- benchmark delta report

### Phase 7: Hardening

- integration tests
- reproducibility validation
- final documentation and benchmark summary

Deliverables:

- regression test suite
- final metrics package
- usage and contribution documentation

## 12-Week Schedule

### Community Bonding

#### Week 1

- inspect source materials
- confirm evaluation expectations
- define transcription policy boundaries

#### Week 2

- finalize schemas and document assumptions
- prepare environment and experiment logging

### Coding

#### Week 3

- implement PDF ingestion
- generate page manifests

#### Week 4

- implement preprocessing and segmentation baseline
- review segmentation failure cases

#### Week 5

- implement CRNN baseline training
- produce first inference outputs

#### Week 6

- baseline evaluation and error analysis
- checkpoint: pre-midterm benchmark

#### Week 7

- integrate weighted learning
- compare rare-glyph metrics

#### Week 8

- implement constrained beam search
- checkpoint: midterm ablation report

#### Week 9

- add confidence estimates and top-k outputs
- prepare post-correction interface

#### Week 10

- integrate LLM correction with validation policy
- benchmark full pipeline

#### Week 11

- optimize thresholds and decoder scoring
- generate source-level report

#### Week 12

- finalize tests, documentation, and notebook/report artifacts
- freeze final benchmark package

## Milestone Gates

- Gate 1: manifests and segmentation reproducible
- Gate 2: CRNN baseline metrics established
- Gate 3: weighting gain measured
- Gate 4: constrained decoding gain measured
- Gate 5: LLM correction audited and benchmarked

No later phase should begin without artifacts from the previous gate.
