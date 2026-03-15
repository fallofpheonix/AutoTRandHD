# Contributing

## Purpose

This repository is being built as a reproducible historical OCR system. Contributions must improve implementation quality, reproducibility, or documentation precision.

## Source of Truth

Before adding code, read:

- [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md)
- [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)
- [docs/DATASET_AND_EVALUATION.md](docs/DATASET_AND_EVALUATION.md)
- [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md)

If a change conflicts with those documents, update the documents first or in the same commit.

## Contribution Rules

- keep preprocessing and decoding deterministic by default
- avoid framework-heavy abstractions
- preserve module boundaries defined in the architecture document
- add tests for every non-trivial new module
- include metrics for model or decoder changes
- do not merge undocumented dataset assumptions

## Branching

- use short descriptive branches
- keep commits atomic
- separate documentation-only and code changes when practical

## Required Pull Request Content

Every implementation PR should state:

- what changed
- which canonical document sections it implements or modifies
- metric impact, if any
- test coverage added or updated
- known limitations

## Testing Expectations

Minimum requirements by change type:

- preprocessing: unit tests plus at least one integration sample
- modeling: training/inference smoke test and metric report
- decoding: deterministic regression tests on fixed logits
- LLM postprocessing: mocked interface tests and validation checks
- evaluation: golden-file tests for metric correctness

## Dataset Handling

- do not commit restricted raw datasets unless licensing explicitly permits it
- store manifests, metadata, and reproducible processing scripts instead of opaque outputs
- document any public dataset additions in `docs/DATASET_AND_EVALUATION.md`

## Documentation Policy

When changing:

- scope or acceptance criteria: update `docs/PROJECT_SCOPE.md`
- modules or interfaces: update `docs/SYSTEM_ARCHITECTURE.md`
- splits, metrics, or benchmarking rules: update `docs/DATASET_AND_EVALUATION.md`
- milestones or delivery sequencing: update `docs/IMPLEMENTATION_ROADMAP.md`

## Out of Scope for Initial Contributions

- handwritten-text pipeline
- end-user web UI
- broad multilingual support
- model-serving infrastructure

These may be added later only after the printed-source baseline is stable.
