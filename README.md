# AutoTRandHD

AutoTRandHD is an OCR pipeline for seventeenth-century Spanish printed material. It handles page cleanup, text-region extraction, line segmentation, CRNN inference, and decoding behind a small CLI and FastAPI surface.

## Layout

```text
src/autotrandhd/
  api/         FastAPI application and request schemas
  config/      environment-driven runtime settings
  core/        OCR domain primitives: preprocessing, recognition, decoding
  services/    orchestration layer for model loading and inference
  utils/       small runtime and image helpers
api/           thin server wrapper
cli/           thin CLI wrapper
tests/         critical-path tests
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
autotrandhd-cli info
python api/server.py
pytest -q
```

## Key Decisions

- Core OCR logic stays isolated from transport concerns so the CLI and API share one inference path.
- Settings come from environment variables first; defaults are good enough for local runs.
- The API warms the model on startup when a checkpoint is present, but still boots without one for health checks and local integration work.

## Notes

- The benchmark script falls back to a randomly initialized CRNN when no checkpoint is available. That is deliberate so performance plumbing can still be checked on a clean machine.
- Canonical project docs remain in `docs/` for scope, architecture, and evaluation policy.
