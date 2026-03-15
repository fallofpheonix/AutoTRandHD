# artifacts/

This directory stores **derived outputs** produced by the pipeline.

Large binary files (images, checkpoints, logits, numpy arrays) are excluded
from version control via `.gitignore`.  Only small, human-readable summary
files (JSON reports, YAML configs, CSV metrics) should be committed.

Sub-directories created at runtime:

- `checkpoints/` — model checkpoint files saved by the trainer
- `reports/`     — evaluation metric reports
- `logits/`      — saved decoder logits for reproducible replay
- `line_crops/`  — exported line-crop images and metadata
