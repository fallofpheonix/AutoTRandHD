"""src.models.inference — deterministic transcription from a trained CRNN.

This module provides a single entry point (``transcribe``) that:

1. Loads a checkpoint into a model.
2. Preprocesses a list of line-crop image paths.
3. Runs inference in evaluation mode.
4. Optionally saves the raw logits for reproducible decoder replay.

Logits are saved as ``.npy`` files so the decoder can be re-run without
re-running the model.

Typical usage::

    from src.models.inference import transcribe

    results = transcribe(
        image_paths=["artifacts/line_crops/source_001_p0000_l0000.png"],
        checkpoint_path="artifacts/checkpoints/checkpoint_epoch0050.pt",
        num_classes=97,
        img_height=64,
        logits_dir="artifacts/logits",
    )
    for r in results:
        print(r["image_path"], r["logits_path"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from .crnn import CRNN

logger = logging.getLogger(__name__)


def transcribe(
    image_paths: List[str | Path],
    checkpoint_path: str | Path,
    num_classes: int,
    img_height: int = 64,
    cnn_channels: Optional[List[int]] = None,
    rnn_hidden: int = 256,
    rnn_layers: int = 2,
    device: str = "cpu",
    logits_dir: Optional[str | Path] = None,
    batch_size: int = 16,
) -> List[dict]:
    """Run inference on *image_paths* using a saved checkpoint.

    Parameters
    ----------
    image_paths:
        Ordered list of line-crop image paths.
    checkpoint_path:
        Path to a ``.pt`` checkpoint produced by :class:`Trainer`.
    num_classes:
        Vocabulary size including blank token.
    img_height:
        Line image height expected by the model.
    cnn_channels:
        CNN channel list.  Defaults to ``[1, 64, 128, 256, 256, 512, 512, 512]``.
    rnn_hidden:
        LSTM hidden units per direction.
    rnn_layers:
        Number of LSTM layers.
    device:
        Compute device.
    logits_dir:
        If provided, save per-image logit arrays as ``.npy`` files here.
    batch_size:
        Number of images to process per forward pass.

    Returns
    -------
    List[dict]
        One record per image with keys:
        ``image_path``, ``logits_path`` (or ``None``), ``logits_shape``.
    """
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("OpenCV is required for image loading.") from exc

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = _load_model(
        checkpoint_path, num_classes, img_height, cnn_channels, rnn_hidden,
        rnn_layers, device,
    )

    if logits_dir is not None:
        logits_dir = Path(logits_dir)
        logits_dir.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    dev = torch.device(device)

    # Process in batches.
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start: batch_start + batch_size]
        tensors = []

        for p in batch_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {p}")
            tensor = _preprocess(img, img_height)
            tensors.append(tensor)

        # Stack; images may have different widths so we pad to max width.
        batch_tensor = _pad_batch(tensors).to(dev)  # (B, 1, H, W_max)

        with torch.no_grad():
            log_probs = model(batch_tensor)  # (T, B, C)

        for i, p in enumerate(batch_paths):
            logit_array = log_probs[:, i, :].cpu().numpy()  # (T, C)
            logits_path: Optional[Path] = None
            if logits_dir is not None:
                stem = Path(p).stem
                logits_path = logits_dir / f"{stem}_logits.npy"
                np.save(str(logits_path), logit_array)

            results.append(
                {
                    "image_path": str(p),
                    "logits_path": str(logits_path) if logits_path else None,
                    "logits_shape": list(logit_array.shape),
                }
            )

    logger.info("Transcribed %d images.", len(results))
    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_model(
    checkpoint_path: Path,
    num_classes: int,
    img_height: int,
    cnn_channels: Optional[List[int]],
    rnn_hidden: int,
    rnn_layers: int,
    device: str,
) -> CRNN:
    model = CRNN(
        num_classes=num_classes,
        img_height=img_height,
        cnn_channels=cnn_channels,
        rnn_hidden=rnn_hidden,
        rnn_layers=rnn_layers,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def _preprocess(img: "np.ndarray", target_height: int) -> Tensor:
    """Resize *img* to *target_height* and normalise to [0, 1]."""
    import cv2

    h, w = img.shape[:2]
    if h != target_height:
        scale = target_height / h
        new_w = max(1, int(w * scale))
        img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img).float() / 255.0  # (H, W)
    return tensor.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)


def _pad_batch(tensors: List[Tensor]) -> Tensor:
    """Pad a list of ``(1, 1, H, W_i)`` tensors to the same width."""
    max_w = max(t.shape[-1] for t in tensors)
    padded = []
    for t in tensors:
        pad_w = max_w - t.shape[-1]
        if pad_w > 0:
            t = torch.nn.functional.pad(t, (0, pad_w), value=0.0)
        padded.append(t)
    return torch.cat(padded, dim=0)  # (B, 1, H, W_max)
