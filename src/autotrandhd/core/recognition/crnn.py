"""src.models.crnn — Convolutional-Recurrent Neural Network for OCR.

Architecture
------------
The model follows the CRNN design (Shi et al., 2016):

    Input line crop (grayscale, H × W)
        ↓
    CNN encoder  — extracts visual features; output shape: (T, B, C)
        ↓
    Bi-directional LSTM — models sequential dependencies
        ↓
    Linear projection  — maps to vocabulary logits

The output is a sequence of per-step log-probabilities suitable for CTC
decoding.  ``blank_idx = 0`` by convention.

Typical usage::

    from autotrandhd.core.recognition.crnn import CRNN

    model = CRNN(num_classes=97, img_height=64)
    # x: (B, 1, H, W)
    logits = model(x)  # → (T, B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CRNN(nn.Module):
    """Convolutional-Recurrent OCR model.

    Parameters
    ----------
    num_classes:
        Size of the output vocabulary, including the CTC blank token (index 0).
    img_height:
        Fixed height of the input line crops in pixels.
    cnn_channels:
        List of channel widths for each CNN stage.
        The first value must equal 1 (single-channel grayscale input).
    rnn_hidden:
        Number of hidden units per direction in the bidirectional LSTM.
    rnn_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers.
    """

    def __init__(
        self,
        num_classes: int,
        img_height: int = 64,
        cnn_channels: list[int] | None = None,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [1, 64, 128, 256, 256, 512, 512, 512]

        if cnn_channels[0] != 1:
            raise ValueError("cnn_channels[0] must be 1 (grayscale input).")

        self.cnn = _build_cnn(cnn_channels, img_height)

        # After the CNN the feature map height is collapsed to 1; width
        # becomes the time dimension.
        cnn_out_channels = cnn_channels[-1]

        self.rnn = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=False,
        )

        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, 1, H, W)`` — batch of grayscale line
            crops, normalised to ``[0, 1]``.

        Returns
        -------
        Tensor
            Log-probabilities of shape ``(T, B, num_classes)``, where *T* is
            the sequence length determined by the CNN stride.
        """
        # CNN: (B, 1, H, W) → (B, C, 1, T)
        features = self.cnn(x)

        # Squeeze height dim and permute to (T, B, C)
        b, c, h, t = features.shape
        assert h == 1, (
            f"CNN output height must be 1 but got {h}. "
            "Check img_height and CNN architecture."
        )
        features = features.squeeze(2)       # (B, C, T)
        features = features.permute(2, 0, 1) # (T, B, C)

        # RNN: (T, B, C) → (T, B, 2*rnn_hidden)
        rnn_out, _ = self.rnn(features)

        # Classifier: (T, B, 2*rnn_hidden) → (T, B, num_classes)
        logits = self.classifier(rnn_out)
        return torch.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# CNN builder
# ---------------------------------------------------------------------------

def _build_cnn(channels: list[int], img_height: int) -> nn.Sequential:
    """Build a VGG-style CNN that maps a line image to a feature sequence.

    The network halves the spatial dimensions (H and W) at each pooling layer
    until H is reduced to 1.  The number of pooling operations is derived from
    ``img_height``.

    Architecture per stage:
        Conv(in, out, 3×3) → BatchNorm → ReLU → optional MaxPool(2×2)

    Pooling is applied after stages 1, 2, and then every second stage
    (height-only pool for the last stages to preserve T).
    """
    layers: list[nn.Module] = []

    # Pool schedule: each entry is (pool_h, pool_w, stride_h, stride_w).
    # We pool height aggressively (to reach H=1) but width conservatively
    # to retain sequence resolution.
    pool_schedule = [
        (2, 2, 2, 2),  # after stage 1 → H/2
        (2, 2, 2, 2),  # after stage 2 → H/4
        (2, 1, 2, 1),  # after stage 3 → H/8
        (2, 1, 2, 1),  # after stage 4 → H/16
        (2, 1, 2, 1),  # after stage 5 → H/32
        (2, 1, 2, 1),  # after stage 6 → H/64
    ]

    stage_count = len(channels) - 1
    for i in range(stage_count):
        in_c = channels[i]
        out_c = channels[i + 1]
        layers += [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if i < len(pool_schedule):
            ph, pw, sh, sw = pool_schedule[i]
            layers.append(nn.MaxPool2d(kernel_size=(ph, pw), stride=(sh, sw)))

    return nn.Sequential(*layers)
