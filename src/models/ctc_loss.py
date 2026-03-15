"""src.models.ctc_loss — CTC loss wrapper with length computation.

PyTorch's ``nn.CTCLoss`` requires the predicted sequence length (``input_lengths``)
and target sequence length (``target_lengths``) as explicit tensors.  This
wrapper computes those lengths automatically from the logit tensor and the
target tensor, reducing boilerplate in the training loop.

Typical usage::

    from src.models.ctc_loss import CTCLossWrapper

    criterion = CTCLossWrapper(blank=0, reduction="mean")
    # log_probs: (T, B, C) — output of CRNN.forward()
    # targets: (B, S) or 1-D concatenated target tensor
    # target_lengths: (B,) — number of characters in each target sequence
    loss = criterion(log_probs, targets, target_lengths)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CTCLossWrapper(nn.Module):
    """Wrapper around ``nn.CTCLoss`` that derives ``input_lengths`` automatically.

    Parameters
    ----------
    blank:
        Index of the CTC blank token.  Must match the vocabulary convention
        used in the decoder (default: 0).
    reduction:
        ``"mean"`` or ``"sum"``.  Passed directly to ``nn.CTCLoss``.
    zero_infinity:
        If ``True``, zero out infinite losses (useful for very long sequences).
    """

    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
    ) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Compute CTC loss.

        Parameters
        ----------
        log_probs:
            Log-probabilities from the model, shape ``(T, B, C)``.
        targets:
            Ground-truth token indices.  Either:

            - shape ``(B, S)`` — padded targets (max length S), or
            - shape ``(sum(target_lengths),)`` — concatenated, unpadded.
        target_lengths:
            Number of valid tokens in each target, shape ``(B,)``.

        Returns
        -------
        Tensor
            Scalar CTC loss.
        """
        T, B, _ = log_probs.shape
        input_lengths = torch.full(
            (B,), T, dtype=torch.long, device=log_probs.device
        )

        # Pass targets directly; nn.CTCLoss supports both 1-D concatenated
        # targets of length sum(target_lengths) and 2-D padded targets of
        # shape (B, S), using target_lengths to determine the valid tokens.
        return self.ctc(log_probs, targets, input_lengths, target_lengths)
