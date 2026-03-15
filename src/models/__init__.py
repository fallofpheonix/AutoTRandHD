"""src.models — CRNN OCR model, CTC loss, training loop, and inference.

Public API
----------
- crnn.CRNN               : CNN encoder + bi-RNN sequence model
- ctc_loss.CTCLossWrapper : CTC loss with input/target length handling
- trainer.Trainer         : training and validation loop
- inference.transcribe    : run a trained model on line-crop images
"""

from .crnn import CRNN
from .ctc_loss import CTCLossWrapper
from .trainer import Trainer
from .inference import transcribe

__all__ = [
    "CRNN",
    "CTCLossWrapper",
    "Trainer",
    "transcribe",
]
