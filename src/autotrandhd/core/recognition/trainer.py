"""src.models.trainer — training and validation loop for the CRNN model.

The trainer is intentionally minimal: it handles the core epoch loop, metric
logging, and checkpoint save/load.  Dataset construction, data loaders, and
vocabulary management remain external to keep this module composable.

Typical usage::

    from autotrandhd.core.recognition.trainer import Trainer
    from autotrandhd.core.recognition.crnn import CRNN
    from autotrandhd.core.recognition.ctc_loss import CTCLossWrapper

    model = CRNN(num_classes=97)
    criterion = CTCLossWrapper()
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimiser=optimiser,
        checkpoint_dir="artifacts/checkpoints",
        device="cuda",
    )
    trainer.fit(train_loader, val_loader, num_epochs=50)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Training and validation loop for a CTC-trained CRNN model.

    Parameters
    ----------
    model:
        The CRNN model to train.
    criterion:
        CTC loss criterion (e.g. ``CTCLossWrapper``).
    optimiser:
        PyTorch optimiser.
    checkpoint_dir:
        Directory for saving/loading checkpoints.
    device:
        Compute device (``"cpu"``, ``"cuda"``, ``"mps"``).
    grad_clip:
        Maximum gradient norm.  ``0`` disables clipping.
    log_interval:
        Log training loss every *N* batches.
    seed:
        Random seed applied before training for reproducibility.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimiser: Optimizer,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
        device: str = "cpu",
        grad_clip: float = 5.0,
        log_interval: int = 10,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.seed = seed

        self._set_seed(seed)
        self.model.to(self.device)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        val_interval: int = 1,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader:
            DataLoader yielding ``(images, targets, target_lengths)`` batches.
        val_loader:
            Optional validation DataLoader.
        num_epochs:
            Total number of training epochs.
        val_interval:
            Validate every *N* epochs.

        Returns
        -------
        dict
            Training summary with keys ``train_losses``, ``val_losses``.
        """
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(train_loss)

            val_loss = None
            if val_loader is not None and epoch % val_interval == 0:
                val_loss = self._val_epoch(val_loader)
                val_losses.append(val_loss)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d — train_loss: %.4f — val_loss: %s — %.1fs",
                epoch,
                num_epochs,
                train_loss,
                f"{val_loss:.4f}" if val_loss is not None else "—",
                elapsed,
            )

            self.save_checkpoint(epoch, extra={"train_loss": train_loss})

        summary = {"train_losses": train_losses, "val_losses": val_losses}
        self._save_summary(summary)
        return summary

    def save_checkpoint(self, epoch: int, extra: Optional[Dict] = None) -> Path:
        """Save a model checkpoint.

        Parameters
        ----------
        epoch:
            Current epoch number (used in the filename).
        extra:
            Optional dict of extra metadata to embed in the checkpoint.

        Returns
        -------
        Path
            Path of the saved checkpoint file.
        """
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, ckpt_path)
        logger.debug("Saved checkpoint → %s", ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a model checkpoint from *path*.

        Parameters
        ----------
        path:
            Path to a ``.pt`` checkpoint file.

        Returns
        -------
        int
            The epoch number stored in the checkpoint.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimiser.load_state_dict(payload["optimiser_state_dict"])
        epoch = int(payload.get("epoch", 0))
        logger.info("Loaded checkpoint from %s (epoch %d)", path, epoch)
        return epoch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader, 1):
            images: Tensor = batch[0].to(self.device)
            targets: Tensor = batch[1].to(self.device, dtype=torch.long)
            target_lengths: Tensor = batch[2].to(self.device, dtype=torch.long)

            self.optimiser.zero_grad()
            log_probs = self.model(images)
            loss = self.criterion(log_probs, targets, target_lengths)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimiser.step()
            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.log_interval == 0:
                logger.info(
                    "  Epoch %d | batch %d | loss %.4f",
                    epoch,
                    batch_idx,
                    loss.item(),
                )

        return total_loss / max(num_batches, 1)

    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                images: Tensor = batch[0].to(self.device)
                targets: Tensor = batch[1].to(self.device, dtype=torch.long)
                target_lengths: Tensor = batch[2].to(self.device, dtype=torch.long)
                log_probs = self.model(images)
                loss = self.criterion(log_probs, targets, target_lengths)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_summary(self, summary: Dict[str, Any]) -> None:
        summary_path = self.checkpoint_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Training summary → %s", summary_path)

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
