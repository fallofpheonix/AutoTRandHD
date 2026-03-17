"""Smoke tests for src.models (training and inference).

These tests do not require a GPU or real data.  They verify that the model
can be instantiated, run a forward pass, compute a loss, and save/load a
checkpoint.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# CRNN
# ---------------------------------------------------------------------------

class TestCRNN:
    def _make_model(self, num_classes: int = 20) -> "torch.nn.Module":
        from autotrandhd.core.recognition.crnn import CRNN

        # img_height=32 needs 5 halving operations → 5 stages → 6 channel values.
        return CRNN(
            num_classes=num_classes,
            img_height=32,
            cnn_channels=[1, 8, 16, 16, 16, 16],
            rnn_hidden=32,
            rnn_layers=1,
        )

    def test_instantiation(self):
        model = self._make_model()
        assert model is not None

    def test_forward_output_shape(self):
        model = self._make_model(num_classes=20)
        model.eval()
        # B=2, C=1, H=32, W=128
        x = torch.zeros(2, 1, 32, 128)
        with torch.no_grad():
            logits = model(x)
        # logits: (T, B, num_classes)
        assert logits.ndim == 3
        assert logits.shape[1] == 2
        assert logits.shape[2] == 20

    def test_log_softmax_output(self):
        model = self._make_model(num_classes=20)
        model.eval()
        x = torch.zeros(1, 1, 32, 64)
        with torch.no_grad():
            logits = model(x)
        # log-softmax values should be <= 0
        assert (logits <= 0).all()

    def test_invalid_cnn_channels_raises(self):
        from autotrandhd.core.recognition.crnn import CRNN

        with pytest.raises(ValueError, match=r"cnn_channels\[0\]"):
            CRNN(num_classes=10, cnn_channels=[3, 64])


# ---------------------------------------------------------------------------
# CTCLossWrapper
# ---------------------------------------------------------------------------

class TestCTCLossWrapper:
    def test_loss_is_scalar(self):
        from autotrandhd.core.recognition.ctc_loss import CTCLossWrapper

        criterion = CTCLossWrapper(blank=0)
        T, B, C = 10, 2, 15
        log_probs = torch.randn(T, B, C).log_softmax(dim=-1)
        targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
        target_lengths = torch.tensor([3, 3], dtype=torch.long)
        loss = criterion(log_probs, targets, target_lengths)
        assert loss.shape == ()

    def test_loss_non_negative(self):
        from autotrandhd.core.recognition.ctc_loss import CTCLossWrapper

        criterion = CTCLossWrapper(blank=0)
        T, B, C = 10, 2, 15
        log_probs = torch.randn(T, B, C).log_softmax(dim=-1)
        targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
        target_lengths = torch.tensor([3, 3], dtype=torch.long)
        loss = criterion(log_probs, targets, target_lengths)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Trainer (smoke test)
# ---------------------------------------------------------------------------

class TestTrainer:
    def _make_loader(self, batch_size: int = 2, img_w: int = 64):
        """Return a minimal DataLoader with random data."""
        from torch.utils.data import DataLoader, TensorDataset

        B = batch_size
        images = torch.zeros(B, 1, 32, img_w)
        # Each sample has 3 target tokens (value 1), stored as flat padded rows.
        targets = torch.ones(B, 3, dtype=torch.long)
        target_lengths = torch.full((B,), 3, dtype=torch.long)
        dataset = TensorDataset(images, targets, target_lengths)
        return DataLoader(dataset, batch_size=B)

    def test_fit_one_epoch(self, tmp_path):
        from autotrandhd.core.recognition.crnn import CRNN
        from autotrandhd.core.recognition.ctc_loss import CTCLossWrapper
        from autotrandhd.core.recognition.trainer import Trainer

        model = CRNN(num_classes=15, img_height=32, cnn_channels=[1, 8, 16, 16, 16, 16], rnn_hidden=32, rnn_layers=1)
        criterion = CTCLossWrapper(blank=0)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimiser=optimiser,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu",
            seed=42,
        )
        loader = self._make_loader()
        summary = trainer.fit(loader, num_epochs=1)
        assert "train_losses" in summary
        assert len(summary["train_losses"]) == 1

    def test_checkpoint_saved(self, tmp_path):
        from autotrandhd.core.recognition.crnn import CRNN
        from autotrandhd.core.recognition.ctc_loss import CTCLossWrapper
        from autotrandhd.core.recognition.trainer import Trainer

        model = CRNN(num_classes=15, img_height=32, cnn_channels=[1, 8, 16, 16, 16, 16], rnn_hidden=32, rnn_layers=1)
        criterion = CTCLossWrapper(blank=0)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt_dir = tmp_path / "ckpts"
        trainer = Trainer(model, criterion, optimiser, checkpoint_dir=str(ckpt_dir), device="cpu")
        loader = self._make_loader()
        trainer.fit(loader, num_epochs=1)
        ckpts = list(ckpt_dir.glob("*.pt"))
        assert len(ckpts) == 1

    def test_load_checkpoint(self, tmp_path):
        from autotrandhd.core.recognition.crnn import CRNN
        from autotrandhd.core.recognition.ctc_loss import CTCLossWrapper
        from autotrandhd.core.recognition.trainer import Trainer

        model = CRNN(num_classes=15, img_height=32, cnn_channels=[1, 8, 16, 16, 16, 16], rnn_hidden=32, rnn_layers=1)
        criterion = CTCLossWrapper(blank=0)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt_dir = tmp_path / "ckpts"
        trainer = Trainer(model, criterion, optimiser, checkpoint_dir=str(ckpt_dir), device="cpu")
        loader = self._make_loader()
        trainer.fit(loader, num_epochs=1)
        ckpt_path = list(ckpt_dir.glob("*.pt"))[0]
        epoch = trainer.load_checkpoint(ckpt_path)
        assert epoch == 1


# ---------------------------------------------------------------------------
# inference (smoke test with saved checkpoint)
# ---------------------------------------------------------------------------

class TestInference:
    def _save_dummy_checkpoint(self, tmp_path: Path, num_classes: int = 15) -> Path:
        from autotrandhd.core.recognition.crnn import CRNN

        model = CRNN(num_classes=num_classes, img_height=32, cnn_channels=[1, 8, 16, 16, 16, 16], rnn_hidden=32, rnn_layers=1)
        ckpt_path = tmp_path / "dummy.pt"
        torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "optimiser_state_dict": {}}, ckpt_path)
        return ckpt_path

    def _save_dummy_image(self, tmp_path: Path, h: int = 32, w: int = 64) -> Path:
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available")
        img_path = tmp_path / "line.png"
        img = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        return img_path

    def test_transcribe_returns_records(self, tmp_path):
        from autotrandhd.core.recognition.inference import transcribe

        ckpt = self._save_dummy_checkpoint(tmp_path)
        img = self._save_dummy_image(tmp_path)
        results = transcribe(
            image_paths=[str(img)],
            checkpoint_path=str(ckpt),
            num_classes=15,
            img_height=32,
            cnn_channels=[1, 8, 16, 16, 16, 16],
            rnn_hidden=32,
            rnn_layers=1,
        )
        assert len(results) == 1
        assert "image_path" in results[0]

    def test_transcribe_saves_logits(self, tmp_path):
        from autotrandhd.core.recognition.inference import transcribe

        ckpt = self._save_dummy_checkpoint(tmp_path)
        img = self._save_dummy_image(tmp_path)
        logits_dir = tmp_path / "logits"
        results = transcribe(
            image_paths=[str(img)],
            checkpoint_path=str(ckpt),
            num_classes=15,
            img_height=32,
            cnn_channels=[1, 8, 16, 16, 16, 16],
            rnn_hidden=32,
            rnn_layers=1,
            logits_dir=str(logits_dir),
        )
        assert results[0]["logits_path"] is not None
        assert Path(results[0]["logits_path"]).exists()
