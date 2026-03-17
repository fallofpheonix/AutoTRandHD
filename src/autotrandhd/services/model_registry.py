from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

import torch.nn as nn

from autotrandhd.config import AppSettings
from autotrandhd.core.recognition.loader import load_model

logger = logging.getLogger(__name__)


class ModelDescription(TypedDict, total=False):
    loaded: bool
    checkpoint: str
    device: str
    parameters: int


@dataclass
class ModelRegistry:
    settings: AppSettings
    _model: nn.Module | None = None

    def load_if_present(self) -> None:
        if self._model is not None:
            return
        if not self.settings.model_path.exists():
            logger.warning("Model checkpoint is missing at %s", self.settings.model_path)
            return
        self._model = load_model(self.settings.model_path, device=self.settings.device)

    def require_model(self):
        self.load_if_present()
        if self._model is None:
            raise FileNotFoundError(f"Model checkpoint not found: {self.settings.model_path}")
        return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> ModelDescription:
        self.load_if_present()
        if self._model is None:
            return {"loaded": False, "checkpoint": str(self.settings.model_path)}
        return {
            "loaded": True,
            "checkpoint": str(self.settings.model_path),
            "device": self.settings.device,
            "parameters": sum(parameter.numel() for parameter in self._model.parameters()),
        }
