from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch

from autotrandhd.config import AppSettings
from autotrandhd.core.decoding.beam_search import beam_decode
from autotrandhd.core.preprocessing.deskew import deskew_image
from autotrandhd.core.preprocessing.line_segment import segment_lines
from autotrandhd.core.preprocessing.normalize import normalize_page
from autotrandhd.core.preprocessing.text_region import extract_text_region
from autotrandhd.services.model_registry import ModelRegistry
from autotrandhd.utils.image_io import decode_image_bytes, read_grayscale_image


class InferenceResult(TypedDict):
    image: str
    text: str
    confidence: float
    latency_ms: int


class OCRPipelineService:
    def __init__(self, settings: AppSettings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self.settings.scratch_dir.mkdir(parents=True, exist_ok=True)

    def transcribe_path(self, image_path: str | Path, beam_width: int = 10) -> InferenceResult:
        image = read_grayscale_image(image_path)
        return self._transcribe_image(image=image, image_name=Path(image_path).name, beam_width=beam_width)

    def transcribe_bytes(self, payload: bytes, image_name: str, beam_width: int = 10) -> InferenceResult:
        image = decode_image_bytes(payload)
        return self._transcribe_image(image=image, image_name=image_name, beam_width=beam_width)

    def _transcribe_image(self, image: np.ndarray, image_name: str, beam_width: int) -> InferenceResult:
        started_at = time.time()
        model = self.registry.require_model()

        normalized = normalize_page(image)
        deskewed, _ = deskew_image(normalized)
        region, _ = extract_text_region(deskewed)

        scratch_dir = self.settings.scratch_dir / "line_crops"
        line_records = segment_lines(region, "runtime", 0, scratch_dir)
        if not line_records:
            return {
                "image": image_name,
                "text": "",
                "confidence": 0.0,
                "latency_ms": int((time.time() - started_at) * 1000),
            }

        texts: list[str] = []
        for record in line_records:
            line_path = cast(str, record["image_path"])
            line_image = read_grayscale_image(line_path)
            line_tensor = torch.from_numpy(line_image).float().unsqueeze(0).unsqueeze(0) / 255.0
            line_tensor = line_tensor.to(self.settings.device)
            with torch.no_grad():
                logits = model(line_tensor)
            hypotheses = beam_decode(logits.squeeze(1).cpu().numpy(), vocab=self.settings.vocab, beam_width=beam_width)
            if hypotheses:
                texts.append(cast(str, hypotheses[0]["text"]))

        confidence = 0.95 if texts else 0.0
        return {
            "image": image_name,
            "text": " ".join(texts).strip(),
            "confidence": confidence,
            "latency_ms": int((time.time() - started_at) * 1000),
        }
