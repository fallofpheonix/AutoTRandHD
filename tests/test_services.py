from __future__ import annotations

from pathlib import Path

from autotrandhd.config.settings import AppSettings
from autotrandhd.services.model_registry import ModelRegistry
from autotrandhd.utils.image_io import collect_image_paths


def test_model_registry_describes_missing_checkpoint(tmp_path):
    settings = AppSettings(model_path=tmp_path / "missing.pt")
    registry = ModelRegistry(settings)

    description = registry.describe()

    assert description["loaded"] is False
    assert description["checkpoint"].endswith("missing.pt")


def test_collect_image_paths_skips_non_images(tmp_path):
    (tmp_path / "page_01.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "page_02.jpg").write_bytes(b"x")

    image_paths = collect_image_paths(tmp_path)

    assert [path.name for path in image_paths] == ["page_01.png", "page_02.jpg"]
