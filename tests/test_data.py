"""Unit tests for src.data.manifest and src.data.splits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.manifest import (
    REQUIRED_COLUMNS,
    build_manifest,
    build_line_metadata,
    load_manifest,
    merge_manifests,
    save_manifest,
)
from src.data.splits import generate_splits


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_basic_columns(self):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(5)]
        df = build_manifest("src_001", paths, dpi=300)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_row_count(self):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(7)]
        df = build_manifest("src_001", paths, dpi=300)
        assert len(df) == 7

    def test_source_id_populated(self):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(3)]
        df = build_manifest("my_source", paths, dpi=200)
        assert (df["source_id"] == "my_source").all()

    def test_page_ids_sequential(self):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(4)]
        df = build_manifest("src", paths, dpi=300)
        assert df["page_id"].tolist() == [0, 1, 2, 3]

    def test_split_empty_by_default(self):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(3)]
        df = build_manifest("src", paths, dpi=300)
        assert (df["split"] == "").all()

    def test_transcript_column_added(self):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(2)]
        transcripts = [Path(f"/tmp/tr_{i}.txt") for i in range(2)]
        df = build_manifest("src", paths, dpi=300, transcription_paths=transcripts)
        assert "transcript_path" in df.columns

    def test_transcript_length_mismatch_raises(self):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(3)]
        transcripts = [Path(f"/tmp/tr_{i}.txt") for i in range(2)]
        with pytest.raises(ValueError, match="must match"):
            build_manifest("src", paths, dpi=300, transcription_paths=transcripts)


# ---------------------------------------------------------------------------
# save_manifest / load_manifest
# ---------------------------------------------------------------------------

class TestManifestIO:
    def test_roundtrip(self, tmp_path):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(4)]
        df = build_manifest("src", paths, dpi=300)
        csv_path = tmp_path / "manifest.csv"
        save_manifest(df, csv_path)
        loaded = load_manifest(csv_path)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded.reset_index(drop=True))

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.csv")

    def test_load_missing_columns_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"col_a": [1]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_manifest(csv_path)


# ---------------------------------------------------------------------------
# merge_manifests
# ---------------------------------------------------------------------------

class TestMergeManifests:
    def test_merge_two(self):
        paths_a = [Path(f"/tmp/a_{i}.png") for i in range(3)]
        paths_b = [Path(f"/tmp/b_{i}.png") for i in range(2)]
        df_a = build_manifest("src_a", paths_a, dpi=300)
        df_b = build_manifest("src_b", paths_b, dpi=300)
        merged = merge_manifests([df_a, df_b])
        assert len(merged) == 5

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            merge_manifests([])


# ---------------------------------------------------------------------------
# generate_splits
# ---------------------------------------------------------------------------

class TestGenerateSplits:
    def _make_manifest(self, n: int, source_id: str = "src") -> pd.DataFrame:
        paths = [Path(f"/tmp/img_{i}.png") for i in range(n)]
        return build_manifest(source_id, paths, dpi=300)

    def test_all_rows_assigned(self):
        df = self._make_manifest(20)
        result = generate_splits(df, strategy="page", seed=42)
        assert result["split"].isin(["train", "val", "test"]).all()

    def test_reproducible_with_seed(self):
        df = self._make_manifest(30)
        r1 = generate_splits(df, seed=7)
        r2 = generate_splits(df, seed=7)
        assert r1["split"].tolist() == r2["split"].tolist()

    def test_different_seeds_differ(self):
        df = self._make_manifest(50)
        r1 = generate_splits(df, seed=1)
        r2 = generate_splits(df, seed=99)
        assert r1["split"].tolist() != r2["split"].tolist()

    def test_all_three_splits_present(self):
        df = self._make_manifest(30)
        result = generate_splits(df)
        assert set(result["split"].unique()) >= {"train", "val", "test"}

    def test_does_not_mutate_original(self):
        df = self._make_manifest(20)
        original_splits = df["split"].tolist()
        generate_splits(df)
        assert df["split"].tolist() == original_splits

    def test_source_strategy_keeps_sources_together(self):
        paths = [Path(f"/tmp/img_{i}.png") for i in range(6)]
        df_a = build_manifest("src_a", paths[:3], dpi=300)
        df_b = build_manifest("src_b", paths[3:], dpi=300)
        merged = merge_manifests([df_a, df_b])
        result = generate_splits(merged, strategy="source", seed=42)
        # All pages of a source must be in the same split.
        for src in result["source_id"].unique():
            src_splits = result.loc[result["source_id"] == src, "split"].unique()
            assert len(src_splits) == 1

    def test_bad_ratios_raise(self):
        df = self._make_manifest(10)
        with pytest.raises(ValueError, match="must equal 1.0"):
            generate_splits(df, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_bad_strategy_raises(self):
        df = self._make_manifest(10)
        with pytest.raises(ValueError, match="Unknown split strategy"):
            generate_splits(df, strategy="random")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_line_metadata
# ---------------------------------------------------------------------------

class TestBuildLineMetadata:
    def test_basic(self):
        records = [
            {"line_id": 0, "bbox": [0, 10, 500, 30], "image_path": "/tmp/l0.png", "transcript": "foo"},
            {"line_id": 1, "bbox": [0, 45, 500, 30], "image_path": "/tmp/l1.png", "transcript": ""},
        ]
        df = build_line_metadata("src", 0, records)
        assert "source_id" in df.columns
        assert "page_id" in df.columns
        assert len(df) == 2
