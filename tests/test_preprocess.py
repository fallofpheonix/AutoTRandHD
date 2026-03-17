"""Unit tests for src.preprocess modules."""

from __future__ import annotations

import numpy as np
import pytest

from autotrandhd.core.preprocessing import deskew


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gray(h: int = 128, w: int = 256, seed: int = 0) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    return rng.integers(50, 200, size=(h, w), dtype=np.uint8)


def _make_color(h: int = 128, w: int = 256) -> "np.ndarray":
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalizePage:
    def test_returns_uint8(self):
        from autotrandhd.core.preprocessing.normalize import normalize_page
        img = _make_gray()
        out = normalize_page(img)
        assert out.dtype == np.uint8

    def test_output_is_2d(self):
        from autotrandhd.core.preprocessing.normalize import normalize_page
        img = _make_gray()
        out = normalize_page(img)
        assert out.ndim == 2

    def test_color_input_converted(self):
        from autotrandhd.core.preprocessing.normalize import normalize_page
        img = _make_color()
        out = normalize_page(img)
        assert out.ndim == 2

    def test_deterministic(self):
        from autotrandhd.core.preprocessing.normalize import normalize_page
        img = _make_gray(seed=7)
        out1 = normalize_page(img, denoise_h=5, clahe_clip_limit=2.0)
        out2 = normalize_page(img, denoise_h=5, clahe_clip_limit=2.0)
        np.testing.assert_array_equal(out1, out2)

    def test_no_denoise_skip(self):
        from autotrandhd.core.preprocessing.normalize import normalize_page
        img = _make_gray()
        out = normalize_page(img, denoise_h=0, clahe_clip_limit=0)
        np.testing.assert_array_equal(out, img)

    def test_shape_preserved(self):
        from autotrandhd.core.preprocessing.normalize import normalize_page
        img = _make_gray(100, 200)
        out = normalize_page(img)
        assert out.shape == (100, 200)

    def test_unsupported_shape_raises(self):
        from autotrandhd.core.preprocessing.normalize import normalize_page
        img = np.zeros((10, 10, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image shape"):
            normalize_page(img)


# ---------------------------------------------------------------------------
# deskew
# ---------------------------------------------------------------------------

class TestDeskewImage:
    def test_returns_tuple(self):
        from autotrandhd.core.preprocessing.deskew import deskew_image
        img = _make_gray()
        result = deskew_image(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_zero_angle_unchanged(self):
        from autotrandhd.core.preprocessing.deskew import deskew_image
        # A flat image with horizontal lines should detect near-zero skew.
        img = np.full((64, 256), 255, dtype=np.uint8)
        img[20, :] = 0  # horizontal black line
        corrected, angle = deskew_image(img)
        assert abs(angle) <= 10.0  # within tolerance

    def test_large_angle_skipped(self, monkeypatch):
        """Skew correction is skipped when the estimated angle exceeds max_angle_deg."""
        img = np.zeros((32, 32), dtype=np.uint8)

        def fake_estimate_skew(_img):
            return 10.0  # degrees, well above threshold

        monkeypatch.setattr(deskew, "_estimate_skew", fake_estimate_skew)
        corrected, angle = deskew.deskew_image(img, max_angle_deg=0.05)
        assert angle == 0.0
        assert corrected.shape == img.shape
        assert np.array_equal(corrected, img)

    def test_output_shape_reasonable(self):
        from autotrandhd.core.preprocessing.deskew import deskew_image
        img = _make_gray(100, 200)
        corrected, _ = deskew_image(img)
        assert corrected.shape[0] >= 90  # may grow slightly due to rotation padding


# ---------------------------------------------------------------------------
# text_region
# ---------------------------------------------------------------------------

class TestExtractTextRegion:
    def test_returns_tuple(self):
        from autotrandhd.core.preprocessing.text_region import extract_text_region
        img = _make_gray()
        region, bbox = extract_text_region(img)
        assert isinstance(bbox, tuple)
        assert len(bbox) == 4

    def test_bbox_within_image(self):
        from autotrandhd.core.preprocessing.text_region import extract_text_region
        img = _make_gray(200, 400)
        region, (x, y, w, h) = extract_text_region(img)
        assert x >= 0 and y >= 0
        assert x + w <= 400
        assert y + h <= 200

    def test_region_shape_matches_bbox(self):
        from autotrandhd.core.preprocessing.text_region import extract_text_region
        img = _make_gray(200, 400)
        region, (x, y, w, h) = extract_text_region(img)
        assert region.shape == (h, w)

    def test_blank_image_returns_full(self):
        from autotrandhd.core.preprocessing.text_region import extract_text_region
        img = np.full((64, 128), 255, dtype=np.uint8)
        _, bbox = extract_text_region(img)
        assert bbox == (0, 0, 128, 64)

    def test_clustering_excludes_marginal_components(self):
        """Main-text cluster (large) should exclude small marginal note cluster."""
        from autotrandhd.core.preprocessing.text_region import extract_text_region

        # Page: 300 px tall, 400 px wide (all white).
        img = np.full((300, 400), 255, dtype=np.uint8)

        # Main-text block: dense rows of black pixels in the vertical centre.
        # Each stripe is 12 px tall (> default min_component_height of 10).
        for row_y in range(100, 200, 18):
            img[row_y : row_y + 12, 40:360] = 0  # wide text-like stripes

        # Marginal note: a single small cluster at the very top of the page,
        # separated from the main block by a large vertical gap.
        img[5:20, 5:80] = 0  # small isolated blob near top margin

        _, (bx, by, bw, bh) = extract_text_region(img, padding=0)

        # The returned region must contain the main-text block.
        assert by <= 100
        assert by + bh >= 196

        # The top margin of the returned bbox must NOT extend all the way
        # to row 5 (where the marginal note lives); the clustering step
        # should have discarded that small isolated cluster.
        assert by > 5


# ---------------------------------------------------------------------------
# line_segment
# ---------------------------------------------------------------------------

class TestSegmentLines:
    def _make_page_with_lines(self, n_lines: int = 3) -> "np.ndarray":
        """Create a synthetic page with *n_lines* horizontal black bands."""
        img = np.full((200, 300), 255, dtype=np.uint8)
        spacing = 200 // (n_lines + 1)
        for i in range(n_lines):
            y = spacing * (i + 1)
            img[y : y + 10, 10:290] = 0
        return img

    def test_returns_list(self, tmp_path):
        from autotrandhd.core.preprocessing.line_segment import segment_lines
        img = self._make_page_with_lines(3)
        records = segment_lines(img, "src", 0, str(tmp_path))
        assert isinstance(records, list)

    def test_detects_lines(self, tmp_path):
        from autotrandhd.core.preprocessing.line_segment import segment_lines
        img = self._make_page_with_lines(3)
        records = segment_lines(img, "src", 0, str(tmp_path), min_line_height=5)
        assert len(records) >= 1

    def test_record_fields(self, tmp_path):
        from autotrandhd.core.preprocessing.line_segment import segment_lines
        img = self._make_page_with_lines(2)
        records = segment_lines(img, "src", 0, str(tmp_path), min_line_height=5)
        for rec in records:
            assert "line_id" in rec
            assert "bbox" in rec
            assert "image_path" in rec
            assert "transcript" in rec

    def test_crops_saved(self, tmp_path):
        from autotrandhd.core.preprocessing.line_segment import segment_lines
        import os
        img = self._make_page_with_lines(2)
        records = segment_lines(img, "src", 0, str(tmp_path), min_line_height=5)
        for rec in records:
            assert os.path.exists(rec["image_path"])

    def test_transcript_empty(self, tmp_path):
        from autotrandhd.core.preprocessing.line_segment import segment_lines
        img = self._make_page_with_lines(2)
        records = segment_lines(img, "src", 0, str(tmp_path), min_line_height=5)
        for rec in records:
            assert rec["transcript"] == ""

    def test_bbox_in_page_coords(self, tmp_path):
        from autotrandhd.core.preprocessing.line_segment import segment_lines
        img = self._make_page_with_lines(2)
        page_bbox = (10, 20, 300, 200)
        records = segment_lines(
            img, "src", 0, str(tmp_path),
            page_bbox=page_bbox, min_line_height=5
        )
        for rec in records:
            bx, by, bw, bh = rec["bbox"]
            assert bx == 10  # page_bbox x offset
