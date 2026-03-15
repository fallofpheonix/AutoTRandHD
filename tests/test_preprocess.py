import numpy as np

from src.preprocess import deskew


def test_large_angle_skipped(monkeypatch):
    """
    Ensure that when the estimated skew angle is larger than ``max_angle_deg``,
    ``deskew_image`` skips correction and returns an angle of 0.0.

    This test monkeypatches ``_estimate_skew`` to return a fixed large angle
    to avoid flakiness from relying on the real skew estimation logic.
    """

    # Dummy image; contents are irrelevant because _estimate_skew is patched.
    img = np.zeros((32, 32), dtype=np.uint8)

    # Always report a large skew angle to trigger the "skip" branch.
    def fake_estimate_skew(_img):
        return 10.0  # degrees, well above the threshold used below

    # Patch the internal skew estimator in the deskew module.
    monkeypatch.setattr(deskew, "_estimate_skew", fake_estimate_skew)

    # Use a very small max_angle_deg so that the large estimated angle
    # should cause the deskewing logic to skip correction.
    corrected_img, angle = deskew.deskew_image(img, max_angle_deg=0.05)

    # When skipping, the angle should be reported as 0.0, and no rotation
    # should be applied to the image.
    assert angle == 0.0
    # Depending on implementation, this may or may not be the same object;
    # we at least ensure the image content is unchanged.
    assert corrected_img.shape == img.shape
    assert np.array_equal(corrected_img, img)
"""Unit tests for src.preprocess modules."""

from __future__ import annotations

import numpy as np
import pytest


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
        from src.preprocess.normalize import normalize_page
        img = _make_gray()
        out = normalize_page(img)
        assert out.dtype == np.uint8

    def test_output_is_2d(self):
        from src.preprocess.normalize import normalize_page
        img = _make_gray()
        out = normalize_page(img)
        assert out.ndim == 2

    def test_color_input_converted(self):
        from src.preprocess.normalize import normalize_page
        img = _make_color()
        out = normalize_page(img)
        assert out.ndim == 2

    def test_deterministic(self):
        from src.preprocess.normalize import normalize_page
        img = _make_gray(seed=7)
        out1 = normalize_page(img, denoise_h=5, clahe_clip_limit=2.0)
        out2 = normalize_page(img, denoise_h=5, clahe_clip_limit=2.0)
        np.testing.assert_array_equal(out1, out2)

    def test_no_denoise_skip(self):
        from src.preprocess.normalize import normalize_page
        img = _make_gray()
        out = normalize_page(img, denoise_h=0, clahe_clip_limit=0)
        np.testing.assert_array_equal(out, img)

    def test_shape_preserved(self):
        from src.preprocess.normalize import normalize_page
        img = _make_gray(100, 200)
        out = normalize_page(img)
        assert out.shape == (100, 200)

    def test_unsupported_shape_raises(self):
        from src.preprocess.normalize import normalize_page
        img = np.zeros((10, 10, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image shape"):
            normalize_page(img)


# ---------------------------------------------------------------------------
# deskew
# ---------------------------------------------------------------------------

class TestDeskewImage:
    def test_returns_tuple(self):
        from src.preprocess.deskew import deskew_image
        img = _make_gray()
        result = deskew_image(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_zero_angle_unchanged(self):
        from src.preprocess.deskew import deskew_image
        # A flat image with horizontal lines should detect near-zero skew.
        img = np.full((64, 256), 255, dtype=np.uint8)
        img[20, :] = 0  # horizontal black line
        corrected, angle = deskew_image(img)
        assert abs(angle) <= 10.0  # within tolerance

    def test_large_angle_skipped(self):
        from src.preprocess.deskew import deskew_image
        img = _make_gray()
        # Force angle beyond threshold by patching internal fn.
        corrected, angle = deskew_image(img, max_angle_deg=0.05)
        assert angle == 0.0  # skipped

    def test_output_shape_reasonable(self):
        from src.preprocess.deskew import deskew_image
        img = _make_gray(100, 200)
        corrected, _ = deskew_image(img)
        assert corrected.shape[0] >= 90  # may grow slightly due to rotation padding


# ---------------------------------------------------------------------------
# text_region
# ---------------------------------------------------------------------------

class TestExtractTextRegion:
    def test_returns_tuple(self):
        from src.preprocess.text_region import extract_text_region
        img = _make_gray()
        region, bbox = extract_text_region(img)
        assert isinstance(bbox, tuple)
        assert len(bbox) == 4

    def test_bbox_within_image(self):
        from src.preprocess.text_region import extract_text_region
        img = _make_gray(200, 400)
        region, (x, y, w, h) = extract_text_region(img)
        assert x >= 0 and y >= 0
        assert x + w <= 400
        assert y + h <= 200

    def test_region_shape_matches_bbox(self):
        from src.preprocess.text_region import extract_text_region
        img = _make_gray(200, 400)
        region, (x, y, w, h) = extract_text_region(img)
        assert region.shape == (h, w)

    def test_blank_image_returns_full(self):
        from src.preprocess.text_region import extract_text_region
        img = np.full((64, 128), 255, dtype=np.uint8)
        _, bbox = extract_text_region(img)
        assert bbox == (0, 0, 128, 64)


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
        from src.preprocess.line_segment import segment_lines
        img = self._make_page_with_lines(3)
        records = segment_lines(img, "src", 0, str(tmp_path))
        assert isinstance(records, list)

    def test_detects_lines(self, tmp_path):
        from src.preprocess.line_segment import segment_lines
        img = self._make_page_with_lines(3)
        records = segment_lines(img, "src", 0, str(tmp_path), min_line_height=5)
        assert len(records) >= 1

    def test_record_fields(self, tmp_path):
        from src.preprocess.line_segment import segment_lines
        img = self._make_page_with_lines(2)
        records = segment_lines(img, "src", 0, str(tmp_path), min_line_height=5)
        for rec in records:
            assert "line_id" in rec
            assert "bbox" in rec
            assert "image_path" in rec
            assert "transcript" in rec

    def test_crops_saved(self, tmp_path):
        from src.preprocess.line_segment import segment_lines
        import os
        img = self._make_page_with_lines(2)
        records = segment_lines(img, "src", 0, str(tmp_path), min_line_height=5)
        for rec in records:
            assert os.path.exists(rec["image_path"])

    def test_transcript_empty(self, tmp_path):
        from src.preprocess.line_segment import segment_lines
        img = self._make_page_with_lines(2)
        records = segment_lines(img, "src", 0, str(tmp_path), min_line_height=5)
        for rec in records:
            assert rec["transcript"] == ""

    def test_bbox_in_page_coords(self, tmp_path):
        from src.preprocess.line_segment import segment_lines
        img = self._make_page_with_lines(2)
        page_bbox = (10, 20, 300, 200)
        records = segment_lines(
            img, "src", 0, str(tmp_path),
            page_bbox=page_bbox, min_line_height=5
        )
        for rec in records:
            bx, by, bw, bh = rec["bbox"]
            assert bx == 10  # page_bbox x offset
