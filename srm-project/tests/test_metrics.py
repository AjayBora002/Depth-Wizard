"""
test_metrics.py
────────────────
Sanity-check tests for PSNR, SSIM, and SAM metric implementations.

These tests use synthetic arrays — no external data or GPU required.
Key invariants tested:
  - Identical inputs yield perfect scores (PSNR=inf, SSIM=1.0, SAM=0.0)
  - Noisy inputs produce degraded but valid scores
  - All outputs are within physically meaningful ranges
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import psnr, ssim, sam, sre, ergas, uiq, evaluate_pair


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_pair():
    """Identical random image pair — all metrics should be perfect."""
    rng = np.random.default_rng(0)
    img = rng.random((4, 64, 64), dtype=np.float32)
    return img, img.copy()


@pytest.fixture
def noisy_pair():
    """Slightly noisy pair — metrics should be degraded but not catastrophically."""
    rng = np.random.default_rng(1)
    gt = rng.random((4, 64, 64), dtype=np.float32)
    pred = (gt + rng.normal(0, 0.05, gt.shape).astype(np.float32)).clip(0, 1)
    return pred, gt


@pytest.fixture
def single_band_pair():
    """Single-band (grayscale) pair for edge-case testing."""
    rng = np.random.default_rng(2)
    gt = rng.random((1, 64, 64), dtype=np.float32)
    pred = (gt + 0.01).clip(0, 1)
    return pred, gt


@pytest.fixture
def multiband_constant_pair():
    """Constant spectral vectors — SAM should be 0 everywhere."""
    # pred and gt have the same spectral shape, just different magnitudes
    gt = np.ones((4, 32, 32), dtype=np.float32) * 0.5
    pred = np.ones((4, 32, 32), dtype=np.float32) * 0.7   # scaled, same angle
    return pred, gt


# ──────────────────────────────────────────────────────────────────────────────
# PSNR tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPSNR:
    def test_identical_images_infinite_psnr(self, perfect_pair):
        pred, gt = perfect_pair
        score = psnr(pred, gt)
        assert score == float("inf") or score > 100.0, \
            f"PSNR of identical images should be inf or >100dB, got {score}"

    def test_noisy_psnr_in_valid_range(self, noisy_pair):
        pred, gt = noisy_pair
        score = psnr(pred, gt)
        assert 0.0 < score < 60.0, f"PSNR out of expected range: {score}"

    def test_psnr_decreases_with_more_noise(self):
        rng = np.random.default_rng(42)
        gt = rng.random((3, 64, 64), dtype=np.float32)
        pred_low_noise = (gt + rng.normal(0, 0.02, gt.shape).astype(np.float32)).clip(0, 1)
        pred_high_noise = (gt + rng.normal(0, 0.2, gt.shape).astype(np.float32)).clip(0, 1)
        assert psnr(pred_low_noise, gt) > psnr(pred_high_noise, gt)

    def test_psnr_single_band(self, single_band_pair):
        pred, gt = single_band_pair
        score = psnr(pred, gt)
        assert math.isfinite(score)
        assert score > 0

    def test_shape_mismatch_raises(self):
        a = np.zeros((3, 64, 64), dtype=np.float32)
        b = np.zeros((3, 128, 128), dtype=np.float32)
        with pytest.raises(ValueError):
            psnr(a, b)


# ──────────────────────────────────────────────────────────────────────────────
# SSIM tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSSIM:
    def test_identical_images_ssim_one(self, perfect_pair):
        pred, gt = perfect_pair
        score = ssim(pred, gt)
        assert abs(score - 1.0) < 1e-4, f"SSIM of identical images should be 1.0, got {score}"

    def test_noisy_ssim_below_one(self, noisy_pair):
        pred, gt = noisy_pair
        score = ssim(pred, gt)
        assert score < 1.0
        assert score > 0.5, f"SSIM of mildly noisy pair should be > 0.5, got {score}"

    def test_ssim_in_valid_range(self, noisy_pair):
        pred, gt = noisy_pair
        score = ssim(pred, gt)
        assert -1.0 <= score <= 1.0

    def test_ssim_single_band(self, single_band_pair):
        pred, gt = single_band_pair
        score = ssim(pred, gt)
        assert math.isfinite(score)
        assert score < 1.0    # pred != gt

    def test_ssim_multiband_averaged(self):
        """SSIM should average across all bands."""
        rng = np.random.default_rng(5)
        gt = rng.random((4, 64, 64), dtype=np.float32)
        pred = (gt + 0.05).clip(0, 1)
        score = ssim(pred, gt)
        # All bands have same noise level → SSIM should be consistent
        assert math.isfinite(score)


# ──────────────────────────────────────────────────────────────────────────────
# SAM tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSAM:
    def test_identical_images_sam_zero(self, perfect_pair):
        pred, gt = perfect_pair
        score = sam(pred, gt)
        assert abs(score) < 1e-5, f"SAM of identical images should be 0°, got {score}"

    def test_same_spectral_shape_different_magnitude_zero_sam(self, multiband_constant_pair):
        """Scaled versions of the same spectral vector have 0° angle."""
        pred, gt = multiband_constant_pair
        score = sam(pred, gt)
        assert abs(score) < 1e-3, f"SAM of same-shape spectral vectors should be ~0°, got {score}"

    def test_noisy_sam_in_valid_range(self, noisy_pair):
        pred, gt = noisy_pair
        score = sam(pred, gt)
        assert 0.0 <= score <= 90.0, f"SAM out of [0°, 90°]: {score}"

    def test_sam_increases_with_more_spectral_distortion(self):
        rng = np.random.default_rng(7)
        gt = rng.random((4, 32, 32), dtype=np.float32) + 0.1
        pred_mild = (gt + rng.normal(0, 0.01, gt.shape).astype(np.float32)).clip(0, 1)
        pred_severe = (gt + rng.normal(0, 0.3, gt.shape).astype(np.float32)).clip(0, 1)
        assert sam(pred_mild, gt) < sam(pred_severe, gt)

    def test_sam_single_band_zero(self):
        """Single-band SAM is trivially 0 (no spectral angle to measure)."""
        rng = np.random.default_rng(8)
        gt = rng.random((1, 32, 32), dtype=np.float32) + 0.1
        pred = (gt * 2.0).clip(0, 1)
        score = sam(pred, gt)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_sam_max_angle(self):
        """Orthogonal spectral vectors → maximum SAM ≈ 90°."""
        gt = np.zeros((4, 16, 16), dtype=np.float32)
        pred = np.zeros((4, 16, 16), dtype=np.float32)
        gt[0] = 1.0      # only band 0 active
        pred[1] = 1.0    # only band 1 active
        score = sam(pred, gt)
        assert abs(score - 90.0) < 1.0, f"Orthogonal vectors should give ~90°, got {score}"


class TestSRE:
    def test_identical_images_infinite_sre(self, perfect_pair):
        pred, gt = perfect_pair
        assert sre(pred, gt) == float("inf")

    def test_noisy_sre_in_valid_range(self, noisy_pair):
        pred, gt = noisy_pair
        val = sre(pred, gt)
        assert 10.0 <= val <= 50.0


class TestERGAS:
    def test_identical_images_zero_ergas(self, perfect_pair):
        pred, gt = perfect_pair
        assert abs(ergas(pred, gt)) < 1e-5

    def test_noisy_ergas_positive(self, noisy_pair):
        pred, gt = noisy_pair
        val = ergas(pred, gt, scale=4.0)
        assert val > 0.0


class TestUIQ:
    def test_identical_images_uiq_one(self, perfect_pair):
        pred, gt = perfect_pair
        assert abs(uiq(pred, gt) - 1.0) < 1e-4

    def test_noisy_uiq_in_range(self, noisy_pair):
        pred, gt = noisy_pair
        val = uiq(pred, gt)
        assert 0.0 <= val <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Integrated evaluate_pair
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluatePair:
    def test_returns_all_keys(self, perfect_pair):
        pred, gt = perfect_pair
        result = evaluate_pair(pred, gt)
        assert "psnr" in result
        assert "ssim" in result
        assert "sam_deg" in result
        assert "sre" in result
        assert "ergas" in result
        assert "uiq" in result

    def test_perfect_pair_scores(self, perfect_pair):
        pred, gt = perfect_pair
        result = evaluate_pair(pred, gt)
        assert result["psnr"] > 100.0 or result["psnr"] == float("inf")
        assert abs(result["ssim"] - 1.0) < 1e-4
        assert abs(result["sam_deg"]) < 1e-4
        assert result["sre"] == float("inf")
        assert abs(result["ergas"]) < 1e-4
        assert abs(result["uiq"] - 1.0) < 1e-4

    def test_noisy_pair_scores_degrade(self, noisy_pair):
        pred, gt = noisy_pair
        result = evaluate_pair(pred, gt)
        assert result["psnr"] < 60.0
        assert result["ssim"] < 1.0
        assert result["sam_deg"] > 0.0
        assert result["ergas"] > 0.0
        assert result["uiq"] < 1.0

