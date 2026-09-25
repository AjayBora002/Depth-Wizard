"""
test_pipeline.py
─────────────────
End-to-end smoke test that runs the full pipeline:
  synthetic tile → pair generation → model inference (CPU) → outputs exist

This test does NOT require GPU or real Sentinel-2 data.
It uses a small (64×64) synthetic tile to keep the test fast on CPU.

The model inference step uses a tiny mock SR function to avoid downloading
weights in CI/test environments — the actual Real-ESRGAN is tested manually.
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
import torch.nn.functional as F
from affine import Affine
from rasterio.crs import CRS

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import validate_tile, extract_rgb_preview
from src.pair_generation import degrade, generate_pairs_from_tile, SyntheticPairDataset
from src.metrics import evaluate_pair
from src.uncertainty import tta_ensemble, tta_ensemble_batched, uncertainty_to_heatmap


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_tile(tmp_path) -> Path:
    """64×64 4-band synthetic GeoTIFF."""
    tif_path = tmp_path / "small_tile.tif"
    rng = np.random.default_rng(0)
    data = rng.integers(1000, 10000, (4, 64, 64), dtype=np.uint16)
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
    with rasterio.open(
        tif_path, "w", driver="GTiff",
        height=64, width=64, count=4, dtype=np.uint16,
        crs=CRS.from_epsg(32643), transform=transform,
    ) as dst:
        dst.write(data)
    return tif_path


# ──────────────────────────────────────────────────────────────────────────────
# Pair generation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPairGeneration:
    def test_degrade_output_shape(self):
        """degrade() returns correct shape for 4× downscale."""
        rng = np.random.default_rng(0)
        hr = rng.random((3, 64, 64), dtype=np.float32)
        lr = degrade(hr, scale=4)
        assert lr.shape == (3, 16, 16), f"Expected (3,16,16), got {lr.shape}"

    def test_degrade_output_range(self):
        """LR output stays in [0, 1]."""
        rng = np.random.default_rng(1)
        hr = rng.random((3, 64, 64), dtype=np.float32)
        lr = degrade(hr, scale=4, apply_jpeg=False)
        assert lr.min() >= 0.0, "LR below 0"
        assert lr.max() <= 1.0, "LR above 1"

    def test_degrade_different_scales(self):
        rng = np.random.default_rng(2)
        for scale in [2, 4]:
            hr = rng.random((3, 64, 64), dtype=np.float32)
            lr = degrade(hr, scale=scale, apply_jpeg=False)
            assert lr.shape == (3, 64 // scale, 64 // scale)

    def test_generate_pairs_from_tile(self, small_tile, tmp_path):
        """generate_pairs_from_tile() creates LR/HR .npy files."""
        out_dir = tmp_path / "pairs"
        count = generate_pairs_from_tile(
            small_tile, out_dir, patch_size=32, overlap=8, scale=4, apply_jpeg=False,
        )
        assert count > 0, "No pairs generated"
        lr_files = list((out_dir / "lr").glob("*_lr.npy"))
        hr_files = list((out_dir / "hr").glob("*_hr.npy"))
        assert len(lr_files) == count
        assert len(hr_files) == count

    def test_pair_shapes_consistent(self, small_tile, tmp_path):
        """LR shape is exactly (scale,scale) smaller than HR shape."""
        out_dir = tmp_path / "pairs2"
        generate_pairs_from_tile(
            small_tile, out_dir, patch_size=32, overlap=8, scale=4, apply_jpeg=False,
        )
        lr_files = sorted((out_dir / "lr").glob("*_lr.npy"))
        hr_files = sorted((out_dir / "hr").glob("*_hr.npy"))
        for lrf, hrf in zip(lr_files, hr_files):
            lr = np.load(lrf)
            hr = np.load(hrf)
            assert hr.shape[0] == lr.shape[0], "Channel count mismatch"
            assert hr.shape[1] == lr.shape[1] * 4, "Height scale mismatch"
            assert hr.shape[2] == lr.shape[2] * 4, "Width scale mismatch"

    def test_dataset_splits(self, small_tile, tmp_path):
        """SyntheticPairDataset correctly splits train/val."""
        out_dir = tmp_path / "pairs3"
        generate_pairs_from_tile(
            small_tile, out_dir, patch_size=32, overlap=0, scale=4, apply_jpeg=False,
        )
        train_ds = SyntheticPairDataset(out_dir, split="train")
        val_ds = SyntheticPairDataset(out_dir, split="val")
        total = len(train_ds) + len(val_ds)
        assert total > 0
        # Validate __getitem__
        lr, hr = train_ds[0]
        assert lr.dtype == np.float32 or hasattr(lr, "float")

    def test_generate_all_pairs_multilocation(self, tmp_path):
        """Test pair generation discovering multiple locations (.tif and .tiff)."""
        from src.pair_generation import generate_all_pairs
        from src.train import CropDataset, _collate_fn
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Create tile 1: 3-band Delhi (tif)
        t1 = raw_dir / "Delhi_Sample.tif"
        rng = np.random.default_rng(1)
        data1 = rng.integers(1000, 10000, (3, 64, 64), dtype=np.uint16)
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        with rasterio.open(t1, "w", driver="GTiff", height=64, width=64, count=3, dtype=np.uint16, transform=transform) as dst:
            dst.write(data1)

        # Create tile 2: 1-band Mumbai (tiff)
        t2 = raw_dir / "Mumbai_B02.tiff"
        data2 = rng.integers(1000, 10000, (1, 64, 64), dtype=np.uint16)
        with rasterio.open(t2, "w", driver="GTiff", height=64, width=64, count=1, dtype=np.uint16, transform=transform) as dst:
            dst.write(data2)

        out_dir = tmp_path / "pairs_multi"
        total = generate_all_pairs(
            raw_dir=raw_dir,
            out_dir=out_dir,
            patch_size=32,
            overlap=0,
            scale=4,
            rgb_only=True,
            patches_per_tile=10,
        )
        assert total > 0
        ds = SyntheticPairDataset(out_dir, split="train")
        crop_ds = CropDataset(ds, crop_size=8, scale=4, rgb_only=True)
        loader = torch.utils.data.DataLoader(crop_ds, batch_size=2, collate_fn=_collate_fn)
        batch_lr, batch_hr = next(iter(loader))
        assert batch_lr.shape == (2, 3, 8, 8), f"Expected (2, 3, 8, 8), got {batch_lr.shape}"
        assert batch_hr.shape == (2, 3, 32, 32), f"Expected (2, 3, 32, 32), got {batch_hr.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty tests (no model required)
# ──────────────────────────────────────────────────────────────────────────────

class TestUncertainty:
    def test_tta_ensemble_shape(self):
        """tta_ensemble returns correct shapes."""
        rng = np.random.default_rng(10)
        lr = rng.random((3, 16, 16), dtype=np.float32)

        # Mock enhance_fn: just bicubic upscale via numpy
        def mock_enhance(patch: np.ndarray) -> np.ndarray:
            import cv2
            c, h, w = patch.shape
            out = []
            for i in range(c):
                big = cv2.resize(patch[i], (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
                out.append(big)
            return np.stack(out, axis=0)

        mean_sr, unc = tta_ensemble(lr, mock_enhance, n_augmentations=4)
        assert mean_sr.shape == (3, 64, 64)
        assert unc.shape == (64, 64)

    def test_identical_augmentations_zero_uncertainty(self):
        """If model always returns the same output, uncertainty should be ~0."""
        rng = np.random.default_rng(11)
        lr = rng.random((3, 16, 16), dtype=np.float32)

        def constant_enhance(patch: np.ndarray) -> np.ndarray:
            import cv2
            c, h, w = patch.shape
            # Always return same fixed array regardless of input
            return np.zeros((c, h * 4, w * 4), dtype=np.float32)

        _, unc = tta_ensemble(lr, constant_enhance, n_augmentations=8)
        assert np.max(unc) < 1e-5, "Constant model should have near-zero uncertainty"

    def test_uncertainty_heatmap_shape(self):
        """uncertainty_to_heatmap returns uint8 RGB."""
        rng = np.random.default_rng(12)
        unc = rng.random((64, 64), dtype=np.float32) * 0.1
        heatmap = uncertainty_to_heatmap(unc)
        assert heatmap.shape == (64, 64, 3)
        assert heatmap.dtype == np.uint8

    def test_tta_ensemble_batched_multiband(self):
        """tta_ensemble_batched processes multi-band (4-band, 9-band) without 3-channel generator crash."""
        import torch
        import torch.nn as nn

        # Mock 3-channel in, 3-channel out generator (like RRDBNet)
        class MockGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.Upsample(scale_factor=4, mode="nearest")

            def forward(self, x):
                assert x.shape[1] == 3, f"Generator requires 3 channels, got {x.shape[1]}"
                return self.up(x)

        gen = MockGenerator()
        device = torch.device("cpu")

        # Test 4 bands (e.g. B04, B03, B02, B08)
        lr_4b = np.random.rand(4, 16, 16).astype(np.float32)
        mean_sr_4b, unc_4b = tta_ensemble_batched(
            lr_4b, gen, device, n_augmentations=4, rgb_band_indices=(0, 1, 2)
        )
        assert mean_sr_4b.shape == (4, 64, 64)
        assert unc_4b.shape == (64, 64)

        # Test 9 bands (Sentinel-2 9-band)
        lr_9b = np.random.rand(9, 16, 16).astype(np.float32)
        mean_sr_9b, unc_9b = tta_ensemble_batched(
            lr_9b, gen, device, n_augmentations=4, rgb_band_indices=(2, 1, 0)
        )
        assert mean_sr_9b.shape == (9, 64, 64)
        assert unc_9b.shape == (64, 64)



# ──────────────────────────────────────────────────────────────────────────────
# Metrics integration
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsIntegration:
    def test_evaluate_pair_on_pipeline_output(self):
        """Run degradation then evaluate — PSNR should be in a meaningful range."""
        x = np.linspace(0, 1, 64, dtype=np.float32)
        y = np.linspace(0, 1, 64, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx * 6) + np.cos(yy * 6) + 2) / 4.0
        hr = np.stack([pattern, pattern * 0.8, pattern * 0.5], axis=0).astype(np.float32)
        lr = degrade(hr, scale=4, apply_jpeg=False)

        # Simulate "SR" by bicubic upsampling the LR back
        import cv2
        sr_channels = []
        for i in range(lr.shape[0]):
            upsampled = cv2.resize(lr[i], (64, 64), interpolation=cv2.INTER_CUBIC)
            sr_channels.append(upsampled)
        sr = np.stack(sr_channels, axis=0).clip(0, 1)

        metrics = evaluate_pair(sr, hr)
        # Bicubic SR won't be perfect, but should be reasonable (>15 dB for smooth signal)
        assert metrics["psnr"] > 15.0, f"PSNR unexpectedly low: {metrics['psnr']}"
        assert metrics["ssim"] > 0.3, f"SSIM unexpectedly low: {metrics['ssim']}"
        assert 0.0 <= metrics["sam_deg"] <= 90.0


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing integration
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessingIntegration:
    def test_validate_and_preview(self, small_tile):
        meta = validate_tile(small_tile)
        assert meta["bands"] == 4

        rgb = extract_rgb_preview(small_tile)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8


# ──────────────────────────────────────────────────────────────────────────────
# Spectral Fusion Integration
# ──────────────────────────────────────────────────────────────────────────────

class TestSpectralFusion:
    def test_spectral_attention_shape(self):
        from src.spectral_fusion import ChannelSpectralAttention
        att = ChannelSpectralAttention(num_channels=4)
        x = torch.rand(2, 4, 16, 16)
        out = att(x)
        assert out.shape == (2, 4, 16, 16)

    def test_joint_spectral_sr_forward(self):
        import torch.nn as nn
        from src.spectral_fusion import JointSpectralSR

        class MockBaseGen(nn.Module):
            def forward(self, x):
                return F.interpolate(x, scale_factor=4, mode="nearest")

        base = MockBaseGen()
        joint_model = JointSpectralSR(base_generator=base, num_channels=4, rgb_band_indices=(0, 1, 2))
        x = torch.rand(2, 4, 16, 16)
        out = joint_model(x)
        assert out.shape == (2, 4, 64, 64)
        assert out.min() >= 0.0 and out.max() <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Fusion Integration
# ──────────────────────────────────────────────────────────────────────────────

class TestTemporalFusion:
    def test_vectorized_cloud_aware_fusion(self):
        from src.temporal_fusion import cloud_aware_fusion
        rng = np.random.default_rng(42)
        # Create a 3-frame temporal stack (T=3, C=4, H=32, W=32)
        stack = rng.random((3, 4, 32, 32), dtype=np.float32)
        # Artificially inject clouds (>0.85) in frame 0 and shadows (<0.15) in frame 1
        stack[0, :3, :10, :10] = 0.95
        stack[1, :3, 10:20, 10:20] = 0.05

        fused = cloud_aware_fusion(stack)
        assert fused.shape == (4, 32, 32)
        assert not np.isnan(fused).any(), "Fused output contains NaNs"

    def test_temporal_attention_module(self):
        from src.temporal_fusion import TemporalAttentionModule
        module = TemporalAttentionModule(in_channels=4, hidden_channels=16)
        stack_t = torch.rand(3, 4, 16, 16)
        fused_t = module(stack_t)
        assert fused_t.shape == (4, 16, 16)

