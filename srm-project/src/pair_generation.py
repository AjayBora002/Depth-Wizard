"""
pair_generation.py
───────────────────
Generate synthetic (low-resolution, high-resolution) training pairs from
Sentinel-2 10m GeoTIFF tiles using a realistic degradation pipeline.

DESIGN DECISION (documented):
  True paired LR/HR satellite datasets at Sentinel-2 / sub-4m resolution are
  not freely available at scale. We use controlled downsampling of Sentinel-2
  10m tiles to simulate lower-resolution imagery (nominally ~40m-equivalent,
  i.e. 4× downscale), creating synthetic training pairs.

  Degradation model:
    1. Gaussian blur  (σ drawn from U[0.5, 1.5])       — PSF blur
    2. 4× bicubic downscale                             — pixel-mixing
    3. Gaussian noise (σ drawn from U[1, 5] / 255)     — sensor noise
    4. Optional JPEG compression at quality Q ∈ {75-95} — compression artifact

  The original 10m patch is the high-resolution ground-truth.
  The degraded patch is the low-resolution input.

  This is clearly documented and does NOT represent real-sensor LR imagery.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Numpy-based Gaussian blur using cv2 (no scipy dependency for Windows)
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_filter_numpy(image, sigma):
    """Apply Gaussian filter using cv2."""
    # cv2.GaussianBlur expects (H, W) or (H, W, C) input
    if image.ndim == 2:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif image.ndim == 3:
        channels = [cv2.GaussianBlur(image[..., c], (0, 0), sigmaX=sigma, sigmaY=sigma) for c in range(image.shape[-1])]
        return np.stack(channels, axis=-1)
    else:
        return image


# ──────────────────────────────────────────────────────────────────────────────
# Degradation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def degrade(
    hr_patch: np.ndarray,
    scale: int = 4,
    blur_sigma_range: Tuple[float, float] = (0.5, 1.5),
    noise_sigma_range: Tuple[float, float] = (1.0, 5.0),
    jpeg_quality_range: Tuple[int, int] = (75, 95),
    apply_jpeg: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply the degradation model to an HR patch and return the LR patch.

    Parameters
    ----------
    hr_patch : float32 (C, H, W) in [0, 1]
    scale    : downscale factor (default 4)

    Returns
    -------
    lr_patch : float32 (C, H/scale, W/scale) in [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    c, h, w = hr_patch.shape
    lr_h, lr_w = h // scale, w // scale

    # Work in (H, W, C) for OpenCV compatibility
    img = hr_patch.transpose(1, 2, 0)   # (H, W, C) float32 [0,1]

    # ── Step 1: Gaussian blur (PSF simulation) ────────────────────────────────
    sigma = rng.uniform(*blur_sigma_range)
    if c == 1:
        img = gaussian_filter_numpy(img[..., 0], sigma=sigma)[..., np.newaxis]
    else:
        blurred = np.stack(
            [gaussian_filter_numpy(img[..., i], sigma=sigma) for i in range(c)],
            axis=-1,
        )
        img = blurred

    # ── Step 2: Bicubic downscale ─────────────────────────────────────────────
    if c <= 4:
        # OpenCV handles up to 4-channel images
        img_cv = (img * 65535).clip(0, 65535).astype(np.uint16)
        img_lr_cv = cv2.resize(
            img_cv, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC
        )
        img = img_lr_cv.astype(np.float32) / 65535.0
    else:
        # Fall back to channel-wise numpy resize for > 4 channels
        out_channels = []
        for i in range(c):
            ch = img[..., i]
            ch_cv = (ch * 65535).clip(0, 65535).astype(np.uint16)
            ch_lr = cv2.resize(ch_cv, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            out_channels.append(ch_lr.astype(np.float32) / 65535.0)
        img = np.stack(out_channels, axis=-1)

    # ── Step 3: Gaussian noise (sensor noise) ────────────────────────────────
    noise_sigma = rng.uniform(*noise_sigma_range) / 255.0
    noise = rng.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = (img + noise).clip(0, 1)

    # ── Step 4: Optional JPEG compression ────────────────────────────────────
    if apply_jpeg and c in (1, 3):
        quality = int(rng.integers(*jpeg_quality_range))
        # Encode / decode cycle on uint8
        img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
        if c == 1:
            encode_img = img_u8[..., 0]
            _, buf = cv2.imencode(".jpg", encode_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            img = img[..., np.newaxis]
        else:
            encode_img = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", encode_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return img.transpose(2, 0, 1)   # back to (C, H/scale, W/scale)


# ──────────────────────────────────────────────────────────────────────────────
# Batch pair generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_pairs_from_tile(
    tile_path: Path | str,
    out_dir: Path | str,
    patch_size: int = 512,
    overlap: int = 32,
    scale: int = 4,
    rgb_only: bool = False,
    apply_jpeg: bool = True,
    max_patches: Optional[int] = None,
    seed: Optional[int] = 42,
) -> int:
    """
    Tile a GeoTIFF, apply the degradation model, and save (LR, HR) patch pairs.

    Pair filenames: {stem}_{row:04d}_{col:04d}_lr.npy / _hr.npy

    Supports diverse geographic locations, multi-band or single-band tiles,
    and optional patch limits per tile for balanced training.

    Returns the number of pairs generated.
    """
    import rasterio
    from rasterio.windows import Window

    tile_path = Path(tile_path)
    out_dir = Path(out_dir)
    lr_dir = out_dir / "lr"
    hr_dir = out_dir / "hr"
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    stem = tile_path.stem
    count = 0

    with rasterio.open(tile_path) as src:
        n_bands = src.count
        H, W = src.height, src.width
        max_val = np.iinfo(np.dtype(src.dtypes[0])).max if np.issubdtype(
            np.dtype(src.dtypes[0]), np.integer
        ) else 1.0

        # Adjust patch size if tile is smaller than default patch_size
        eff_patch = min(patch_size, min(H, W))
        eff_patch = (eff_patch // scale) * scale
        if eff_patch < scale * 8:
            logger.warning("Tile %s is too small (%dx%d), skipping", tile_path.name, W, H)
            return 0

        stride = max(scale, eff_patch - overlap)
        row_starts = list(range(0, H - eff_patch + 1, stride))
        if not row_starts or row_starts[-1] + eff_patch < H:
            row_starts.append(max(0, H - eff_patch))
        col_starts = list(range(0, W - eff_patch + 1, stride))
        if not col_starts or col_starts[-1] + eff_patch < W:
            col_starts.append(max(0, W - eff_patch))

        bands_to_use = [1, 2, 3] if (rgb_only and n_bands >= 3) else list(range(1, n_bands + 1))

        for ri, row_off in enumerate(row_starts):
            if max_patches is not None and count >= max_patches:
                break
            for ci, col_off in enumerate(col_starts):
                if max_patches is not None and count >= max_patches:
                    break
                win = Window(col_off, row_off, eff_patch, eff_patch)
                data = src.read(bands_to_use, window=win)   # (C, H, W)

                # Ensure 3-channel RGB consistency if rgb_only is requested
                if rgb_only and data.shape[0] == 1:
                    data = np.repeat(data, 3, axis=0)

                # Skip mostly-zero / empty border patches
                if np.sum(data > 0) / data.size < 0.5:
                    continue

                hr = (data.astype(np.float32) / max_val).clip(0, 1)
                lr = degrade(hr, scale=scale, apply_jpeg=apply_jpeg, rng=rng)

                fname = f"{stem}_{ri:04d}_{ci:04d}"
                np.save(lr_dir / f"{fname}_lr.npy", lr)
                np.save(hr_dir / f"{fname}_hr.npy", hr)
                count += 1

    logger.info("Generated %d pairs from %s → %s", count, tile_path.name, out_dir)
    return count


def generate_all_pairs(
    raw_dir: Path | str = Path("data/raw"),
    out_dir: Path | str = Path("data/synthetic_pairs"),
    patch_size: int = 512,
    overlap: int = 64,
    scale: int = 4,
    rgb_only: bool = False,
    patches_per_tile: Optional[int] = None,
    max_patches_per_tile: Optional[int] = None,
    seed: int = 42,
) -> int:
    """
    Generate pairs from all GeoTIFFs across different locations in raw_dir.
    Supports both .tif and .tiff files.
    Returns total pair count.
    """
    raw_dir = Path(raw_dir)
    limit = max_patches_per_tile if max_patches_per_tile is not None else patches_per_tile

    # Find all .tif and .tiff files (case-insensitive)
    tifs = sorted([
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    ])
    if not tifs:
        logger.error("No GeoTIFF files found in %s", raw_dir)
        return 0

    logger.info("Discovered %d raw GeoTIFF tiles across locations in %s", len(tifs), raw_dir)
    total = 0
    for tif in tifs:
        total += generate_pairs_from_tile(
            tif,
            out_dir,
            patch_size=patch_size,
            overlap=overlap,
            scale=scale,
            rgb_only=rgb_only,
            max_patches=limit,
            seed=seed,
        )
    logger.info("Total pairs generated across all locations: %d", total)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class (for training)
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticPairDataset:
    """
    PyTorch-compatible dataset of (lr, hr) numpy patch pairs.
    Returns float32 tensors with values in [0, 1].
    """

    def __init__(self, pairs_dir: Path | str, split: str = "train", val_fraction: float = 0.1):
        import torch  # noqa: F401 — deferred so module imports without PyTorch

        self.lr_files = sorted((Path(pairs_dir) / "lr").glob("*_lr.npy"))
        if not self.lr_files:
            raise FileNotFoundError(f"No LR patch files found in {pairs_dir}/lr/")

        rng = random.Random(42)
        rng.shuffle(self.lr_files)
        n_val = max(1, int(len(self.lr_files) * val_fraction))
        if split == "val":
            self.lr_files = self.lr_files[:n_val]
        else:
            self.lr_files = self.lr_files[n_val:]

        logger.info("Dataset (%s): %d pairs in %s", split, len(self.lr_files), pairs_dir)

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int):
        import torch

        lr_path = self.lr_files[idx]
        hr_path = lr_path.parent.parent / "hr" / lr_path.name.replace("_lr.npy", "_hr.npy")

        lr = torch.from_numpy(np.load(lr_path)).float()
        hr = torch.from_numpy(np.load(hr_path)).float()
        return lr, hr


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic LR/HR training pairs")
    parser.add_argument("--raw-dir", default="data/raw", help="Input GeoTIFF directory")
    parser.add_argument("--out-dir", default="data/synthetic_pairs", help="Output directory")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between consecutive patches")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--rgb-only", action="store_true",
                        help="Only use RGB bands (faster, good for display)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = generate_all_pairs(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        scale=args.scale,
        rgb_only=args.rgb_only,
        seed=args.seed,
    )
    print(f"\nDone. Generated {n} training pairs.")
