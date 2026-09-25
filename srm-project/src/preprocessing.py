"""
preprocessing.py
─────────────────
Sentinel-2 GeoTIFF utilities:
  - Open tiles, validate CRS / transform
  - Tile large scenes into overlapping patches (preserving geo-metadata per patch)
  - Reconstruct full image from patches (for inference output)
  - Save output as geo-referenced GeoTIFF

Sentinel-2 L2A band order expected in the stacked GeoTIFF:
  Band index  Sentinel band  λ centre   Resolution
  0           B02 (Blue)     490 nm     10 m
  1           B03 (Green)    560 nm     10 m
  2           B04 (Red)      665 nm     10 m
  3           B08 (NIR)      842 nm     10 m
  4           B05            705 nm     20 m (resampled to 10 m)
  5           B06            740 nm     20 m (resampled)
  6           B07            783 nm     20 m (resampled)
  7           B11 (SWIR1)    1610 nm    20 m (resampled)
  8           B12 (SWIR2)    2190 nm    20 m (resampled)

If the file has fewer bands (e.g. just RGB) the code handles it gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds, Affine
from rasterio.windows import Window

logger = logging.getLogger(__name__)

# ─── RGB band indices (0-based) in a standard Sentinel-2 stack ───────────────
S2_RGB_IDX = (2, 1, 0)   # B04, B03, B02 → R, G, B


# ──────────────────────────────────────────────────────────────────────────────
# Tile / patch utilities
# ──────────────────────────────────────────────────────────────────────────────

def tile_image(
    src_path: Path | str,
    patch_size: int = 512,
    overlap: int = 64,
    min_valid_fraction: float = 0.5,
) -> Generator[Tuple[np.ndarray, rasterio.transform.Affine, dict], None, None]:
    """
    Yield (patch_array, patch_transform, meta) tuples for every valid patch
    extracted from a GeoTIFF.

    patch_array  : float32 array of shape (C, H, W), normalised to [0, 1]
    patch_transform : Affine transform for this specific patch window
    meta         : rasterio metadata dict (crs, dtype, count …)
    """
    src_path = Path(src_path)
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        full_width = src.width
        full_height = src.height
        stride = patch_size - overlap

        row_starts = list(range(0, full_height - patch_size + 1, stride))
        if not row_starts or row_starts[-1] + patch_size < full_height:
            row_starts.append(max(0, full_height - patch_size))

        col_starts = list(range(0, full_width - patch_size + 1, stride))
        if not col_starts or col_starts[-1] + patch_size < full_width:
            col_starts.append(max(0, full_width - patch_size))

        for row_off in row_starts:
            for col_off in col_starts:
                win = Window(col_off, row_off, patch_size, patch_size)
                data = src.read(window=win)          # (C, H, W) uint16 or float

                # Skip patches with too many nodata / zero pixels
                valid = np.sum(data > 0) / data.size
                if valid < min_valid_fraction:
                    continue

                # Normalise to [0, 1] float32.
                # BUG-3 FIX: for float-dtype inputs always use 1.0 as the
                # scale factor.  The old code used data.max() per tile, which
                # made each tile independently stretch to its own local peak
                # and caused a visible brightness grid after blending.  Using
                # the dtype's full-range maximum (integer) or the fixed 1.0
                # (float) ensures globally consistent normalisation.
                dtype = src.dtypes[0]
                if np.issubdtype(np.dtype(dtype), np.integer):
                    max_val = np.iinfo(np.dtype(dtype)).max
                else:
                    max_val = 1.0   # float GeoTIFFs are expected to already be in [0, 1]
                patch = (data.astype(np.float32) / max_val).clip(0, 1)

                patch_transform = src.window_transform(win)
                yield patch, patch_transform, meta


def save_patch(
    array: np.ndarray,
    transform: Affine,
    crs,
    out_path: Path | str,
    dtype: str = "float32",
) -> None:
    """Save a (C, H, W) numpy array as a GeoTIFF with the given transform/CRS."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c, h, w = array.shape
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(array.astype(dtype))


# ──────────────────────────────────────────────────────────────────────────────
# GeoTIFF validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_tile(path: Path | str) -> dict:
    """
    Open a GeoTIFF and return a metadata dict.
    Raises ValueError with a descriptive message if the tile is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tile not found: {path}")

    with rasterio.open(path) as src:
        info = {
            "path": str(path),
            "crs": str(src.crs),
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes[0],
            "bounds": src.bounds,
            "nodata": src.nodata,
        }

    if info["crs"] is None or info["crs"] == "None":
        raise ValueError(f"Tile has no CRS: {path}")
    if info["width"] < 64 or info["height"] < 64:
        raise ValueError(f"Tile too small ({info['width']}×{info['height']}): {path}")

    logger.info(
        "✓ %s | %s | %d×%d | %d bands | %s",
        path.name,
        info["crs"],
        info["width"],
        info["height"],
        info["bands"],
        info["dtype"],
    )
    return info


def validate_all_tiles(data_raw_dir: Path | str) -> List[dict]:
    """Validate every GeoTIFF in the raw data directory."""
    data_raw_dir = Path(data_raw_dir)
    tifs = sorted(data_raw_dir.glob("**/*.tif")) + sorted(data_raw_dir.glob("**/*.tiff"))
    if not tifs:
        logger.warning("No GeoTIFF files found in %s", data_raw_dir)
        return []

    results = []
    for p in tifs:
        try:
            results.append(validate_tile(p))
        except Exception as exc:
            logger.error("✗ %s: %s", p.name, exc)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Extract RGB preview (for display in dashboard)
# ──────────────────────────────────────────────────────────────────────────────

def extract_rgb_preview(
    path: Path | str,
    band_indices: tuple = S2_RGB_IDX,
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Return a uint8 (H, W, 3) RGB array suitable for display.
    Uses percentile stretch for contrast normalisation.
    """
    with rasterio.open(path) as src:
        n_bands = src.count
        # Clamp band indices to available bands
        valid_idx = [min(i, n_bands - 1) + 1 for i in band_indices]
        rgb = src.read(valid_idx).astype(np.float32)   # (3, H, W)

    # Per-band percentile clip
    out = np.zeros_like(rgb)
    for i in range(3):
        lo, hi = np.percentile(rgb[i][rgb[i] > 0], percentile_clip)
        out[i] = np.clip((rgb[i] - lo) / (hi - lo + 1e-8), 0, 1)

    return (out.transpose(1, 2, 0) * 255).astype(np.uint8)   # (H, W, 3)


# NOTE: Patch reconstruction and Hann-window overlap blending are implemented in
# src/inference.py (run_inference) using geo-referencing coordinates and SR offsets.


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Validate Sentinel-2 tiles in data/raw/")
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing GeoTIFF tiles (default: data/raw)",
    )
    args = parser.parse_args()

    results = validate_all_tiles(args.data_dir)
    print("\n" + "-" * 60)
    print(f"Validated {len(results)} tile(s)")
    for r in results:
        print(f"  {Path(r['path']).name}: {r['width']}x{r['height']} | {r['crs']} | {r['bands']} bands")
