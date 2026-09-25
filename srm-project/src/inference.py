"""
inference.py
─────────────
End-to-end SR inference pipeline for a single Sentinel-2 GeoTIFF.

Pipeline:
  GeoTIFF in
    → tile into patches (src/preprocessing.py)
    → [optional] SR + uncertainty per patch (src/model.py + src/uncertainty.py)
    → reconstruct full SR image with geo-referenced output
    → [optional] compute PSNR/SSIM/SAM vs ground-truth (src/metrics.py)
    → save outputs to data/outputs/

Usage:
  python src/inference.py --input data/raw/tile.tif --output-dir data/outputs/
  python src/inference.py --input data/raw/tile.tif --gt data/raw/gt.tif --output-dir data/outputs/
  python src/inference.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Hann-window blend helper
# ──────────────────────────────────────────────────────────────────────────────

def _make_hann_window(h: int, w: int) -> np.ndarray:
    """
    Build a 2-D Hann window of shape (h, w).

    Always built from the *actual* patch dimensions so that even boundary
    patches (which may be smaller than the full tile after canvas clipping)
    receive a complete 0 → 1 → 0 taper across their full extent.  This
    prevents the near-zero leading-edge slice that caused bright/dark seams
    at the right and bottom edges of the scene.
    """
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    return np.outer(win_h, win_w).astype(np.float32)


def unsharp_mask(image: np.ndarray, strength: float = 0.35, sigma: float = 1.0) -> np.ndarray:
    """Apply edge-preserving unsharp mask to boost building outlines and linear road features."""
    import cv2
    blurred = np.empty_like(image)
    for c in range(image.shape[0]):
        blurred[c] = cv2.GaussianBlur(image[c], (0, 0), sigma)
    detail = image - blurred
    return np.clip(image + strength * detail, 0.0, 1.0)


def detect_edges(image: np.ndarray) -> np.ndarray:
    """Detect edges using Sobel operator for adaptive sharpening."""
    import cv2
    c, h, w = image.shape
    edges = np.zeros((c, h, w), dtype=np.float32)
    for i in range(c):
        gray = (image[i] * 255).astype(np.uint8)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges[i] = np.sqrt(sobel_x**2 + sobel_y**2) / 255.0
    return edges


def edge_aware_sharpen(sr_image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply stronger sharpening only on textured/edge regions, not flat areas."""
    edges = detect_edges(sr_image)
    # Normalize edges to [0, 1] per band
    for i in range(sr_image.shape[0]):
        edge_max = edges[i].max()
        if edge_max > 0:
            edges[i] = edges[i] / edge_max

    # Apply sharpening scaled by edge strength
    sharpened = unsharp_mask(sr_image, strength=strength * 0.5)
    # Blend based on edge presence
    result = np.zeros_like(sr_image)
    for i in range(sr_image.shape[0]):
        edge_mask = edges[i] ** 1.5  # Boost high-edge regions
        result[i] = sr_image[i] * (1 - edge_mask * 0.5) + sharpened[i] * edge_mask * 0.5

    return result


def multi_scale_sharpen(sr_image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply unsharp mask at multiple scales for better detail recovery."""
    result = sr_image.copy()

    # Multi-scale: fine details (sigma=0.5), medium (sigma=1.0), coarse (sigma=2.0)
    for sigma in [0.5, 1.0, 2.0]:
        usm = unsharp_mask(sr_image, strength=strength * 0.35, sigma=sigma)
        result = result * 0.5 + usm * 0.5

    return result


def enhanced_post_process(sr_image: np.ndarray, lr_image: Optional[np.ndarray] = None,
                          strength: float = 0.5, high_quality: bool = False) -> np.ndarray:
    """
    Advanced post-processing for better visual quality.

    1. Multi-scale unsharp mask - preserves both fine details and broader structures
    2. Edge-aware sharpening - stronger sharpening on textured regions only
    3. Optional detail injection from original LR image
    4. Contrast enhancement via CLAHE for local contrast improvement

    Parameters
    ----------
    sr_image : float32 (C, H, W) in [0, 1] - SR output from model
    lr_image : float32 (C, H, W) in [0, 1] - Original LR input (optional, for detail injection)
    strength : float - overall sharpening strength (0.0 to 1.0)
    high_quality : bool - enable additional quality enhancements

    Returns
    -------
    float32 (C, H, W) in [0, 1] - processed image
    """
    import cv2

    result = sr_image.copy()

    # Step 1: Multi-scale sharpening
    result = multi_scale_sharpen(result, strength=strength)

    # Step 2: Edge-aware sharpening
    result = edge_aware_sharpen(result, strength=strength)

    # Step 3: Optional detail injection from LR (high-frequency preservation)
    if high_quality and lr_image is not None:
        # Scale LR to match SR dimensions
        c, sr_h, sr_w = result.shape
        lr_c, lr_h, lr_w = lr_image.shape

        # Compute scale factor
        scale = sr_h // lr_h

        # Process each band
        for i in range(min(c, lr_c)):
            lr = lr_image[i]
            # Extract high-frequency details from LR
            blurred_lr = cv2.GaussianBlur(lr, (0, 0), sigma=0.5)
            laplacian = lr - blurred_lr

            # Upscale detail to SR dimensions
            detail_up = cv2.resize(
                laplacian,
                (sr_w, sr_h),
                interpolation=cv2.INTER_CUBIC,
            )

            # Inject detail into SR output
            result[i] = result[i] + detail_up * 0.3 * strength

    # Step 4: CLAHE for local contrast enhancement (optional, for high quality)
    if high_quality:
        result = apply_clahe(result)

    return result.clip(0, 1)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization per channel.

    Improves local contrast without over-amplifying noise.
    """
    import cv2
    c, h, w = image.shape
    result = np.zeros_like(image)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

    for i in range(c):
        # Convert to 8-bit for CLAHE
        img_8bit = (image[i] * 255).astype(np.uint8)
        clahe_img = clahe.apply(img_8bit)
        result[i] = clahe_img.astype(np.float32) / 255.0

    return result


def run_inference(
    input_path: Path | str,
    output_dir: Path | str = "data/outputs",
    output_path: Optional[Path | str] = None,
    gt_path: Optional[Path | str] = None,
    checkpoint_path: Optional[Path | str] = None,
    model_key: str = "x4plus",
    scale: int = 4,
    patch_size: int = 512,
    overlap: int = 64,
    compute_uncertainty: bool = True,
    tta_n: int = 4,
    tile_size: int = 256,
    half: bool = False,
    rgb_band_indices: tuple = (2, 1, 0),
    sharpen: float = 0.5,
    high_quality: bool = False,
) -> dict:
    """
    Run full inference on a single GeoTIFF and save outputs.

    Returns a dict with output paths and computed metrics (if GT provided).
    """
    import torch
    from src.model import load_realesrgan, enhance_multiband, _generator_from_upsampler
    from src.preprocessing import validate_tile, extract_rgb_preview
    from src.uncertainty import tta_ensemble_batched, save_uncertainty_heatmap
    from src.metrics import evaluate_pair

    input_path = Path(input_path)
    if output_path is not None:
        sr_tif_path = Path(output_path)
        output_dir = sr_tif_path.parent
        stem = sr_tif_path.stem
    else:
        output_dir = Path(output_dir)
        stem = input_path.stem
        sr_tif_path = output_dir / f"{stem}_sr_x{scale}.tif"

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-enable FP16 on CUDA for speed boost
    if device.type == "cuda" and not half:
        half = True
        logger.info("Auto-enabling FP16 inference (CUDA detected)")

    logger.info("Device: %s | FP16: %s", device, half)

    # ── Validate input ────────────────────────────────────────────────────────
    meta = validate_tile(input_path)
    logger.info("Input: %s | %d×%d | %d bands", stem, meta["width"], meta["height"], meta["bands"])

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading Real-ESRGAN …")
    upsampler = load_realesrgan(
        model_key=model_key,
        checkpoint_path=checkpoint_path,
        scale=scale,
        # BUG-1 FIX: pass tile=0 to disable RealESRGANer's *internal* tiling.
        # The library's built-in tiler uses reflect-padding with its own
        # tile_pad boundary, which creates a second independent grid of seams
        # underneath our outer Hann-blended loop.  Setting tile=0 forces the
        # library to process each outer patch in one shot, leaving all
        # boundary handling to our weighted blend.
        tile=0,
        half=(half and device.type == "cuda"),
        device=device,
    )

    # Extract bare generator for batched fast paths (Fix A + Fix B).
    # upsampler.model is the RRDBNet; _generator_from_upsampler puts it in eval().
    generator = _generator_from_upsampler(upsampler)

    # ── Read entire input image ───────────────────────────────────────────────
    with rasterio.open(input_path) as src:
        crs = src.crs
        orig_transform = src.transform
        orig_h = src.height
        orig_w = src.width
        n_bands = src.count
        dtype = src.dtypes[0]
        max_val = np.iinfo(np.dtype(dtype)).max if np.issubdtype(np.dtype(dtype), np.integer) else 1.0
        full_data = src.read().astype(np.float32) / max_val   # (C, H, W)

    # ── Tiled inference with Hann blending ────────────────────────────────────
    sr_h = orig_h * scale
    sr_w = orig_w * scale
    canvas = np.zeros((n_bands, sr_h, sr_w), dtype=np.float32)
    weight_map = np.zeros((sr_h, sr_w), dtype=np.float32)
    unc_canvas = np.zeros((sr_h, sr_w), dtype=np.float32)
    unc_weight = np.zeros((sr_h, sr_w), dtype=np.float32)

    stride = patch_size - overlap
    row_starts = list(range(0, orig_h - patch_size + 1, stride))
    if not row_starts or row_starts[-1] + patch_size < orig_h:
        row_starts.append(max(0, orig_h - patch_size))
    col_starts = list(range(0, orig_w - patch_size + 1, stride))
    if not col_starts or col_starts[-1] + patch_size < orig_w:
        col_starts.append(max(0, orig_w - patch_size))

    total_patches = len(row_starts) * len(col_starts)
    logger.info("Processing %d patches (patch_size=%d, overlap=%d) …", total_patches, patch_size, overlap)
    t0 = time.time()
    processed = 0
    # BUG-2 FIX: Hann window is now built *per patch* inside the loop from the
    # actual SR output dimensions (see _make_hann_window call below).

    for row_off in row_starts:
        for col_off in col_starts:
            # Extract patch
            lr_patch = full_data[:, row_off:row_off + patch_size, col_off:col_off + patch_size]

            # Skip low-content patches
            if np.mean(lr_patch > 0) < 0.3:
                processed += 1
                continue

            t_patch = time.time()

            if compute_uncertainty:
                # FAST PATH (Fix A): all 8 TTA augmentations run in one batched
                # generator.forward() call instead of 8 sequential enhance() calls.
                sr_patch, unc_patch = tta_ensemble_batched(
                    lr_patch, generator, device,
                    n_augmentations=tta_n,
                    half=(half and device.type == "cuda"),
                    rgb_band_indices=rgb_band_indices,
                )
            else:
                # FAST PATH (Fix B): pass generator so grayscale bands are batched.
                sr_patch = enhance_multiband(
                    upsampler, lr_patch,
                    rgb_band_indices=rgb_band_indices,
                    outscale=scale,
                    generator=generator,
                )
                unc_patch = np.zeros((sr_patch.shape[1], sr_patch.shape[2]), dtype=np.float32)

            # Log first-patch wall time so speedup is immediately visible
            if processed == 0:
                logger.info(
                    "  First patch completed in %.2fs (patch_size=%d, tta=%s, bands=%d)",
                    time.time() - t_patch, patch_size,
                    "batched×%d" % tta_n if compute_uncertainty else "off",
                    lr_patch.shape[0],
                )

            # SR patch position in output canvas
            sr_row = row_off * scale
            sr_col = col_off * scale
            sr_ph = sr_patch.shape[1]
            sr_pw = sr_patch.shape[2]

            # Clip to canvas bounds
            row_end = min(sr_row + sr_ph, sr_h)
            col_end = min(sr_col + sr_pw, sr_w)
            pr_end = row_end - sr_row
            pc_end = col_end - sr_col

            # BUG-2 FIX: build the Hann window from the *actual* dimensions of
            # the valid SR region (pr_end × pc_end) rather than slicing a
            # pre-built max-size window.  Slicing would grab only the near-zero
            # leading edge of the window for boundary patches, causing the
            # weight_map to accumulate almost nothing there → divide-by-near-
            # zero → bright / dark seams at the right and bottom scene edges.
            hann = _make_hann_window(pr_end, pc_end)

            canvas[:, sr_row:row_end, sr_col:col_end] += (
                sr_patch[:, :pr_end, :pc_end] * hann
            )
            weight_map[sr_row:row_end, sr_col:col_end] += hann
            unc_canvas[sr_row:row_end, sr_col:col_end] += (
                unc_patch[:pr_end, :pc_end] * hann
            )
            unc_weight[sr_row:row_end, sr_col:col_end] += hann

            processed += 1
            if processed % 10 == 0 or processed == total_patches:
                elapsed = time.time() - t0
                logger.info("  %d/%d patches (%.1fs)", processed, total_patches, elapsed)

    # Normalise by blend weights
    wm = np.where(weight_map > 0, weight_map, 1.0)
    sr_full = (canvas / wm).astype(np.float32).clip(0, 1)

    # Apply post-processing
    if sharpen > 0.0 or high_quality:
        if high_quality:
            logger.info(f"Applying enhanced post-processing (HQ mode: sharpen={sharpen:.2f}) …")
            sr_full = enhanced_post_process(sr_full, full_data, strength=sharpen, high_quality=True)
        else:
            logger.info(f"Applying standard sharpening (strength={sharpen:.2f}) …")
            sr_full = multi_scale_sharpen(sr_full, strength=sharpen)

    uw = np.where(unc_weight > 0, unc_weight, 1.0)
    unc_full = (unc_canvas / uw).astype(np.float32)

    elapsed_total = time.time() - t0
    logger.info("Inference complete in %.1fs", elapsed_total)

    # ── Save SR GeoTIFF with preserved geo-referencing ────────────────────────
    sr_transform = Affine(
        orig_transform.a / scale, orig_transform.b, orig_transform.c,
        orig_transform.d, orig_transform.e / scale, orig_transform.f,
    )
    with rasterio.open(
        sr_tif_path, "w",
        driver="GTiff",
        height=sr_h, width=sr_w,
        count=n_bands,
        dtype="float32",
        crs=crs,
        transform=sr_transform,
        compress="lzw",
    ) as dst:
        dst.write(sr_full)
    logger.info("SR GeoTIFF saved → %s", sr_tif_path)

    # ── Save uncertainty heatmap & npy ────────────────────────────────────────
    unc_path = output_dir / f"{stem}_uncertainty.png"
    if compute_uncertainty:
        # Extract RGB preview for blended display
        rgb_idx = [i for i in rgb_band_indices if i < n_bands]
        if len(rgb_idx) >= 3:
            from src.preprocessing import extract_rgb_preview
            sr_rgb = extract_rgb_preview(sr_tif_path, band_indices=tuple(rgb_idx))
        else:
            ch = sr_full[0]
            sr_rgb = np.stack([ch, ch, ch], axis=-1)
            sr_rgb = (sr_rgb * 255).astype(np.uint8)
        save_uncertainty_heatmap(unc_full, unc_path, sr_image_rgb=sr_rgb)
        np.save(output_dir / f"{stem}_uncertainty.npy", unc_full)
        logger.info("Uncertainty heatmap saved → %s", unc_path)

    # ── Compute metrics vs ground truth (if provided) ─────────────────────────
    metrics = {}
    if gt_path is not None:
        gt_path = Path(gt_path)
        logger.info("Computing metrics vs GT: %s", gt_path.name)
        with rasterio.open(gt_path) as gt_src:
            gt_data = gt_src.read().astype(np.float32)
            gt_max = np.iinfo(np.dtype(gt_src.dtypes[0])).max if np.issubdtype(
                np.dtype(gt_src.dtypes[0]), np.integer
            ) else 1.0
            gt_data /= gt_max

        # Match spatial extent (crop to minimum of sr / gt)
        common_c = min(sr_full.shape[0], gt_data.shape[0])
        common_h = min(sr_full.shape[1], gt_data.shape[1])
        common_w = min(sr_full.shape[2], gt_data.shape[2])
        pred_crop = sr_full[:common_c, :common_h, :common_w]
        gt_crop = gt_data[:common_c, :common_h, :common_w]

        from src.metrics import evaluate_pair
        metrics = evaluate_pair(pred_crop, gt_crop)
        logger.info("Metrics: PSNR=%.2f dB | SSIM=%.4f | SAM=%.4f°",
                    metrics["psnr"], metrics["ssim"], metrics["sam_deg"])

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "input": str(input_path),
        "sr_output": str(sr_tif_path),
        "uncertainty_map": str(unc_path) if compute_uncertainty else None,
        "scale": scale,
        "elapsed_seconds": round(elapsed_total, 2),
        "crs": str(crs),
        "sr_shape": [n_bands, sr_h, sr_w],
        "metrics": metrics or None,
    }
    report_path = output_dir / f"{stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved → %s", report_path)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Super-Resolution inference (SIH26142)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input GeoTIFF path")
    parser.add_argument("--output", default=None, help="Explicit output GeoTIFF file path (optional)")
    parser.add_argument("--output-dir", default="data/outputs", help="Output directory")
    parser.add_argument("--gt", default=None, help="Ground-truth GeoTIFF (optional, for metrics)")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint .pth (optional)")
    parser.add_argument("--model-key", default="x4plus", choices=["x4plus", "x4plus_anime"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size (default 512 for speed)")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between patches (default 64)")
    parser.add_argument("--tile-size", type=int, default=256, help="Real-ESRGAN tile size (VRAM trade-off)")
    parser.add_argument("--no-uncertainty", action="store_true", help="Skip uncertainty estimation (faster)")
    parser.add_argument("--tta-n", type=int, default=4, help="Number of TTA augmentations (default 4, use 8 for max quality)")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference (CUDA only)")
    parser.add_argument("--sharpen", type=float, default=0.5, help="Edge sharpening strength (0.0 = off, 0.3-0.6 = crisp)")
    parser.add_argument("--verbose", "-v", action="store_true")

    # Preset modes for quick access
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default=None,
        help="Quick preset: fast=(512px,no TTA), balanced=(512px,4 TTA), quality=(384px,8 TTA)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Apply preset values while allowing explicit CLI options to remain useful.
    # Presets are intended for the SIH demo: fast prioritizes latency, quality
    # prioritizes TTA stability and overlap blending.
    preset_values = {
        "fast": {"patch_size": 512, "overlap": 48, "tta_n": 0, "sharpen": 0.4, "uncertainty": False},
        "balanced": {"patch_size": 512, "overlap": 64, "tta_n": 4, "sharpen": 0.5, "uncertainty": True},
        "quality": {"patch_size": 384, "overlap": 64, "tta_n": 8, "sharpen": 0.6, "uncertainty": True},
    }
    selected = preset_values.get(args.preset)
    if selected:
        args.patch_size = selected["patch_size"]
        args.overlap = selected["overlap"]
        args.tta_n = selected["tta_n"]
        args.sharpen = selected["sharpen"]
        compute_uncertainty = selected["uncertainty"]
    else:
        compute_uncertainty = not args.no_uncertainty

    result = run_inference(
        input_path=args.input,
        output_dir=args.output_dir,
        output_path=args.output,
        gt_path=args.gt,
        checkpoint_path=args.checkpoint,
        model_key=args.model_key,
        scale=args.scale,
        patch_size=args.patch_size,
        overlap=args.overlap,
        compute_uncertainty=compute_uncertainty,
        tta_n=args.tta_n,
        tile_size=args.tile_size,
        half=args.half,
        sharpen=args.sharpen,
        high_quality=(args.preset == "quality"),
    )

    print("\n" + "═" * 60)
    print("SR OUTPUT :", result["sr_output"])
    print("UNCERTAINTY:", result["uncertainty_map"])
    if result["metrics"]:
        m = result["metrics"]
        print(f"PSNR  : {m['psnr']:.2f} dB")
        print(f"SSIM  : {m['ssim']:.4f}")
        print(f"SAM   : {m['sam_deg']:.4f}°")
    print("═" * 60)
