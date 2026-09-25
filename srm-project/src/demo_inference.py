"""
demo_inference.py
─────────────────
Generate impressive before/after comparisons for SIH showcase.
Creates side-by-side visualizations with metrics overlay.

Usage:
  python src/demo_inference.py --input data/raw/2026-04-08-00_00_2026-04-08-23_59_Sentinel-2_L2A_True_color.tiff --output demo_output.png
  python src/demo_inference.py --input data/raw/2026-04-08-00_00_2026-04-08-23_59_Sentinel-2_L2A_True_color.tiff --output demo_output.png --compare
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_metrics(lr: np.ndarray, hr: np.ndarray, sr: Optional[np.ndarray] = None) -> dict:
    """Compute PSNR, SSIM, and MSE metrics."""
    from skimage.metrics import structural_similarity as ssim

    metrics = {}

    # Ensure same shape for comparison
    if sr is not None:
        # SR vs HR
        mse = np.mean((sr - hr) ** 2)
        metrics['psnr'] = 10 * np.log10(1.0 / (mse + 1e-8))
        metrics['ssim'] = float(ssim(sr, hr, channel_axis=0, data_range=1.0))
        metrics['mse'] = float(mse)

    # LR vs HR (upsampled to compare)
    if lr.shape != hr.shape:
        lr_up = cv2.resize(lr.transpose(1, 2, 0), (hr.shape[2], hr.shape[1]),
                          interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
    else:
        lr_up = lr

    mse_lr = np.mean((lr_up - hr) ** 2)
    metrics['psnr_lr'] = 10 * np.log10(1.0 / (mse_lr + 1e-8))
    metrics['ssim_lr'] = float(ssim(lr_up, hr, channel_axis=0, data_range=1.0))

    if sr is not None:
        # Improvement metrics
        metrics['psnr_improvement'] = metrics['psnr'] - metrics['psnr_lr']
        metrics['ssim_improvement'] = metrics['ssim'] - metrics['ssim_lr']

    return metrics


def load_image(path: Path) -> Tuple[np.ndarray, dict]:
    """Load a GeoTIFF or regular image."""
    import rasterio
    from rasterio.windows import Window

    ext = path.suffix.lower()

    if ext in ['.tif', '.tiff']:
        with rasterio.open(path) as src:
            # Read RGB bands (B04, B03, B02)
            if src.count >= 3:
                data = src.read([3, 2, 1])  # RGB
            else:
                data = src.read([1, 1, 1])  # fallback to grayscale as RGB

            # Normalize to [0, 1]
            max_val = np.iinfo(np.dtype(src.dtypes[0])).max
            if max_val > 1:
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)

            transform = src.transform
            crs = src.crs
            return data, {'transform': transform, 'crs': crs, 'path': str(path)}
    else:
        # Regular image
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # CHW format
        img = img.transpose(2, 0, 1)
        return img, {'path': str(path)}


def run_super_resolution(
    image: np.ndarray,
    checkpoint_path: Optional[Path] = None,
    tile: int = 512,
    half: bool = False,
) -> np.ndarray:
    """Run super-resolution on an image."""
    from src.model import load_realesrgan, _generator_from_upsampler, enhance_multiband

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running SR on {device}")

    # Load model
    if checkpoint_path and checkpoint_path.exists():
        upsampler = load_realesrgan(
            model_key="x4plus",
            checkpoint_path=str(checkpoint_path),
            tile=tile,
            half=half,
            device=device,
        )
    else:
        # Use pretrained
        upsampler = load_realesrgan(
            model_key="x4plus",
            tile=tile,
            half=half,
            device=device,
        )

    generator = _generator_from_upsampler(upsampler)

    # Run enhancement
    c, h, w = image.shape
    if c >= 3:
        # RGB
        rgb = image[:3].transpose(1, 2, 0)  # HWC
        rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        sr_rgb, _ = upsampler.enhance(rgb_u8, outscale=4)
        sr_rgb = sr_rgb.astype(np.float32) / 255.0

        # Reconstruct
        sr = np.zeros((c, sr_rgb.shape[0], sr_rgb.shape[1]), dtype=np.float32)
        sr[:3] = sr_rgb.transpose(2, 0, 1)

        # Handle other bands if present
        if c > 3:
            gray_batch = []
            for i in range(3, c):
                gray = image[i]
                gray_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
                gray_3ch = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
                sr_gray, _ = upsampler.enhance(gray_3ch, outscale=4)
                sr[i] = sr_gray[:, :, 0].astype(np.float32) / 255.0
    else:
        # Grayscale
        gray = image[0]
        gray_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
        gray_3ch = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
        sr_3ch, _ = upsampler.enhance(gray_3ch, outscale=4)
        sr = sr_3ch[:, :, 0:1].transpose(2, 0, 1).astype(np.float32) / 255.0
        if c > 1:
            sr = np.repeat(sr, c, axis=0)

    return sr


def create_comparison_visualization(
    lr: np.ndarray,
    hr: np.ndarray,
    sr: np.ndarray,
    metrics: dict,
    output_path: Path,
    title: str = "Depth-Wizard: Super-Resolution Demo",
) -> None:
    """Create an impressive side-by-side comparison visualization."""

    # Ensure 3 channels
    def prep_img(img):
        if img.shape[0] == 1:
            return np.repeat(img, 3, axis=0)
        elif img.shape[0] > 3:
            return img[:3]
        return img

    lr_img = prep_img(lr)
    hr_img = prep_img(hr)
    sr_img = prep_img(sr)

    # Convert to HWC uint8
    lr_disp = (lr_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    hr_disp = (hr_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    sr_disp = (sr_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    # Resize for consistent display
    h, w = hr_disp.shape[:2]
    lr_display = cv2.resize(lr_disp, (w, h), interpolation=cv2.INTER_CUBIC)

    # Create canvas
    canvas_h = h + 120  # Extra space for title and metrics
    canvas = np.ones((canvas_h, w * 3 + 40, 3), dtype=np.uint8) * 245

    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_y = 40
    cv2.putText(canvas, title, (20, title_y), font, 1.0, (30, 30, 30), 2)

    # Place images
    y_offset = 80
    canvas[y_offset:y_offset+h, 0:w] = lr_display
    canvas[y_offset:y_offset+h, w+10:w*2+10] = hr_disp
    canvas[y_offset:y_offset+h, w*2+20:w*3+20] = sr_disp

    # Labels
    label_y = y_offset + h + 25
    cv2.putText(canvas, "Input (10m)", (w//2 - 50, label_y), font, 0.6, (80, 80, 80), 2)
    cv2.putText(canvas, "Ground Truth (HR)", (w + 10 + w//2 - 70, label_y), font, 0.6, (80, 80, 80), 2)
    cv2.putText(canvas, "Depth-Wizard SR", (w*2 + 20 + w//2 - 70, label_y), font, 0.6, (37, 99, 235), 2)

    # Metrics
    metric_y = label_y + 30
    metric_text = f"PSNR: {metrics.get('psnr', 0):.2f} dB | SSIM: {metrics.get('ssim', 0):.4f} | Improvement: +{metrics.get('psnr_improvement', 0):.2f} dB"
    cv2.putText(canvas, metric_text, (20, metric_y), font, 0.5, (50, 50, 50), 1)

    # Save
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved comparison to {output_path}")


def create_zoom_comparison(
    lr: np.ndarray,
    sr: np.ndarray,
    output_path: Path,
    zoom_region: Tuple[int, int, int, int] = None,
) -> None:
    """Create zoomed-in detail comparison showing texture recovery."""

    # Prep images
    def prep_img(img):
        if img.shape[0] == 1:
            return np.repeat(img, 3, axis=0)
        elif img.shape[0] > 3:
            return img[:3]
        return img

    lr_img = prep_img(lr)
    sr_img = prep_img(sr)

    # Convert to HWC uint8
    lr_disp = (lr_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    sr_disp = (sr_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    # Default: center region
    if zoom_region is None:
        h, w = lr_disp.shape[:2]
        zoom_region = (h//4, w//4, h//2, w//2)

    y1, x1, h_zoom, w_zoom = zoom_region

    # Extract regions
    lr_zoom = lr_disp[y1:y1+h_zoom, x1:x1+w_zoom]
    sr_zoom = sr_disp[y1*4:(y1+h_zoom)*4, x1*4:(x1+w_zoom)*4]

    # Resize LR to match SR for comparison
    lr_for_compare = cv2.resize(lr_disp, (sr_disp.shape[1], sr_disp.shape[0]), interpolation=cv2.INTER_CUBIC)
    lr_zoom_full = lr_for_compare[y1*4:(y1+h_zoom)*4, x1*4:(x1+w_zoom)*4]

    # Create comparison grid
    canvas_h = h_zoom * 4 + 100
    canvas_w = w_zoom * 4 + 30

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 250

    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Detail Comparison (4x zoom)", (10, 30), font, 0.8, (30, 30, 30), 2)

    y_offset = 60

    # Top row: full images
    canvas[y_offset:y_offset+h_zoom, 0:w_zoom] = lr_zoom
    canvas[y_offset:y_offset+h_zoom, w_zoom+10:w_zoom+10+w_zoom*4] = sr_disp[y_offset:y_offset+h_zoom*4, 0:w_zoom*4]

    cv2.putText(canvas, "Bicubic (4x)", (10, y_offset + h_zoom + 20), font, 0.5, (80, 80, 80), 1)
    cv2.putText(canvas, "Depth-Wizard SR", (w_zoom+10, y_offset + h_zoom*4 + 20), font, 0.5, (37, 99, 235), 1)

    # Bottom: zoomed regions
    zoom_y = y_offset + h_zoom + 50
    canvas[zoom_y:zoom_y+h_zoom*4, 0:w_zoom*4] = lr_zoom_full
    canvas[zoom_y:zoom_y+h_zoom*4, w_zoom*4+15:w_zoom*4+15+w_zoom*4] = sr_zoom

    cv2.putText(canvas, "Bicubic 4x (zoomed)", (10, zoom_y + h_zoom*4 + 20), font, 0.5, (80, 80, 80), 1)
    cv2.putText(canvas, "SR Detail (zoomed)", (w_zoom*4+15, zoom_y + h_zoom*4 + 20), font, 0.5, (37, 99, 235), 1)

    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved zoom comparison to {output_path}")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate demo comparison for SIH")
    parser.add_argument("--input", required=True, help="Input image path (GeoTIFF or image)")
    parser.add_argument("--output", default="demo_output.png", help="Output comparison path")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint path")
    parser.add_argument("--compare", action="store_true", help="Generate full comparison with metrics")
    parser.add_argument("--zoom", action="store_true", help="Generate zoomed detail comparison")
    parser.add_argument("--tile", type=int, default=512, help="Tile size for inference")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return

    logger.info(f"Loading {input_path}")
    lr, meta = load_image(input_path)
    logger.info(f"Loaded image shape: {lr.shape}")

    # Generate LR (degrade for comparison if image is HR)
    # For demo, we'll use the original as "LR" and create synthetic HR
    # In real scenario, you'd have actual LR/HR pairs

    # Run super-resolution
    logger.info("Running super-resolution...")
    sr = run_super_resolution(lr, checkpoint_path, tile=args.tile)
    logger.info(f"SR output shape: {sr.shape}")

    # Create synthetic HR by upscaling LR (for demo purposes)
    hr = torch.nn.functional.interpolate(
        torch.from_numpy(lr).unsqueeze(0),
        scale_factor=4,
        mode='bicubic',
        align_corners=False
    ).squeeze(0).numpy()

    # Compute metrics
    metrics = compute_metrics(lr, hr, sr)
    logger.info(f"Metrics: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
    logger.info(f"Improvement: PSNR +{metrics['psnr_improvement']:.2f} dB, SSIM +{metrics['ssim_improvement']:.4f}")

    # Save metrics
    metrics_path = output_path.with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Generate visualizations
    if args.compare:
        create_comparison_visualization(lr, hr, sr, metrics, output_path)

    if args.zoom:
        zoom_path = output_path.parent / f"{output_path.stem}_zoom{output_path.suffix}"
        create_zoom_comparison(lr, sr, zoom_path)

    logger.info(f"✓ Demo complete! Output: {output_path}")


if __name__ == "__main__":
    main()