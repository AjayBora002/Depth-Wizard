"""
metrics.py
───────────
Objective image quality metrics for super-resolution evaluation:

  - PSNR  : Peak Signal-to-Noise Ratio (dB)      — via scikit-image
  - SSIM  : Structural Similarity Index           — via scikit-image
  - SAM   : Spectral Angle Mapper (degrees)       — custom implementation

SAM reference:
  Yuhas, R.H., Goetz, A.F.H., and Boardman, J.W. (1992).
  Discrimination Among Semi-Arid Landscape Endmembers Using the
  Spectral Angle Mapper (SAM) Algorithm. AVIRIS Workshop.

  per-pixel SAM(u, v) = arccos( dot(u,v) / (||u|| * ||v||) )
  scene SAM = mean over all pixels (in degrees)

All functions accept:
  pred / gt : float32 numpy arrays (C, H, W) or (H, W, C) or (H, W) in [0,1]
  band_axis  : which axis is the spectral / band axis (default 0)

Returns a scalar float or dict of floats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_chw(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (C, H, W). If 2D (H,W), add a channel dim."""
    if arr.ndim == 2:
        return arr[np.newaxis]          # (1, H, W)
    if arr.ndim == 3 and arr.shape[-1] <= 4 and arr.shape[0] > 4:
        return arr.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
    return arr                          # already (C, H, W)


def _check_shapes(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape; got {pred.shape} vs {gt.shape}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# PSNR
# ──────────────────────────────────────────────────────────────────────────────

def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """
    Compute PSNR between pred and gt.

    Parameters
    ----------
    pred, gt   : float32 arrays of identical shape, values in [0, data_range]
    data_range : dynamic range (default 1.0 for [0,1]-normalised images)

    Returns
    -------
    float  (dB); +inf when pred == gt exactly
    """
    from skimage.metrics import peak_signal_noise_ratio

    pred = _to_chw(pred.astype(np.float32))
    gt = _to_chw(gt.astype(np.float32))
    _check_shapes(pred, gt)

    # scikit-image PSNR on flattened multi-band (treat entire array as image)
    return float(
        peak_signal_noise_ratio(gt, pred, data_range=data_range)
    )


# ──────────────────────────────────────────────────────────────────────────────
# SSIM
# ──────────────────────────────────────────────────────────────────────────────

def ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """
    Compute mean SSIM across all bands.

    Returns
    -------
    float in [-1, 1]; 1.0 = perfect match
    """
    from skimage.metrics import structural_similarity

    pred = _to_chw(pred.astype(np.float32))
    gt = _to_chw(gt.astype(np.float32))
    _check_shapes(pred, gt)

    scores = []
    for c in range(pred.shape[0]):
        s = structural_similarity(
            gt[c], pred[c], data_range=data_range
        )
        scores.append(s)
    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# SAM — Spectral Angle Mapper
# ──────────────────────────────────────────────────────────────────────────────

def sam(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """
    Spectral Angle Mapper — mean angle (degrees) between spectral vectors.

    Works on single-band images too (returns 0.0 trivially — SAM is only
    meaningful for multi-band comparisons).

    Parameters
    ----------
    pred, gt : float32 (C, H, W) arrays in [0, 1]
    eps      : small constant to avoid division by zero

    Returns
    -------
    float  (degrees); 0.0 = perfect spectral match
    """
    pred = _to_chw(pred.astype(np.float32))
    gt = _to_chw(gt.astype(np.float32))
    _check_shapes(pred, gt)

    # Single-band images have no spectral angle
    if pred.shape[0] == 1:
        return 0.0

    if np.array_equal(pred, gt):
        return 0.0

    # Use float64 for intermediate dot and norms to prevent numerical precision
    # loss in arccos near cos_angle = 1.0
    pred_64 = pred.astype(np.float64)
    gt_64 = gt.astype(np.float64)

    dot = np.sum(pred_64 * gt_64, axis=0)                   # (H, W)
    norm_pred = np.linalg.norm(pred_64, axis=0)             # (H, W)
    norm_gt = np.linalg.norm(gt_64, axis=0)                 # (H, W)
    denom = norm_pred * norm_gt                             # (H, W)

    valid_mask = denom > eps
    if not np.any(valid_mask):
        return 0.0

    cos_angle = np.clip(dot[valid_mask] / denom[valid_mask], -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(np.mean(angle_deg))


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# SRE — Signal to Reconstruction Error Ratio (dB)
# ──────────────────────────────────────────────────────────────────────────────

def sre(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-10) -> float:
    """
    Signal to Reconstruction Error ratio in dB.
    Standard remote-sensing metric:
        SRE(dB) = 10 * log10( (mean(gt)**2) / (mean((pred - gt)**2) + eps) )

    Parameters
    ----------
    pred, gt : float32 arrays of identical shape
    eps      : small epsilon to avoid divide-by-zero

    Returns
    -------
    float (dB); higher is better. Returns +inf if pred == gt.
    """
    pred = _to_chw(pred.astype(np.float32))
    gt = _to_chw(gt.astype(np.float32))
    _check_shapes(pred, gt)

    if np.array_equal(pred, gt):
        return float("inf")

    mse = float(np.mean((pred - gt) ** 2))
    if mse <= eps:
        return float("inf")

    mean_gt = float(np.mean(gt))
    if abs(mean_gt) <= eps:
        return 0.0

    ratio = (mean_gt ** 2) / mse
    return float(10.0 * np.log10(max(ratio, eps)))


# ──────────────────────────────────────────────────────────────────────────────
# ERGAS — Relative Dimensionless Global Error of Synthesis
# ──────────────────────────────────────────────────────────────────────────────

def ergas(pred: np.ndarray, gt: np.ndarray, scale: float = 4.0, eps: float = 1e-10) -> float:
    """
    Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).
    Widely used in multi-spectral satellite super-resolution (DSen2, Wald et al.).
        ERGAS = 100/scale * sqrt( 1/B * sum_{b=1}^B (RMSE_b / (mean(gt_b) + eps))**2 )

    Parameters
    ----------
    pred, gt : float32 arrays of identical shape
    scale    : super-resolution scale factor (e.g. 4 for 10m -> 2.5m)
    eps      : small epsilon

    Returns
    -------
    float; lower is better (0.0 = perfect reconstruction).
    """
    pred = _to_chw(pred.astype(np.float32))
    gt = _to_chw(gt.astype(np.float32))
    _check_shapes(pred, gt)

    if np.array_equal(pred, gt):
        return 0.0

    c = pred.shape[0]
    band_ratios_sq = []
    for b in range(c):
        rmse_b = float(np.sqrt(np.mean((pred[b] - gt[b]) ** 2)))
        mean_b = float(np.mean(gt[b]))
        if abs(mean_b) <= eps:
            band_ratios_sq.append(0.0)
        else:
            band_ratios_sq.append((rmse_b / mean_b) ** 2)

    mean_sq = float(np.mean(band_ratios_sq))
    return float((100.0 / scale) * np.sqrt(mean_sq))


# ──────────────────────────────────────────────────────────────────────────────
# UIQ — Universal Image Quality Index
# ──────────────────────────────────────────────────────────────────────────────

def uiq(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-10) -> float:
    """
    Universal Image Quality Index (UIQ, Wang & Bovik 2002).
    Evaluates loss of correlation, luminance distortion, and contrast distortion:
        Q = (4 * cov_xy * mean_x * mean_y) / ((var_x + var_y) * (mean_x**2 + mean_y**2) + eps)

    Averaged across all spectral bands.

    Parameters
    ----------
    pred, gt : float32 arrays of identical shape

    Returns
    -------
    float in [-1, 1]; 1.0 = identical images.
    """
    pred = _to_chw(pred.astype(np.float32))
    gt = _to_chw(gt.astype(np.float32))
    _check_shapes(pred, gt)

    if np.array_equal(pred, gt):
        return 1.0

    scores = []
    c = pred.shape[0]
    for b in range(c):
        x = gt[b].astype(np.float64)
        y = pred[b].astype(np.float64)

        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))

        var_x = float(np.var(x))
        var_y = float(np.var(y))
        cov_xy = float(np.mean((x - mean_x) * (y - mean_y)))

        denom = (var_x + var_y) * (mean_x ** 2 + mean_y ** 2)
        if denom <= eps:
            scores.append(1.0 if abs(mean_x - mean_y) <= eps else 0.0)
        else:
            q = (4.0 * cov_xy * mean_x * mean_y) / denom
            scores.append(float(np.clip(q, -1.0, 1.0)))

    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_pair(pred: np.ndarray, gt: np.ndarray, scale: float = 4.0) -> Dict[str, float]:
    """Compute PSNR, SSIM, SAM, SRE, ERGAS, and UIQ for a single (pred, gt) pair."""
    return {
        "psnr": psnr(pred, gt),
        "ssim": ssim(pred, gt),
        "sam_deg": sam(pred, gt),
        "sre": sre(pred, gt),
        "ergas": ergas(pred, gt, scale=scale),
        "uiq": uiq(pred, gt),
    }


def evaluate_directory(
    pred_dir: Path | str,
    gt_dir: Path | str,
    ext: str = ".npy",
    scale: float = 4.0,
) -> Dict[str, float]:
    """
    Compute mean PSNR / SSIM / SAM / SRE / ERGAS / UIQ over all matching
    (pred, gt) file pairs in pred_dir and gt_dir.

    Files are matched by stem (filename without extension).

    Returns dict with mean metrics and n_samples.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pred_files = sorted(pred_dir.glob(f"*{ext}"))
    if not pred_files:
        logger.warning("No prediction files found in %s", pred_dir)
        return {
            "mean_psnr": float("nan"),
            "mean_ssim": float("nan"),
            "mean_sam": float("nan"),
            "mean_sre": float("nan"),
            "mean_ergas": float("nan"),
            "mean_uiq": float("nan"),
            "n_samples": 0,
        }

    psnr_vals, ssim_vals, sam_vals = [], [], []
    sre_vals, ergas_vals, uiq_vals = [], [], []
    missing = 0

    for pf in pred_files:
        gf = gt_dir / pf.name
        if not gf.exists():
            missing += 1
            continue
        pred = np.load(pf) if ext == ".npy" else _load_tif(pf)
        gt_arr = np.load(gf) if ext == ".npy" else _load_tif(gf)

        try:
            m = evaluate_pair(pred, gt_arr, scale=scale)
            psnr_vals.append(m["psnr"])
            ssim_vals.append(m["ssim"])
            sam_vals.append(m["sam_deg"])
            sre_vals.append(m["sre"])
            ergas_vals.append(m["ergas"])
            uiq_vals.append(m["uiq"])
        except Exception as exc:
            logger.warning("Skipping %s: %s", pf.name, exc)

    if missing:
        logger.warning("%d prediction file(s) had no matching GT", missing)

    n = len(psnr_vals)
    return {
        "mean_psnr": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
        "mean_ssim": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
        "mean_sam": float(np.mean(sam_vals)) if sam_vals else float("nan"),
        "mean_sre": float(np.mean(sre_vals)) if sre_vals else float("nan"),
        "mean_ergas": float(np.mean(ergas_vals)) if ergas_vals else float("nan"),
        "mean_uiq": float(np.mean(uiq_vals)) if uiq_vals else float("nan"),
        "n_samples": n,
    }


def _load_tif(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute SR metrics (PSNR/SSIM/SAM)")
    parser.add_argument("--pred-dir", required=True, help="Directory of SR output files (.npy)")
    parser.add_argument("--gt-dir", required=True, help="Directory of ground-truth files (.npy)")
    parser.add_argument("--out", default="data/outputs/metrics_report.json")
    parser.add_argument("--ext", default=".npy", choices=[".npy", ".tif", ".tiff"])
    args = parser.parse_args()

    results = evaluate_directory(args.pred_dir, args.gt_dir, ext=args.ext)
    print(json.dumps(results, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
