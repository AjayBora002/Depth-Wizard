"""
uncertainty.py
───────────────
Uncertainty quantification for the SR model output.

Method: Test-Time Augmentation (TTA) Ensemble
─────────────────────────────────────────────
We apply 8 geometric augmentations (identity + 3 rotations + 4 flips),
run SR on each, undo the augmentation, then measure per-pixel variance across
the N outputs.

  uncertainty_map[h,w] = var_across_augmentations(sr_outputs[h,w])

This gives a principled estimate of model confidence:
  - Low variance  → model is consistent → high confidence
  - High variance → model disagrees across augmentations → low confidence

The variance heatmap is meaningful because satellite texture patterns are
approximately rotationally consistent, so inconsistency across augmentations
indicates the model is "guessing" fine detail.

Alternative (MC Dropout):
  If the model has Dropout layers that can be activated at inference time,
  N stochastic forward passes can be used instead. A fallback implementation
  is provided but TTA is the default because RRDBNet has no dropout by default.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _augment_chw(img: np.ndarray, aug_id: int) -> np.ndarray:
    """Apply augmentation aug_id ∈ [0,7] to (C, H, W) float32 array."""
    _, h, w = img.shape
    t = img.transpose(1, 2, 0)   # (H, W, C) for cv2

    if aug_id == 0:
        out = t
    elif aug_id == 1:                            # rot90
        out = np.rot90(t, 1)
    elif aug_id == 2:                            # rot180
        out = np.rot90(t, 2)
    elif aug_id == 3:                            # rot270
        out = np.rot90(t, 3)
    elif aug_id == 4:                            # flip-LR
        out = np.fliplr(t)
    elif aug_id == 5:                            # flip-UD
        out = np.flipud(t)
    elif aug_id == 6:                            # rot90 + flip-LR
        out = np.fliplr(np.rot90(t, 1))
    else:                                        # rot90 + flip-UD
        out = np.flipud(np.rot90(t, 1))

    return out.transpose(2, 0, 1)   # back to (C, H, W)


def _deaugment_chw(img: np.ndarray, aug_id: int) -> np.ndarray:
    """
    Reverse augmentation aug_id from (C, H, W) float32 array.

    Inverse composition rule: if forward = A then B, inverse = B⁻¹ then A⁻¹.
    For aug_id 6: forward = rot90(1) then fliplr → inverse = fliplr then rot90(-1)
    For aug_id 7: forward = rot90(1) then flipud → inverse = flipud then rot90(-1)
    """
    _, h, w = img.shape
    t = img.transpose(1, 2, 0)

    if aug_id == 0:
        out = t
    elif aug_id == 1:
        out = np.rot90(t, -1)
    elif aug_id == 2:
        out = np.rot90(t, -2)
    elif aug_id == 3:
        out = np.rot90(t, -3)
    elif aug_id == 4:
        out = np.fliplr(t)
    elif aug_id == 5:
        out = np.flipud(t)
    elif aug_id == 6:
        # BUG-4 FIX: forward was rot90(1) THEN fliplr.
        # Correct inverse: fliplr THEN rot90(-1)  (reverse op order).
        out = np.rot90(np.fliplr(t), -1)
    else:
        # BUG-4 FIX: forward was rot90(1) THEN flipud.
        # Correct inverse: flipud THEN rot90(-1)  (reverse op order).
        out = np.rot90(np.flipud(t), -1)

    return out.transpose(2, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# TTA Ensemble
# ──────────────────────────────────────────────────────────────────────────────

def tta_ensemble(
    lr_patch: np.ndarray,
    enhance_fn: Callable[[np.ndarray], np.ndarray],
    n_augmentations: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TTA ensemble and return (mean_sr, uncertainty_map).

    Parameters
    ----------
    lr_patch     : float32 (C, H, W) in [0, 1] — the low-resolution input
    enhance_fn   : function that takes a (C, H, W) float32 array and returns
                   a (C, H*scale, W*scale) float32 array.
                   You can wrap model.enhance_multiband() as this function.
    n_augmentations: number of augmentations to use (max 8)

    Returns
    -------
    mean_sr       : float32 (C, H_sr, W_sr) — mean SR output across augmentations
    uncertainty   : float32 (H_sr, W_sr)    — per-pixel std dev across augmentations
    """
    n_aug = min(n_augmentations, 8)
    outputs: List[np.ndarray] = []

    for aug_id in range(n_aug):
        aug_lr = _augment_chw(lr_patch, aug_id)
        try:
            sr_aug = enhance_fn(aug_lr)           # (C, H_sr, W_sr)
        except Exception as exc:
            logger.warning("TTA augmentation %d failed: %s — skipping", aug_id, exc)
            continue
        sr_canonical = _deaugment_chw(sr_aug, aug_id)
        outputs.append(sr_canonical)

    if not outputs:
        raise RuntimeError("All TTA augmentations failed")

    stack = np.stack(outputs, axis=0)             # (N, C, H_sr, W_sr)
    mean_sr = np.mean(stack, axis=0)              # (C, H_sr, W_sr)

    # Per-pixel std across bands and augmentations
    std_map = np.std(stack, axis=0)               # (C, H_sr, W_sr)
    uncertainty = np.mean(std_map, axis=0)        # (H_sr, W_sr) — mean across bands

    return mean_sr.astype(np.float32), uncertainty.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Batched TTA Ensemble (fast path — requires bare generator nn.Module)
# ──────────────────────────────────────────────────────────────────────────────

def tta_ensemble_batched(
    lr_patch: np.ndarray,
    generator,              # torch.nn.Module — the raw RRDBNet generator
    device,                 # torch.device
    n_augmentations: int = 8,
    half: bool = False,
    rgb_band_indices: Tuple[int, ...] = (0, 1, 2),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched TTA ensemble: run all N augmentations in batched forward pass(es).

    Supports multi-band Sentinel-2 inputs (e.g. 3, 4, 9, 12 bands):
      1. Identifies RGB bands from rgb_band_indices and non-RGB (grayscale) bands.
      2. For 3-channel RGB: augments and batches all N augmentations into (N, 3, H, W)
         and runs a single generator.forward() pass.
      3. For non-RGB bands: replicates each to 3 channels, batches augmentations,
         and runs generator.forward() in chunks of n_aug, extracting channel 0.
      4. Deaugments each band group consistently and aggregates across all bands
         into mean_sr (C, H_sr, W_sr) and per-pixel uncertainty (H_sr, W_sr).

    Parameters
    ----------
    lr_patch        : float32 (C, H, W) in [0, 1] — the low-resolution input
    generator       : torch.nn.Module (RRDBNet) — obtained from upsampler.model
    device          : torch.device to run inference on
    n_augmentations : number of augmentations to use (max 8)
    half            : if True, use float16 (requires CUDA)
    rgb_band_indices: indices of (R, G, B) bands within patch

    Returns
    -------
    mean_sr     : float32 (C, H_sr, W_sr) — mean SR output across augmentations
    uncertainty : float32 (H_sr, W_sr)    — per-pixel std dev across augmentations
    """
    import torch

    c, h, w = lr_patch.shape
    n_aug = min(n_augmentations, 8)
    valid_aug_ids = list(range(n_aug))

    # Identify RGB vs grayscale bands
    valid_rgb = [i for i in rgb_band_indices if i < c]
    use_rgb = (len(valid_rgb) == 3)
    gray_indices = [i for i in range(c) if (not use_rgb or i not in valid_rgb)]

    was_training = generator.training
    generator.eval()
    t_fwd = time.time()

    try:
        deaug_rgb = None
        h_sr, w_sr = None, None

        # ── 1. Process RGB bands if available ─────────────────────────────────
        if use_rgb:
            rgb_patch = lr_patch[list(valid_rgb)]  # (3, H, W)
            aug_rgb_list = [_augment_chw(rgb_patch, aug_id) for aug_id in valid_aug_ids]
            batch_rgb_np = np.stack(aug_rgb_list, axis=0).astype(np.float32)  # (N_aug, 3, H, W)
            batch_rgb_t = torch.from_numpy(batch_rgb_np).to(device)
            if half and device.type == "cuda":
                batch_rgb_t = batch_rgb_t.half()

            with torch.no_grad():
                sr_rgb_t = generator(batch_rgb_t)  # (N_aug, 3, H_sr, W_sr)

            sr_rgb_np = sr_rgb_t.float().cpu().numpy().clip(0, 1)
            h_sr, w_sr = sr_rgb_np.shape[2], sr_rgb_np.shape[3]

            # Deaugment each RGB output
            deaug_rgb = np.stack([
                _deaugment_chw(sr_rgb_np[i], aug_id)
                for i, aug_id in enumerate(valid_aug_ids)
            ], axis=0)  # (N_aug, 3, H_sr, W_sr)

        # ── 2. Process non-RGB (grayscale) bands ──────────────────────────────
        all_deaug_gray = {}
        if gray_indices:
            all_gray_aug = []
            for g_idx in gray_indices:
                g_patch = lr_patch[g_idx]  # (H, W)
                g_3ch = np.stack([g_patch, g_patch, g_patch], axis=0)  # (3, H, W)
                for aug_id in valid_aug_ids:
                    all_gray_aug.append(_augment_chw(g_3ch, aug_id))

            # Run in batches of n_aug to control VRAM usage
            batch_size = n_aug
            sr_gray_parts = []
            for start_idx in range(0, len(all_gray_aug), batch_size):
                chunk_np = np.stack(all_gray_aug[start_idx:start_idx + batch_size], axis=0).astype(np.float32)
                chunk_t = torch.from_numpy(chunk_np).to(device)
                if half and device.type == "cuda":
                    chunk_t = chunk_t.half()
                with torch.no_grad():
                    sr_chunk_t = generator(chunk_t)
                sr_gray_parts.append(sr_chunk_t[:, 0:1].float().cpu().numpy().clip(0, 1))

            all_sr_gray = np.concatenate(sr_gray_parts, axis=0)  # (len(gray_indices) * n_aug, 1, H_sr, W_sr)
            if h_sr is None:
                h_sr, w_sr = all_sr_gray.shape[2], all_sr_gray.shape[3]

            for g_pos, band_idx in enumerate(gray_indices):
                band_sr_aug = all_sr_gray[g_pos * n_aug : (g_pos + 1) * n_aug]  # (N_aug, 1, H_sr, W_sr)
                deaug_g = np.stack([
                    _deaugment_chw(band_sr_aug[i], aug_id)
                    for i, aug_id in enumerate(valid_aug_ids)
                ], axis=0)  # (N_aug, 1, H_sr, W_sr)
                all_deaug_gray[band_idx] = deaug_g[:, 0, :, :]

    finally:
        if was_training:
            generator.train()

    logger.debug("TTA batched forward (%d augs, %d bands) took %.3fs", n_aug, c, time.time() - t_fwd)

    # ── 3. Assemble full output across all C channels ─────────────────────────
    full_aug_outputs = np.zeros((n_aug, c, h_sr, w_sr), dtype=np.float32)

    if use_rgb and deaug_rgb is not None:
        for out_pos, band_idx in enumerate(valid_rgb):
            full_aug_outputs[:, band_idx, :, :] = deaug_rgb[:, out_pos, :, :]

    for band_idx, deaug_band in all_deaug_gray.items():
        full_aug_outputs[:, band_idx, :, :] = deaug_band

    # ── 4. Aggregate mean and uncertainty ─────────────────────────────────────
    mean_sr = np.mean(full_aug_outputs, axis=0)        # (C, H_sr, W_sr)
    std_map = np.std(full_aug_outputs, axis=0)         # (C, H_sr, W_sr)
    uncertainty = np.mean(std_map, axis=0)             # (H_sr, W_sr)

    return mean_sr.astype(np.float32), uncertainty.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty visualisation
# ──────────────────────────────────────────────────────────────────────────────

def uncertainty_to_heatmap(
    uncertainty: np.ndarray,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """
    Convert a float32 (H, W) uncertainty map to a uint8 (H, W, 3) RGB heatmap.

    Normalises to [0, 255] using percentile stretch for visual clarity.
    Higher intensity = higher uncertainty (less confident SR detail).
    """
    unc = uncertainty.astype(np.float32)
    lo, hi = np.percentile(unc, 2), np.percentile(unc, 98)
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    unc_norm = np.clip((unc - lo) / (hi - lo), 0, 1)
    unc_u8 = (unc_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(unc_u8, colormap)   # (H, W, 3) BGR
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # RGB


def save_uncertainty_heatmap(
    uncertainty: np.ndarray,
    out_path: str | Path,
    sr_image_rgb: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> None:
    """
    Save an uncertainty heatmap (optionally blended with the SR image).

    Parameters
    ----------
    uncertainty    : float32 (H, W) uncertainty map
    out_path       : output PNG path
    sr_image_rgb   : optional uint8 (H, W, 3) SR image for overlay blend
    alpha          : blend ratio (0 = pure heatmap, 1 = pure SR image)
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = uncertainty_to_heatmap(uncertainty)

    if sr_image_rgb is not None:
        # Resize heatmap to match SR image if needed
        if heatmap.shape[:2] != sr_image_rgb.shape[:2]:
            heatmap = cv2.resize(heatmap, (sr_image_rgb.shape[1], sr_image_rgb.shape[0]))
        blended = cv2.addWeighted(sr_image_rgb, alpha, heatmap, 1 - alpha, 0)
        canvas = np.hstack([sr_image_rgb, blended, heatmap])
    else:
        canvas = heatmap

    # Save with matplotlib (adds colorbar legend)
    fig, axes = plt.subplots(1, 1 if sr_image_rgb is None else 3, figsize=(12 if sr_image_rgb is not None else 5, 4))
    if sr_image_rgb is None:
        axes = [axes]

    if sr_image_rgb is not None:
        axes[0].imshow(sr_image_rgb)
        axes[0].set_title("SR Output", fontsize=10)
        axes[0].axis("off")
        axes[1].imshow(blended)
        axes[1].set_title("SR + Uncertainty Overlay", fontsize=10)
        axes[1].axis("off")
        im = axes[2].imshow(uncertainty, cmap="viridis")
        axes[2].set_title("Uncertainty (per-pixel std)", fontsize=10)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Std Dev")
    else:
        im = axes[0].imshow(uncertainty, cmap="viridis")
        axes[0].set_title("Uncertainty Map", fontsize=10)
        axes[0].axis("off")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Std Dev")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Uncertainty heatmap saved → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# MC Dropout fallback (optional)
# ──────────────────────────────────────────────────────────────────────────────

def mc_dropout_uncertainty(
    lr_patch: np.ndarray,
    generator_fn: Callable[[np.ndarray], np.ndarray],
    n_passes: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout uncertainty estimate (fallback / alternative to TTA).

    Requires the model to have Dropout layers that activate in train() mode.
    RRDBNet by default has no Dropout — use TTA instead unless you've added
    Dropout layers via monkeypatching.

    Parameters
    ----------
    generator_fn : function called with lr_patch (C,H,W) → sr (C,H_sr,W_sr)
                   Must be called with model.train() active.

    Returns
    -------
    mean_sr, uncertainty  (same semantics as tta_ensemble)
    """
    outputs = []
    for _ in range(n_passes):
        sr = generator_fn(lr_patch)
        outputs.append(sr)

    stack = np.stack(outputs, axis=0)     # (N, C, H_sr, W_sr)
    mean_sr = np.mean(stack, axis=0)
    uncertainty = np.mean(np.std(stack, axis=0), axis=0)
    return mean_sr.astype(np.float32), uncertainty.astype(np.float32)
