"""
model.py
─────────
Real-ESRGAN wrapper for Sentinel-2 super-resolution.

Key responsibilities:
  1. Auto-download pretrained RealESRGAN_x4plus weights (if not cached)
  2. Handle Sentinel-2 multi-band input:
       - RGB bands (B04/B03/B02) → standard 3-ch Real-ESRGAN
       - Additional bands (NIR, SWIR …) → grayscale processing per-band
  3. Return geo-referenced output (transform preserved externally via inference.py)
  4. Support fine-tuning mode (generator-only, L1 + perceptual loss)
  5. MC-Dropout / TTA mode for uncertainty estimation (see uncertainty.py)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F_t
# Compatibility patch for basicsr with modern torchvision
sys.modules["torchvision.transforms.functional_tensor"] = F_t

logger = logging.getLogger(__name__)

# ─── Weight download URLs ──────────────────────────────────────────────────────
WEIGHTS = {
    "x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
    },
    "x4plus_anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
    },
}

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# ─── Sentinel-2 band roles ─────────────────────────────────────────────────────
# Indices into a stacked S2 L2A GeoTIFF (0-based)
S2_RGB_BANDS = (2, 1, 0)   # B04, B03, B02 → R, G, B


# ──────────────────────────────────────────────────────────────────────────────
# Weight download helper
# ──────────────────────────────────────────────────────────────────────────────

def _download_weights(model_key: str = "x4plus") -> Path:
    """Download pretrained weights if not already present. Returns local path."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    entry = WEIGHTS[model_key]
    dest = CHECKPOINT_DIR / entry["filename"]
    if dest.exists():
        logger.info("Weights already cached: %s", dest)
        return dest

    logger.info("Downloading pretrained weights from %s …", entry["url"])
    start = time.time()

    def reporthook(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / max(total_size, 1)))
        if count % 50 == 0:
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(entry["url"], dest, reporthook)
    print()
    elapsed = time.time() - start
    logger.info("Downloaded %s in %.1fs", dest.name, elapsed)
    return dest


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

class StandaloneRealESRGANer:
    """Standalone drop-in replacement for RealESRGANer using pure PyTorch.
    Ensures inference works even if the external realesrgan package is not installed.
    """
    def __init__(
        self,
        scale: int = 4,
        model_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.scale = scale
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half
        self.model = model.to(self.device)
        if model_path:
            loadnet = torch.load(model_path, map_location=self.device)
            if isinstance(loadnet, dict) and "params_ema" in loadnet:
                keyname = "params_ema"
            elif isinstance(loadnet, dict) and "params" in loadnet:
                keyname = "params"
            else:
                keyname = None
            state_dict = loadnet[keyname] if keyname is not None else loadnet
            self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        if self.half and self.device.type == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def enhance(self, img: np.ndarray, outscale: Optional[int] = None):
        is_uint8 = (img.dtype == np.uint8)
        if is_uint8:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        else:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_t = img_t.unsqueeze(0).to(self.device)
        if self.half and self.device.type == "cuda":
            img_t = img_t.half()
        output = self.model(img_t)
        output = output.data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        output = output.transpose(1, 2, 0)
        if is_uint8:
            output = (output * 255.0).round().astype(np.uint8)
        return output, None


def load_realesrgan(
    model_key: str = "x4plus",
    checkpoint_path: Optional[Path | str] = None,
    scale: int = 4,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 10,
    half: bool = False,
    device: Optional[torch.device] = None,
):
    """
    Load Real-ESRGAN upsampler.

    Parameters
    ----------
    model_key        : 'x4plus' | 'x4plus_anime'
    checkpoint_path  : Path to .pth weights file (auto-downloaded if None)
    scale            : Upscale factor (must be 4 for x4plus)
    tile             : Tile size for internal tiling (0 = disabled)
    tile_pad         : Padding around tiles (pixels)
    pre_pad          : Pre-padding applied to image border
    half             : Use FP16 inference (requires CUDA)
    device           : torch.device (auto-detected if None)

    Returns
    -------
    RealESRGANer instance ready for .enhance()
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        from src.rrdbnet import RRDBNet
    try:
        from realesrgan import RealESRGANer
    except ImportError:
        RealESRGANer = StandaloneRealESRGANer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading Real-ESRGAN on %s (half=%s)", device, half)

    # Architecture definition
    if model_key == "x4plus":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
    elif model_key == "x4plus_anime":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=4,
        )
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    # Resolve weight file
    if checkpoint_path is not None:
        weight_path = Path(checkpoint_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
    else:
        weight_path = _download_weights(model_key)

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weight_path),
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        device=device,
    )
    return upsampler


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helper
# ──────────────────────────────────────────────────────────────────────────────

def _generator_from_upsampler(upsampler) -> torch.nn.Module:
    """
    Extract the bare RRDBNet generator from a RealESRGANer wrapper.

    RealESRGANer stores the generator under ``upsampler.model`` after
    loading weights.  This helper centralises that knowledge so callers
    do not need to know the internal attribute name.

    Returns
    -------
    generator : torch.nn.Module — the RRDBNet, in eval() mode, on the
                same device as the upsampler.
    """
    generator = upsampler.model
    generator.eval()
    return generator


# ──────────────────────────────────────────────────────────────────────────────
# Multi-band Sentinel-2 inference
# ──────────────────────────────────────────────────────────────────────────────

def enhance_multiband(
    upsampler,
    patch: np.ndarray,
    rgb_band_indices: Tuple[int, ...] = S2_RGB_BANDS,
    outscale: int = 4,
    generator: Optional[torch.nn.Module] = None,
    joint_spectral: bool = False,
    spectral_fusion_module: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """
    Run Real-ESRGAN on a Sentinel-2 multi-band patch.

    Strategy:
      - RGB bands (indices rgb_band_indices) → standard 3-ch Real-ESRGAN via
        upsampler.enhance() (uint8 API preserved for compatibility).
      - Remaining bands → when ``generator`` is provided, ALL non-RGB bands are
        stacked into a single (N_gray, 3, H, W) batch and processed in ONE
        generator.forward() call.  When ``generator`` is None, falls back to
        the original serial per-band upsampler.enhance() loop (backward compat).

    Parameters
    ----------
    upsampler       : RealESRGANer instance
    patch           : float32 (C, H, W) in [0, 1]
    rgb_band_indices: indices of (R, G, B) bands within patch
    outscale        : upscale factor (must match model scale)
    generator       : optional bare RRDBNet nn.Module — when provided, grayscale
                      bands are batched into one forward pass (fast path).
                      Obtain via ``_generator_from_upsampler(upsampler)``.

    Returns
    -------
    sr_patch : float32 (C, H*outscale, W*outscale) in [0, 1]
    """
    c, h, w = patch.shape
    sr_out = [None] * c

    # ── RGB upsampling (always via upsampler.enhance — uint8 API) ─────────────
    valid_rgb = [i for i in rgb_band_indices if i < c]
    if len(valid_rgb) == 3:
        rgb = patch[list(valid_rgb)].transpose(1, 2, 0)          # (H, W, 3) float32
        rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)        # uint8 for RealESRGANer
        sr_rgb, _ = upsampler.enhance(rgb_u8, outscale=outscale)  # uint8 (H*4, W*4, 3)
        sr_rgb_f32 = sr_rgb.astype(np.float32) / 255.0            # back to [0,1]
        for out_idx, band_idx in enumerate(valid_rgb):
            sr_out[band_idx] = sr_rgb_f32[:, :, out_idx]
    elif c >= 1:
        # Fewer than 3 bands — treat all as grayscale
        valid_rgb = list(range(min(c, 3)))

    # ── Remaining bands (grayscale) ───────────────────────────────────────────
    gray_indices = [i for i in range(c) if sr_out[i] is None]

    if not gray_indices:
        pass  # All bands already processed as RGB

    elif generator is not None and len(gray_indices) > 0:
        # ── FAST PATH: batch all grayscale bands into one forward pass ────────
        # Stack: each band replicated to 3 channels → (N_gray, 3, H, W) float32
        import torch
        gray_batch_np = np.stack(
            [np.stack([patch[i], patch[i], patch[i]], axis=0) for i in gray_indices],
            axis=0,
        ).astype(np.float32)                          # (N_gray, 3, H, W)

        device = next(generator.parameters()).device
        gray_t = torch.from_numpy(gray_batch_np).to(device)
        if next(generator.parameters()).dtype == torch.float16:
            gray_t = gray_t.half()

        was_training = generator.training
        generator.eval()
        try:
            with torch.no_grad():
                sr_gray_t = generator(gray_t)          # (N_gray, 3, H*4, W*4)
        finally:
            if was_training:
                generator.train()

        sr_gray_np = sr_gray_t.float().cpu().numpy().clip(0, 1)  # (N_gray, 3, H*4, W*4)
        for out_pos, band_idx in enumerate(gray_indices):
            sr_out[band_idx] = sr_gray_np[out_pos, 0]            # extract channel 0

    else:
        # ── SLOW PATH (fallback): serial per-band upsampler.enhance() ─────────
        for i in gray_indices:
            gray = patch[i]                                        # (H, W) float32 [0,1]
            gray_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
            gray_3ch = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)  # (H, W, 3)
            sr_3ch, _ = upsampler.enhance(gray_3ch, outscale=outscale)  # (H*4, W*4, 3) uint8
    sr_stack = np.stack(sr_out, axis=0)   # (C, H*4, W*4)

    if joint_spectral:
        try:
            import torch
            from src.spectral_fusion import JointSpectralRefiner
            if spectral_fusion_module is None:
                spectral_fusion_module = JointSpectralRefiner(num_channels=c)
                if generator is not None:
                    dev = next(generator.parameters()).device
                    spectral_fusion_module = spectral_fusion_module.to(dev)
            dev = next(spectral_fusion_module.parameters()).device
            sr_t = torch.from_numpy(sr_stack[np.newaxis, ...]).to(dev)
            was_training = spectral_fusion_module.training
            spectral_fusion_module.eval()
            with torch.no_grad():
                sr_refined_t = spectral_fusion_module(sr_t)
            if was_training:
                spectral_fusion_module.train()
            sr_stack = sr_refined_t.squeeze(0).float().cpu().numpy().clip(0, 1)
        except Exception as exc:
            logger.warning("Joint spectral fusion failed (%s), using unrefined SR", exc)

    return sr_stack


# ──────────────────────────────────────────────────────────────────────────────
# Generator-only model for fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def load_generator_for_training(
    model_key: str = "x4plus",
    checkpoint_path: Optional[Path | str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load ONLY the generator (RRDBNet) with pretrained weights.
    Used by train.py for fine-tuning (no discriminator, no RealESRGANer wrapper).
    """
    # Always use src.rrdbnet for training — it has gradient checkpointing support
    # that basicsr's RRDBNet does not have.
    from src.rrdbnet import RRDBNet

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_key == "x4plus":
        generator = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
    elif model_key == "x4plus_anime":
        generator = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=4,
        )
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    # Load weights
    if checkpoint_path is not None:
        weight_path = Path(checkpoint_path)
    else:
        weight_path = _download_weights(model_key)

    state_dict = torch.load(weight_path, map_location="cpu")
    # Handle different checkpoint formats (with/without 'params_ema' / 'params' key)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    generator.load_state_dict(state_dict, strict=True)
    generator = generator.to(device)
    logger.info("Generator loaded on %s from %s", device, weight_path.name)
    return generator


def save_generator_checkpoint(
    generator: nn.Module,
    out_path: Path | str,
    epoch: int,
    extra_info: Optional[dict] = None,
) -> None:
    """Save generator state dict as a RealESRGAN-compatible checkpoint."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params_ema": generator.state_dict(),
        "epoch": epoch,
    }
    if extra_info:
        payload.update(extra_info)
    torch.save(payload, out_path)
    logger.info("Checkpoint saved → %s (epoch %d)", out_path, epoch)
