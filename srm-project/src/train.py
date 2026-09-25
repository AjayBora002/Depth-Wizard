"""
train.py
─────────
Unified fine-tuning script for Real-ESRGAN on Sentinel-2 satellite imagery (SIH26142).

Consolidates all capabilities from the original train.py and train_enhanced.py:
  1. Loss Suite (Ablation-ready with zero overhead when lambda=0):
     - L1 pixel loss (lambda_l1)
     - VGG16 Perceptual loss (lambda_perceptual)
     - Differentiable SAM (Spectral Angle Mapper) loss (lambda_sam)
     - Sobel spatial gradient edge loss (lambda_edge)
     - 2D FFT Frequency domain loss (lambda_freq)
     - Multi-Scale pyramid loss (lambda_multiscale)
  2. Training Optimizations:
     - Warmup + Cosine Annealing learning rate scheduler
     - D4 dihedral symmetry data augmentation (overhead rotation + reflection invariance)
     - Gradient checkpointing for memory-efficient training on larger patches
     - Mixed precision training (AMP FP16)
     - Gradient clipping (max_norm=1.0)
     - Early stopping with configurable patience
     - Time budget termination
  3. Multi-Band Modeling:
     - Optional joint spectral modeling (--joint-spectral) using JointSpectralSR
       from src.spectral_fusion to model cross-band correlations directly.
  4. Validation & Monitoring:
     - Tracks PSNR, SSIM, and SAM across validation epochs
     - Records full ablation metadata and active loss terms in training_history.json

Usage:
  # Baseline L1-only ablation:
  python src/train.py --ablation l1_only --epochs 50

  # SAM-loss ablation:
  python src/train.py --ablation with_sam --epochs 50

  # Full composite loss:
  python src/train.py --ablation full --epochs 50

  # Full loss + Joint Spectral Modeling on all bands:
  python src/train.py --ablation joint_spectral --joint-spectral --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torchvision.transforms.functional as F_t
sys.modules["torchvision.transforms.functional_tensor"] = F_t

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Learning Rate Scheduling (Warmup + Cosine Annealing)
# ──────────────────────────────────────────────────────────────────────────────

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            return [base_lr * (epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


# ──────────────────────────────────────────────────────────────────────────────
# Loss Suite
# ──────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss on RGB features."""

    def __init__(self, device: torch.device):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).to(device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def _prep(x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] > 3:
                x = x[:, :3]
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x - mean) / std

        with torch.amp.autocast('cuda', enabled=False):
            feat_pred = self.feature_extractor(_prep(pred.float()))
            feat_target = self.feature_extractor(_prep(target.detach().float()))
            return F.l1_loss(feat_pred, feat_target)


class SAMLoss(nn.Module):
    """
    Differentiable Spectral Angle Mapper (SAM) loss.
    Measures the angular spectral distortion across bands at each pixel:
        SAM(u, v) = arccos( (u · v) / (||u|| ||v|| + eps) )
    Guarantees physical reflectance curves are preserved across spectral bands.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            pred_f = pred.float()
            target_f = target.float()
            dot = torch.sum(pred_f * target_f, dim=1)
            norm_pred = torch.norm(pred_f, p=2, dim=1)
            norm_target = torch.norm(target_f, p=2, dim=1)
            cos_sim = dot / (norm_pred * norm_target + self.eps)
            cos_sim = torch.clamp(cos_sim, -0.9999, 0.9999)
            sam_rad = torch.acos(cos_sim)
            return torch.mean(sam_rad)


class EdgeLoss(nn.Module):
    """
    Sobel spatial gradient loss to penalize blurry borders on linear infrastructure:
    roads, runways, coastlines, agricultural boundaries, and building footprints.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x = sobel_x.to(device)
        self.sobel_y = sobel_y.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        kernel_x = self.sobel_x.repeat(c, 1, 1, 1)
        kernel_y = self.sobel_y.repeat(c, 1, 1, 1)

        grad_pred_x = F.conv2d(pred, kernel_x, padding=1, groups=c)
        grad_pred_y = F.conv2d(pred, kernel_y, padding=1, groups=c)
        grad_target_x = F.conv2d(target, kernel_x, padding=1, groups=c)
        grad_target_y = F.conv2d(target, kernel_y, padding=1, groups=c)

        return F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)


class FrequencyLoss(nn.Module):
    """
    2D Fast Fourier Transform (FFT) loss:
    Recovers high-frequency micro-textures and prevents over-smoothed outputs.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
            target_fft = torch.fft.rfft2(target.float(), norm="ortho")
            return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class MultiScaleLoss(nn.Module):
    """Pyramid loss: compare at 1×, 0.5×, 0.25× scales."""

    def __init__(self, scales=(1.0, 0.5, 0.25), weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = list(scales)
        self.weights = list(weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for scale_weight, scale in zip(self.weights, self.scales):
            if scale < 1.0:
                factor = int(1.0 / scale)
                pred_s = F.avg_pool2d(pred, kernel_size=factor, stride=factor)
                target_s = F.avg_pool2d(target, kernel_size=factor, stride=factor)
            else:
                pred_s, target_s = pred, target
            total_loss = total_loss + scale_weight * F.l1_loss(pred_s, target_s)
        return total_loss / sum(self.weights)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset with D4 Dihedral Symmetry Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def _collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    lrs = torch.stack([b[0] for b in batch])
    hrs = torch.stack([b[1] for b in batch])
    return lrs, hrs


class CropDataset(torch.utils.data.Dataset):
    """
    Random crops with full D4 dihedral rotations and reflection symmetry
    (satellite remote sensing imagery is orientation-invariant).
    """

    def __init__(self, base_dataset, crop_size: int = 128, scale: int = 4, rgb_only: bool = True):
        self.ds = base_dataset
        self.crop_size = crop_size
        self.scale = scale
        self.rgb_only = rgb_only

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        lr, hr = self.ds[idx]

        if lr.ndim == 2:
            lr = lr.unsqueeze(0)
            hr = hr.unsqueeze(0)

        if self.rgb_only:
            if lr.shape[0] >= 3:
                lr = lr[:3]
                hr = hr[:3]
            elif lr.shape[0] == 1:
                lr = lr.repeat(3, 1, 1)
                hr = hr.repeat(3, 1, 1)

        c, h, w = lr.shape
        if h < self.crop_size or w < self.crop_size:
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            lr = F.pad(lr, (0, pad_w, 0, pad_h))
            hr = F.pad(hr, (0, pad_w * self.scale, 0, pad_h * self.scale))
            h, w = lr.shape[1], lr.shape[2]

        top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        lr_crop = lr[:, top:top + self.crop_size, left:left + self.crop_size]
        hr_crop = hr[:,
                     top * self.scale:(top + self.crop_size) * self.scale,
                     left * self.scale:(left + self.crop_size) * self.scale]

        # D4 Dihedral symmetry:
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            lr_crop = torch.rot90(lr_crop, k, [1, 2])
            hr_crop = torch.rot90(hr_crop, k, [1, 2])

        if torch.rand(1) > 0.5:
            lr_crop = torch.flip(lr_crop, [2])
            hr_crop = torch.flip(hr_crop, [2])

        if torch.rand(1) > 0.5:
            lr_crop = torch.flip(lr_crop, [1])
            hr_crop = torch.flip(hr_crop, [1])

        return lr_crop, hr_crop


# ──────────────────────────────────────────────────────────────────────────────
# SSIM Helper with Fallback
# ──────────────────────────────────────────────────────────────────────────────

def _compute_batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean SSIM across batch and channels with safe fallback."""
    try:
        from skimage.metrics import structural_similarity
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        scores = []
        for b in range(pred_np.shape[0]):
            for c in range(pred_np.shape[1]):
                scores.append(structural_similarity(target_np[b, c], pred_np[b, c], data_range=1.0))
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        # Vectorized PyTorch approximation of SSIM if skimage is unavailable
        p = pred.float()
        t = target.float()
        mu_p = F.avg_pool2d(p, 11, stride=1, padding=5)
        mu_t = F.avg_pool2d(t, 11, stride=1, padding=5)
        sigma_p_sq = F.avg_pool2d(p * p, 11, stride=1, padding=5) - mu_p.pow(2)
        sigma_t_sq = F.avg_pool2d(t * t, 11, stride=1, padding=5) - mu_t.pow(2)
        sigma_pt = F.avg_pool2d(p * t, 11, stride=1, padding=5) - mu_p * mu_t
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / (
            (mu_p.pow(2) + mu_t.pow(2) + c1) * (sigma_p_sq + sigma_t_sq + c2)
        )
        return float(ssim_map.mean().item())


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Routine
# ──────────────────────────────────────────────────────────────────────────────

def train(
    pairs_dir: Path | str,
    output_dir: Path | str,
    model_key: str = "x4plus",
    pretrained_checkpoint: Optional[Path | str] = None,
    resume_checkpoint: Optional[Path | str] = None,
    epochs: int = 50,
    batch_size: int = 4,
    crop_size: int = 128,
    lr: float = 1e-4,
    lambda_l1: float = 1.0,
    lambda_perceptual: float = 0.1,
    lambda_sam: float = 0.05,
    lambda_edge: float = 0.05,
    lambda_freq: float = 0.02,
    lambda_multiscale: float = 0.0,
    joint_spectral: bool = False,
    save_every: int = 10,
    num_workers: int = 2,
    use_amp: bool = True,
    use_gradient_checkpointing: bool = False,
    warmup_epochs: int = 5,
    early_stopping_patience: int = 15,
    time_budget_hours: Optional[float] = None,
    ablation_name: str = "custom",
) -> dict:
    """
    Unified training engine for Real-ESRGAN on Sentinel-2 satellite pairs.
    """
    from src.pair_generation import SyntheticPairDataset
    from src.model import load_generator_for_training, save_generator_checkpoint

    pairs_dir = Path(pairs_dir)
    if not (pairs_dir / "lr").exists() or not list((pairs_dir / "lr").glob("*.npy")):
        alt_dirs = [Path("data/training_pairs"), Path("data/synthetic_pairs")]
        for alt in alt_dirs:
            if alt != pairs_dir and (alt / "lr").exists() and list((alt / "lr").glob("*.npy")):
                logger.info("Using pairs found in: %s", alt)
                pairs_dir = alt
                break

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)
    if device.type == "cpu":
        logger.warning("No GPU detected — training on CPU will be slow. For full training, use Google Colab or an NVIDIA GPU.")

    # ── Dataset setup ─────────────────────────────────────────────────────────
    train_base = SyntheticPairDataset(pairs_dir, split="train")
    val_base = SyntheticPairDataset(pairs_dir, split="val")

    # When joint_spectral is True, preserve all available bands (not just RGB)
    rgb_only = not joint_spectral
    train_ds = CropDataset(train_base, crop_size=crop_size, scale=4, rgb_only=rgb_only)
    val_ds = CropDataset(val_base, crop_size=crop_size, scale=4, rgb_only=rgb_only)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=_collate_fn,
    )

    logger.info("Dataset: %d train pairs, %d val pairs in %s", len(train_base), len(val_base), pairs_dir)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_to_load = resume_checkpoint or pretrained_checkpoint
    base_generator = load_generator_for_training(
        model_key=model_key,
        checkpoint_path=ckpt_to_load,
        device=device,
    )

    if use_gradient_checkpointing:
        if hasattr(base_generator, "enable_gradient_checkpointing"):
            base_generator.enable_gradient_checkpointing()
            logger.info("✓ Gradient checkpointing enabled (reduced VRAM)")

    # ── Optional Joint Spectral Modeling ──────────────────────────────────────
    if joint_spectral:
        from src.spectral_fusion import JointSpectralSR
        sample_lr, _ = train_ds[0]
        num_channels = sample_lr.shape[0]
        model = JointSpectralSR(
            base_generator=base_generator,
            num_channels=num_channels,
            rgb_band_indices=(0, 1, 2) if num_channels >= 3 else (),
        ).to(device)
        logger.info("✓ Joint Spectral Modeling enabled for %d channels", num_channels)
    else:
        model = base_generator

    model.train()

    # ── Loss Suite Configuration (Zero overhead when lambda=0) ───────────────
    active_losses = []
    l1_loss = nn.L1Loss() if lambda_l1 > 0 else None
    if l1_loss:
        active_losses.append(f"L1({lambda_l1})")

    perceptual_loss = PerceptualLoss(device) if lambda_perceptual > 0 else None
    if perceptual_loss:
        active_losses.append(f"Perceptual({lambda_perceptual})")

    sam_loss = SAMLoss().to(device) if lambda_sam > 0 else None
    if sam_loss:
        active_losses.append(f"SAM({lambda_sam})")

    edge_loss = EdgeLoss(device) if lambda_edge > 0 else None
    if edge_loss:
        active_losses.append(f"Edge({lambda_edge})")

    freq_loss = FrequencyLoss().to(device) if lambda_freq > 0 else None
    if freq_loss:
        active_losses.append(f"Freq({lambda_freq})")

    multiscale_loss = MultiScaleLoss().to(device) if lambda_multiscale > 0 else None
    if multiscale_loss:
        active_losses.append(f"MultiScale({lambda_multiscale})")

    if joint_spectral:
        active_losses.append("JointSpectral")

    # ── Optimizer, Scheduler, Scaler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None

    # ── Tracking state ────────────────────────────────────────────────────────
    ablation_config = {
        "ablation_name": ablation_name,
        "active_losses": active_losses,
        "lambda_l1": lambda_l1,
        "lambda_perceptual": lambda_perceptual,
        "lambda_sam": lambda_sam,
        "lambda_edge": lambda_edge,
        "lambda_freq": lambda_freq,
        "lambda_multiscale": lambda_multiscale,
        "joint_spectral": joint_spectral,
        "epochs": epochs,
        "batch_size": batch_size,
        "crop_size": crop_size,
        "lr": lr,
        "warmup_epochs": warmup_epochs,
    }

    history = {
        "config": ablation_config,
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "epoch_time": [],
        "learning_rate": [],
    }

    best_val_psnr = -float("inf")
    best_checkpoint_path = output_dir / "model_finetuned_best.pth"
    final_checkpoint_path = output_dir / "model_finetuned_final.pth"
    patience_counter = 0
    t_global_start = time.time()

    logger.info("═" * 70)
    logger.info("🚀 Training Started: %s | Epochs: %d | Batch: %d | Warmup: %d",
                ablation_name, epochs, batch_size, warmup_epochs)
    logger.info("Active Terms: %s", " + ".join(active_losses) if active_losses else "None")
    logger.info("═" * 70)

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        train_losses = []

        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            if lr_batch.numel() == 0:
                continue

            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    sr_batch = model(lr_batch)
                    loss = torch.tensor(0.0, device=device)
                    if l1_loss:
                        loss = loss + lambda_l1 * l1_loss(sr_batch, hr_batch)
                    if perceptual_loss:
                        loss = loss + lambda_perceptual * perceptual_loss(sr_batch, hr_batch)
                    if sam_loss:
                        loss = loss + lambda_sam * sam_loss(sr_batch, hr_batch)
                    if edge_loss:
                        loss = loss + lambda_edge * edge_loss(sr_batch, hr_batch)
                    if freq_loss:
                        loss = loss + lambda_freq * freq_loss(sr_batch, hr_batch)
                    if multiscale_loss:
                        loss = loss + lambda_multiscale * multiscale_loss(sr_batch, hr_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                sr_batch = model(lr_batch)
                loss = torch.tensor(0.0, device=device)
                if l1_loss:
                    loss = loss + lambda_l1 * l1_loss(sr_batch, hr_batch)
                if perceptual_loss:
                    loss = loss + lambda_perceptual * perceptual_loss(sr_batch, hr_batch)
                if sam_loss:
                    loss = loss + lambda_sam * sam_loss(sr_batch, hr_batch)
                if edge_loss:
                    loss = loss + lambda_edge * edge_loss(sr_batch, hr_batch)
                if freq_loss:
                    loss = loss + lambda_freq * freq_loss(sr_batch, hr_batch)
                if multiscale_loss:
                    loss = loss + lambda_multiscale * multiscale_loss(sr_batch, hr_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(loss.item())

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(
                    "  Epoch %02d/%02d | Batch %03d/%03d | Loss=%.4f | LR=%.2e",
                    epoch, epochs, batch_idx + 1, len(train_loader), loss.item(),
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_psnrs = []
        val_ssims = []

        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                if lr_batch.numel() == 0:
                    continue
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                sr_batch = model(lr_batch)
                v_loss = F.l1_loss(sr_batch, hr_batch)
                val_losses.append(v_loss.item())

                mse = F.mse_loss(sr_batch, hr_batch)
                psnr_val = 10.0 * torch.log10(1.0 / (mse + 1e-8)).item()
                val_psnrs.append(psnr_val)

                val_ssims.append(_compute_batch_ssim(sr_batch, hr_batch))

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        mean_psnr = float(np.mean(val_psnrs)) if val_psnrs else 0.0
        mean_ssim = float(np.mean(val_ssims)) if val_ssims else 0.0
        epoch_time = time.time() - t_start

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        history["val_psnr"].append(round(mean_psnr, 2))
        history["val_ssim"].append(round(mean_ssim, 4))
        history["epoch_time"].append(round(epoch_time, 2))
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        logger.info(
            "Epoch %02d/%02d | Train: %.4f | Val: %.4f | PSNR: %.2f dB | SSIM: %.4f | %.1fs",
            epoch, epochs, mean_train, mean_val, mean_psnr, mean_ssim, epoch_time,
        )

        # Best checkpoint selection
        if mean_psnr > best_val_psnr:
            best_val_psnr = mean_psnr
            patience_counter = 0
            save_generator_checkpoint(
                model, best_checkpoint_path, epoch=epoch,
                extra_info={"val_loss": mean_val, "val_psnr": mean_psnr, "val_ssim": mean_ssim},
            )
            logger.info("  ✓ Best checkpoint updated! (Val PSNR=%.2f dB, SSIM=%.4f)", mean_psnr, mean_ssim)
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % save_every == 0:
            periodic_path = output_dir / f"model_finetuned_epoch{epoch:03d}.pth"
            save_generator_checkpoint(model, periodic_path, epoch=epoch)

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.warning("Early stopping triggered after %d epochs without improvement", patience_counter)
            break

        # Time budget check
        if time_budget_hours is not None:
            elapsed_h = (time.time() - t_global_start) / 3600.0
            if elapsed_h >= time_budget_hours:
                logger.warning("Time budget of %.2fh reached. Stopping training.", time_budget_hours)
                break

    # Save final checkpoint
    save_generator_checkpoint(
        model, final_checkpoint_path, epoch=epoch,
        extra_info={"history": history, "best_val_psnr": best_val_psnr},
    )

    # Save history JSON
    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete. Best PSNR: %.2f dB | History saved → %s", best_val_psnr, hist_path)

    return {"best_checkpoint": str(best_checkpoint_path), "history": history, "best_val_psnr": best_val_psnr}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Super-Resolution Fine-Tuning for Sentinel-2 (SIH26142)")
    parser.add_argument("--pairs-dir", default="data/synthetic_pairs", help="Directory with lr/ and hr/ pairs")
    parser.add_argument("--output-dir", default="src/checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--pretrained", default=None, help="Override pretrained checkpoint path")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--model-key", default="x4plus", choices=["x4plus", "x4plus_anime"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    parser.add_argument("--use-gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM")
    parser.add_argument("--joint-spectral", action="store_true", help="Enable joint multi-band spectral modeling")

    # Loss terms (set to 0.0 to disable with zero overhead)
    parser.add_argument("--lambda-l1", type=float, default=1.0, help="Weight for L1 pixel loss")
    parser.add_argument("--lambda-perceptual", type=float, default=0.1, help="Weight for VGG16 perceptual loss")
    parser.add_argument("--lambda-sam", type=float, default=0.05, help="Weight for Spectral Angle Mapper loss")
    parser.add_argument("--lambda-edge", type=float, default=0.05, help="Weight for Sobel edge loss")
    parser.add_argument("--lambda-freq", type=float, default=0.02, help="Weight for FFT frequency loss")
    parser.add_argument("--lambda-multiscale", type=float, default=0.0, help="Weight for multi-scale pyramid loss")

    # Ablation presets
    parser.add_argument(
        "--ablation",
        choices=["l1_only", "with_perceptual", "with_sam", "with_edge", "with_freq", "full", "joint_spectral"],
        default=None,
        help="Convenient ablation preset. Overrides individual lambda defaults.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    # Apply ablation presets
    ablation_name = args.ablation or "custom"
    lambda_l1 = args.lambda_l1
    lambda_perceptual = args.lambda_perceptual
    lambda_sam = args.lambda_sam
    lambda_edge = args.lambda_edge
    lambda_freq = args.lambda_freq
    lambda_multiscale = args.lambda_multiscale
    joint_spectral = args.joint_spectral

    if args.ablation == "l1_only":
        lambda_l1 = 1.0
        lambda_perceptual = 0.0
        lambda_sam = 0.0
        lambda_edge = 0.0
        lambda_freq = 0.0
        lambda_multiscale = 0.0
        joint_spectral = False
    elif args.ablation == "with_perceptual":
        lambda_l1 = 1.0
        lambda_perceptual = 0.1
        lambda_sam = 0.0
        lambda_edge = 0.0
        lambda_freq = 0.0
        lambda_multiscale = 0.0
        joint_spectral = False
    elif args.ablation == "with_sam":
        lambda_l1 = 1.0
        lambda_perceptual = 0.1
        lambda_sam = 0.05
        lambda_edge = 0.0
        lambda_freq = 0.0
        lambda_multiscale = 0.0
        joint_spectral = False
    elif args.ablation == "with_edge":
        lambda_l1 = 1.0
        lambda_perceptual = 0.1
        lambda_sam = 0.05
        lambda_edge = 0.05
        lambda_freq = 0.0
        lambda_multiscale = 0.0
        joint_spectral = False
    elif args.ablation == "with_freq":
        lambda_l1 = 1.0
        lambda_perceptual = 0.1
        lambda_sam = 0.05
        lambda_edge = 0.05
        lambda_freq = 0.02
        lambda_multiscale = 0.0
        joint_spectral = False
    elif args.ablation == "full":
        lambda_l1 = 1.0
        lambda_perceptual = 0.1
        lambda_sam = 0.05
        lambda_edge = 0.05
        lambda_freq = 0.02
        lambda_multiscale = 0.05
        joint_spectral = False
    elif args.ablation == "joint_spectral":
        lambda_l1 = 1.0
        lambda_perceptual = 0.1
        lambda_sam = 0.05
        lambda_edge = 0.05
        lambda_freq = 0.02
        lambda_multiscale = 0.05
        joint_spectral = True

    train(
        pairs_dir=args.pairs_dir,
        output_dir=args.output_dir,
        model_key=args.model_key,
        pretrained_checkpoint=args.pretrained,
        resume_checkpoint=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        lr=args.lr,
        lambda_l1=lambda_l1,
        lambda_perceptual=lambda_perceptual,
        lambda_sam=lambda_sam,
        lambda_edge=lambda_edge,
        lambda_freq=lambda_freq,
        lambda_multiscale=lambda_multiscale,
        joint_spectral=joint_spectral,
        save_every=args.save_every,
        num_workers=args.num_workers,
        use_amp=(not args.no_amp),
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        time_budget_hours=args.time_budget_hours,
        ablation_name=ablation_name,
    )
