"""
spectral_fusion.py
──────────────────
Joint multi-band spectral modeling module for Sentinel-2 Super-Resolution (SIH26142).

In standard Real-ESRGAN, RGB bands are processed together while all other bands
(NIR, SWIR, RedEdge, etc.) are processed independently as replicated grayscale images.
This means no cross-spectral relationship is ever modeled.

In contrast, satellite remote sensing models like DSen2 (Lanaras et al., ISPRS 2018)
and cross-attention architectures demonstrate that jointly modeling all bands
yields significantly higher radiometric and spectral accuracy (lower SAM, higher PSNR/SSIM).

This module provides:
  1. ChannelSpectralAttention:
     Learns inter-band correlation weights via combined global average and max pooling.
  2. JointSpectralFusionModule:
     Cross-spectral convolutional refinement block that mixes spectral channels,
     applies spatial-spectral convolutions, and conditions each band on all others.
  3. JointSpectralSR:
     End-to-end wrapper that pairs with an RRDBNet generator to provide joint
     multi-band super-resolution with residual learning.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Channel Spectral Attention
# ──────────────────────────────────────────────────────────────────────────────

class ChannelSpectralAttention(nn.Module):
    """
    Channel attention mechanism tailored for multi-spectral satellite imagery.

    Models inter-band dependencies (e.g., NDVI-like correlations between Red and NIR,
    water absorption in SWIR vs Green) using dual spatial pooling (avg + max).
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 4):
        super().__init__()
        self.num_channels = num_channels
        mid_channels = max(num_channels // reduction_ratio, 4)

        self.mlp = nn.Sequential(
            nn.Linear(num_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, num_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W)

        Returns
        -------
        out : torch.Tensor of shape (B, C, H, W) with channel-wise modulation
        """
        b, c, _, _ = x.shape
        # Global Average Pooling: (B, C)
        gap = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        # Global Max Pooling: (B, C)
        gmp = F.adaptive_max_pool2d(x, (1, 1)).view(b, c)

        # Shared MLP
        avg_out = self.mlp(gap)
        max_out = self.mlp(gmp)

        channel_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * channel_weights


# ──────────────────────────────────────────────────────────────────────────────
# Joint Spectral Fusion Block
# ──────────────────────────────────────────────────────────────────────────────

class JointSpectralFusionBlock(nn.Module):
    """
    Residual block with 1x1 spectral mixing convs + 3x3 depthwise spatial convs
    + cross-channel spectral attention.
    """

    def __init__(self, num_channels: int, feat_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels

        # 1x1 conv to mix all spectral bands into a rich feature space
        self.spectral_mix_in = nn.Conv2d(num_channels, feat_channels, kernel_size=1, bias=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # 3x3 spatial convolution for local context
        self.spatial_conv = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Cross-channel attention in feature space
        self.attention = ChannelSpectralAttention(feat_channels, reduction_ratio=4)

        # 1x1 conv back to original spectral channels
        self.spectral_mix_out = nn.Conv2d(feat_channels, num_channels, kernel_size=1, bias=True)

        # Learnable residual scale, initialized small so block starts as identity
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        feat = self.act1(self.spectral_mix_in(x))
        feat = self.act2(self.spatial_conv(feat))
        feat = self.attention(feat)
        out = self.spectral_mix_out(feat)
        return residual + self.res_scale * out


# ──────────────────────────────────────────────────────────────────────────────
# Joint Spectral Refinement Module
# ──────────────────────────────────────────────────────────────────────────────

class JointSpectralRefiner(nn.Module):
    """
    Post-SR joint spectral refinement module.

    Takes initial per-band super-resolved outputs (B, C, H_sr, W_sr) and refines
    cross-spectral correlations to eliminate inter-band color/radiometric distortions
    and minimize Spectral Angle Mapper (SAM) error.
    """

    def __init__(self, num_channels: int = 4, num_blocks: int = 3, feat_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.blocks = nn.ModuleList([
            JointSpectralFusionBlock(num_channels=num_channels, feat_channels=feat_channels)
            for _ in range(num_blocks)
        ])
        self.final_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # Small initial residual weight
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W)

        Returns
        -------
        out : torch.Tensor of shape (B, C, H, W)
        """
        residual = x
        feat = x
        for block in self.blocks:
            feat = block(feat)
        delta = self.final_conv(feat)
        out = residual + self.alpha * delta
        return out.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# High-Level Joint Multi-Band Super-Resolution Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class JointSpectralSR(nn.Module):
    """
    Wraps a 3-channel base generator (e.g. RRDBNet) with a JointSpectralRefiner.

    In forward pass:
      1. Splits multi-band input into RGB and non-RGB bands.
      2. Generates initial 4x SR for each band group through the base generator.
      3. Stacks all bands and passes them through JointSpectralRefiner to model
         cross-spectral relationships together.
    """

    def __init__(
        self,
        base_generator: nn.Module,
        num_channels: int = 4,
        rgb_band_indices: Tuple[int, ...] = (0, 1, 2),
        num_refiner_blocks: int = 3,
    ):
        super().__init__()
        self.base_generator = base_generator
        self.num_channels = num_channels
        self.rgb_band_indices = rgb_band_indices
        self.refiner = JointSpectralRefiner(
            num_channels=num_channels,
            num_blocks=num_refiner_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W)

        Returns
        -------
        sr : torch.Tensor of shape (B, C, H*4, W*4)
        """
        b, c, h, w = x.shape
        valid_rgb = [i for i in self.rgb_band_indices if i < c]
        use_rgb = (len(valid_rgb) == 3)
        gray_indices = [i for i in range(c) if (not use_rgb or i not in valid_rgb)]

        # Initial SR tensor
        sr_bands = [None] * c

        if use_rgb:
            rgb_in = x[:, list(valid_rgb), :, :]  # (B, 3, H, W)
            sr_rgb = self.base_generator(rgb_in)  # (B, 3, 4H, 4W)
            for out_pos, b_idx in enumerate(valid_rgb):
                sr_bands[b_idx] = sr_rgb[:, out_pos:out_pos + 1, :, :]

        for g_idx in gray_indices:
            g_in = x[:, g_idx:g_idx + 1, :, :]  # (B, 1, H, W)
            g_3ch = g_in.repeat(1, 3, 1, 1)      # (B, 3, H, W)
            sr_g = self.base_generator(g_3ch)   # (B, 3, 4H, 4W)
            sr_bands[g_idx] = sr_g[:, 0:1, :, :]

        # Stack into joint tensor: (B, C, 4H, 4W)
        sr_initial = torch.cat(sr_bands, dim=1)

        # Cross-spectral refinement across all bands
        if self.refiner.num_channels == c:
            sr_final = self.refiner(sr_initial)
        else:
            sr_final = sr_initial

        return sr_final
