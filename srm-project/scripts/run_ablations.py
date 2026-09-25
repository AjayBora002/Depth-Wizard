#!/usr/bin/env python
"""
run_ablations.py
────────────────
Reproducible ablation study runner for SIH26142 Super-Resolution Mapping.

Executes controlled ablation experiments in sequence or individually:
  A1: L1-only baseline (pure pixel reconstruction)
  A2: + Perceptual loss (VGG16 feature realism)
  A3: + SAM loss (Spectral Angle Mapper for multi-band fidelity)
  A4: + Edge loss (Sobel gradient preservation on linear features)
  A5: + Frequency loss (FFT high-frequency detail recovery)
  A6: + Joint Spectral Modeling (cross-band attention & correlation)

Usage:
  # Run a single ablation configuration:
  python scripts/run_ablations.py --ablation with_sam --epochs 30

  # Run all 6 ablation experiments sequentially:
  python scripts/run_ablations.py --all --epochs 30 --output-root data/ablations/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

ABLATIONS = [
    {
        "name": "A1_l1_only",
        "flag": "l1_only",
        "description": "Baseline L1 loss only",
        "extra_args": [],
    },
    {
        "name": "A2_with_perceptual",
        "flag": "with_perceptual",
        "description": "L1 + VGG16 Perceptual Loss",
        "extra_args": [],
    },
    {
        "name": "A3_with_sam",
        "flag": "with_sam",
        "description": "L1 + Perceptual + SAM Loss (Spectral Preservation)",
        "extra_args": [],
    },
    {
        "name": "A4_with_edge",
        "flag": "with_edge",
        "description": "L1 + Perceptual + SAM + Edge Loss (Sobel Gradients)",
        "extra_args": [],
    },
    {
        "name": "A5_with_freq",
        "flag": "with_freq",
        "description": "L1 + Perceptual + SAM + Edge + Frequency Loss (FFT)",
        "extra_args": [],
    },
    {
        "name": "A6_joint_spectral",
        "flag": "joint_spectral",
        "description": "Full Composite Loss + Joint Multi-Band Spectral Modeling",
        "extra_args": ["--joint-spectral"],
    },
]


def run_ablation(
    ablation: dict,
    pairs_dir: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    crop_size: int,
    use_gradient_checkpointing: bool = False,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "src" / "train.py"),
        "--pairs-dir", pairs_dir,
        "--output-dir", str(output_dir),
        "--ablation", ablation["flag"],
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--crop-size", str(crop_size),
    ]
    if use_gradient_checkpointing:
        cmd.append("--use-gradient-checkpointing")
    cmd.extend(ablation["extra_args"])

    print("=" * 75)
    print(f"▶ Running Ablation: {ablation['name']}")
    print(f"  Description: {ablation['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 75)

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode == 0:
        print(f"✓ Ablation {ablation['name']} completed successfully in {elapsed:.1f}s")
        return True
    else:
        print(f"✗ Ablation {ablation['name']} failed with exit code {res.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="SIH26142 Ablation Study Runner")
    parser.add_argument("--pairs-dir", default="data/synthetic_pairs", help="Dataset directory")
    parser.add_argument("--output-root", default="data/ablations", help="Root directory for ablation outputs")
    parser.add_argument(
        "--ablation",
        choices=[a["flag"] for a in ABLATIONS],
        default=None,
        help="Run a specific ablation preset",
    )
    parser.add_argument("--all", action="store_true", help="Run all 6 ablations sequentially")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_root)

    if args.all:
        for abl in ABLATIONS:
            out_dir = out_root / abl["name"]
            success = run_ablation(
                abl,
                pairs_dir=args.pairs_dir,
                output_dir=out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                crop_size=args.crop_size,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
            )
            if not success:
                print("Stopping ablation suite due to error.")
                sys.exit(1)
    elif args.ablation:
        matched = next(a for a in ABLATIONS if a["flag"] == args.ablation)
        out_dir = out_root / matched["name"]
        run_ablation(
            matched,
            pairs_dir=args.pairs_dir,
            output_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
        )
    else:
        print("Please specify either --ablation <name> or --all. Available ablations:")
        for a in ABLATIONS:
            print(f"  {a['flag']:<18} : {a['description']}")


if __name__ == "__main__":
    main()
