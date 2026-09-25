#!/usr/bin/env python
"""
run_demo.py
───────────
Complete SIH demo workflow: Generate data → Train → Create visualizations

This script automates the entire demo preparation process:
  1. Generate synthetic training pairs from raw Sentinel-2 data
  2. Run enhanced training with optimal hyperparameters
  3. Generate impressive before/after comparisons

Usage:
  python run_demo.py --mode all
  python run_demo.py --mode generate  # Only generate training pairs
  python run_demo.py --mode train     # Only train model
  python run_demo.py --mode demo      # Only create demo visualizations
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and report status."""
    logger.info(f"\n{'='*80}")
    logger.info(f"▶ {description}")
    logger.info(f"Command: {' '.join(str(c) for c in cmd)}")
    logger.info('='*80)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        return False


def generate_training_data(
    raw_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/synthetic_pairs"),
    patch_size: int = 512,
    overlap: int = 64,
    rgb_only: bool = True,
) -> bool:
    """Generate synthetic training pairs from raw Sentinel-2 tiles."""
    cmd = [
        sys.executable,
        "src/pair_generation.py",
        "--raw-dir", str(raw_dir),
        "--out-dir", str(out_dir),
        "--patch-size", str(patch_size),
        "--overlap", str(overlap),
        "--seed", "42",
    ]

    if rgb_only:
        cmd.append("--rgb-only")

    return run_command(cmd, "Generating training pairs")


def train_model(
    pairs_dir: Path = Path("data/synthetic_pairs"),
    output_dir: Path = Path("src/checkpoints"),
    epochs: int = 50,
    batch_size: int = 4,
    crop_size: int = 192,
    lr: float = 5e-5,
    warmup_epochs: int = 5,
    resume: Path = None,
) -> bool:
    """Run enhanced training."""
    cmd = [
        sys.executable,
        "src/train_enhanced.py",
        "--pairs-dir", str(pairs_dir),
        "--output-dir", str(output_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--crop-size", str(crop_size),
        "--lr", str(lr),
        "--warmup-epochs", str(warmup_epochs),
        "--save-every", "5",
        "--num-workers", "2",
    ]

    if resume:
        cmd.extend(["--resume", str(resume)])

    return run_command(cmd, "Training model")


def create_demo_visualizations(
    input_image: Path,
    output_dir: Path = Path("data/outputs"),
    checkpoint: Path = None,
) -> bool:
    """Generate demo comparison images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main comparison
    output_path = output_dir / f"{input_image.stem}_comparison.png"
    cmd = [
        sys.executable,
        "src/demo_inference.py",
        "--input", str(input_image),
        "--output", str(output_path),
        "--compare",
        "--zoom",
    ]

    if checkpoint:
        cmd.extend(["--checkpoint", str(checkpoint)])

    return run_command(cmd, "Creating demo visualizations")


def main():
    parser = argparse.ArgumentParser(
        description="Complete SIH demo workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (recommended for first time)
  python run_demo.py --mode all

  # Only generate training data
  python run_demo.py --mode generate --patch-size 512

  # Only train (requires existing training pairs)
  python run_demo.py --mode train --epochs 100

  # Only create demo visualizations
  python run_demo.py --mode demo --input data/raw/2026-04-08-00_00_2026-04-08-23_59_Sentinel-2_L2A_True_color.tiff
"""
    )

    parser.add_argument(
        "--mode",
        choices=["all", "generate", "train", "demo"],
        default="all",
        help="Which step to run"
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Processing preset: fast (30-60s), balanced (2-3min), quality (5-10min)"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with raw Sentinel-2 GeoTIFFs"
    )
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        default=Path("data/synthetic_pairs"),
        help="Directory for training pairs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/checkpoints"),
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--demo-output-dir",
        type=Path,
        default=Path("data/outputs"),
        help="Directory for demo outputs"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch size for training pair generation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=192,
        help="Crop size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="LR warmup epochs"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image for demo mode"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned checkpoint for demo inference"
    )

    args = parser.parse_args()

    logger.info("\n" + "="*80)
    logger.info("🛰️  Depth-Wizard SIH Demo Workflow")
    logger.info("="*80 + "\n")

    success = True

    # Step 1: Generate training data
    if args.mode in ["all", "generate"]:
        logger.info("\n📍 Step 1: Generating training pairs...")
        if not generate_training_data(
            raw_dir=args.raw_dir,
            out_dir=args.pairs_dir,
            patch_size=args.patch_size,
        ):
            logger.error("Failed to generate training data")
            if args.mode == "all":
                return 1
            success = False

    # Step 2: Train model
    if args.mode in ["all", "train"] and success:
        logger.info("\n📍 Step 2: Training model...")
        if not train_model(
            pairs_dir=args.pairs_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            resume=args.resume,
        ):
            logger.error("Failed to train model")
            if args.mode == "all":
                return 1
            success = False

    # Step 3: Create demo visualizations
    if args.mode in ["all", "demo"] and success:
        logger.info("\n📍 Step 3: Creating demo visualizations...")

        # Find input images
        if args.input:
            input_images = [args.input]
        else:
            input_images = list(args.raw_dir.glob("*.tif")) + list(args.raw_dir.glob("*.tiff"))

        if not input_images:
            logger.error("No input images found")
            return 1

        # Use best checkpoint if available
        checkpoint = args.checkpoint
        if not checkpoint:
            best_ckpt = args.output_dir / "model_finetuned_best.pth"
            if best_ckpt.exists():
                checkpoint = best_ckpt
                logger.info(f"Using best checkpoint: {checkpoint}")

        for img in input_images[:1]:  # Process first image only
            if not create_demo_visualizations(
                input_image=img,
                output_dir=args.demo_output_dir,
                checkpoint=checkpoint,
            ):
                logger.error(f"Failed to create demo for {img}")
                success = False

    if success:
        logger.info("\n" + "="*80)
        logger.info("✅ SIH Demo workflow completed successfully!")
        logger.info("="*80)

        # Print summary
        logger.info("\n📊 Next steps:")
        logger.info(f"  1. View training curves: {args.output_dir}/training_curves.png")
        logger.info(f"  2. View demo outputs: {args.demo_output_dir}/")
        logger.info(f"  3. Best checkpoint: {args.output_dir}/model_finetuned_best.pth")
        logger.info(f"  4. Training history: {args.output_dir}/training_history.json")

        return 0
    else:
        logger.error("\n❌ Workflow failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())