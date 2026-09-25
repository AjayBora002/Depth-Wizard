#!/usr/bin/env python
"""
generate_sih_demo.py
────────────────────
Complete SIH presentation demo generator.

Automatically processes sample images with all three presets (Fast/Balanced/Quality),
generates comparison visualizations, and produces a summary report for the panel.

Usage:
  python generate_sih_demo.py
  python generate_sih_demo.py --input data/raw/sample.tif --output-dir demo_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def run_inference_preset(
    input_path: Path,
    output_dir: Path,
    preset: str,
    checkpoint_path: Path | None = None,
) -> dict:
    """Run inference with a specific preset and return timing + output info."""
    logger.info(f"\n{'='*70}")
    logger.info(f"▶ Processing: {input_path.name} | Preset: {preset.upper()}")
    logger.info(f"{'='*70}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    output_path = output_dir / f"{stem}_{preset}_sr_x4.tif"

    cmd = [
        sys.executable,
        "src/inference.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--preset", preset,
    ]

    if checkpoint_path and checkpoint_path.exists():
        cmd.extend(["--checkpoint", str(checkpoint_path)])

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        elapsed = time.time() - t0
        logger.info(f"✓ {preset.upper()} complete in {elapsed:.1f}s")

        return {
            "preset": preset,
            "elapsed_seconds": round(elapsed, 2),
            "output": str(output_path),
            "success": True,
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {preset.upper()} failed:\n{e.stderr}")
        return {
            "preset": preset,
            "elapsed_seconds": 0,
            "output": None,
            "success": False,
            "error": str(e),
        }


def generate_comparison_report(
    input_path: Path,
    results: List[dict],
    output_dir: Path,
) -> None:
    """Generate a summary HTML report comparing the three presets."""

    report_html = f"""
    <html>
    <head>
        <title>SIH SR Demo Report - {input_path.stem}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2em;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .results {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .result-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .result-card h3 {{
                margin-top: 0;
                color: #333;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 1px;
            }}
            .timing {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 15px 0;
            }}
            .status {{
                padding: 10px;
                border-radius: 4px;
                font-weight: 500;
            }}
            .status.success {{
                background: #d4edda;
                color: #155724;
            }}
            .status.error {{
                background: #f8d7da;
                color: #721c24;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 15px;
                font-size: 0.9em;
            }}
            .metric {{
                background: #f9f9f9;
                padding: 10px;
                border-left: 3px solid #667eea;
            }}
            .metric-label {{
                color: #666;
                font-size: 0.85em;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-weight: bold;
                font-size: 1.1em;
            }}
            .footer {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛰️ Depth-Wizard SR Demo</h1>
            <p>SIH26142 - Smart India Hackathon 2026</p>
            <p>Input: {input_path.name}</p>
        </div>

        <div class="results">
    """

    for result in results:
        preset = result["preset"]
        status = "✓ Success" if result["success"] else "✗ Failed"
        status_class = "success" if result["success"] else "error"
        timing = f"{result['elapsed_seconds']}s"

        report_html += f"""
            <div class="result-card">
                <h3>{preset}</h3>
                <div class="timing">{timing}</div>
                <div class="status {status_class}">{status}</div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Preset</div>
                        <div class="metric-value">{preset.upper()}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Time</div>
                        <div class="metric-value">{timing}</div>
                    </div>
                </div>
        """
        if result["success"]:
            report_html += f"""
                <div class="metric">
                    <div class="metric-label">Output</div>
                    <div class="metric-value"><a href="{result['output']}">{Path(result['output']).name}</a></div>
                </div>
            """
        report_html += """
            </div>
        """

    report_html += """
        </div>

        <div class="footer">
            <p>Generated with Depth-Wizard Super-Resolution System</p>
            <p>Ready for SIH panel presentation 🎉</p>
        </div>
    </body>
    </html>
    """

    report_path = output_dir / f"demo_report_{input_path.stem}.html"
    with open(report_path, "w") as f:
        f.write(report_html)
    logger.info(f"Report saved: {report_path}")

    # Also save JSON report
    json_report_path = output_dir / f"demo_report_{input_path.stem}.json"
    with open(json_report_path, "w") as f:
        json.dump({
            "input": str(input_path),
            "results": results,
        }, f, indent=2)
    logger.info(f"JSON report saved: {json_report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate complete SIH demo with all presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all samples with all presets
  python generate_sih_demo.py

  # Process specific image
  python generate_sih_demo.py --input data/raw/2026-04-08-00_00_2026-04-08-23_59_Sentinel-2_L2A_True_color.tiff

  # Custom output directory
  python generate_sih_demo.py --output-dir /tmp/sih_demo/
"""
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Specific input image (optional; auto-discovers from data/raw if not provided)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/demo_outputs"),
        help="Output directory for demo results"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned checkpoint path (optional)"
    )
    args = parser.parse_args()

    # Find input images
    if args.input:
        input_images = [args.input]
    else:
        raw_dir = Path("data/raw")
        input_images = list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff"))

    if not input_images:
        logger.error("No input images found. Provide --input or place .tif files in data/raw/")
        return 1

    logger.info(f"\n🎬 SIH Demo Generator")
    logger.info(f"{'='*70}")
    logger.info(f"Found {len(input_images)} image(s) to process")
    logger.info(f"Presets: Fast (30-60s) | Balanced (2-3min) | Quality (5-10min)")
    logger.info(f"{'='*70}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process first 3 images with all presets
    for img_idx, img_path in enumerate(input_images[:3]):
        logger.info(f"\n📍 Image {img_idx+1}/{min(3, len(input_images))}: {img_path.name}")

        img_output_dir = args.output_dir / img_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for preset in ["fast", "balanced", "quality"]:
            result = run_inference_preset(
                img_path,
                img_output_dir,
                preset,
                args.checkpoint,
            )
            results.append(result)

        # Generate comparison report
        generate_comparison_report(img_path, results, img_output_dir)

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Demo generation complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*70}\n")

    # Print summary
    logger.info("📊 Summary:")
    logger.info("  Fast:     30-60s   | Large patches, no uncertainty")
    logger.info("  Balanced: 2-3min   | Optimized defaults (recommended)")
    logger.info("  Quality:  5-10min  | Full TTA, small patches")
    logger.info("\n💡 Tip: Use 'Balanced' preset for live SIH panel demo\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
