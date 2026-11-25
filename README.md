# Depth-Wizard — Sentinel-2 Super-Resolution Mapping (SRM)

> **SIH26142 (Smart India Hackathon 2026) | Sponsored by NTRO**
>
> **4× Deep Learning Super-Resolution for Multi-Band Sentinel-2 L2A Satellite Imagery (10 m → 2.5 m-equivalent)** with full geo-referencing preservation, cross-band spectral modeling, Test-Time Augmentation (TTA) uncertainty quantification, and an interactive Streamlit + Folium web dashboard.

---

## Table of Contents

1. [Problem Statement & Objectives](#-problem-statement--objectives)
2. [Key Features & Innovations](#-key-features--innovations)
3. [System Architecture](#-system-architecture)
4. [Repository Structure](#-repository-structure)
5. [Methodology & Technical Details](#-methodology--technical-details)
   - [Sentinel-2 Band Stack & Preprocessing](#1-sentinel-2-band-stack--preprocessing)
   - [Synthetic Training Pair Generation](#2-synthetic-training-pair-generation)
   - [Super-Resolution Backbone & Joint Spectral Modeling](#3-super-resolution-backbone--joint-spectral-modeling)
   - [Multi-Temporal Fusion](#4-multi-temporal-fusion)
   - [Composite Loss Suite & Training Optimizations](#5-composite-loss-suite--training-optimizations)
   - [Geo-Aware Tiled Inference & Seamless Blending](#6-geo-aware-tiled-inference--seamless-blending)
   - [Uncertainty Quantification (TTA Ensemble)](#7-uncertainty-quantification-tta-ensemble)
   - [Remote Sensing Evaluation Metrics](#8-remote-sensing-evaluation-metrics)
6. [Benchmark Results](#-benchmark-results)
7. [Installation & Environment Setup](#-installation--environment-setup)
8. [Usage Guide](#-usage-guide)
   - [1. Acquire & Validate Sentinel-2 Data](#1-acquire--validate-sentinel-2-data)
   - [2. Generate Synthetic Training Pairs](#2-generate-synthetic-training-pairs)
   - [3. Fine-Tune the Model](#3-fine-tune-the-model)
   - [4. Run Ablation Studies](#4-run-ablation-studies)
   - [5. Run Super-Resolution Inference](#5-run-super-resolution-inference)
   - [6. Evaluate Metrics](#6-evaluate-metrics)
   - [7. Launch the Web Dashboard](#7-launch-the-web-dashboard)
   - [8. One-Command Demo Workflow](#8-one-command-demo-workflow)
9. [Running Tests](#-running-tests)
10. [Deployment (Streamlit Community Cloud)](#-deployment-streamlit-community-cloud)
11. [Engineering Fixes & Lessons Learned](#-engineering-fixes--lessons-learned)
12. [Known Limitations](#-known-limitations)
13. [Future Roadmap](#-future-roadmap)
14. [Dependencies](#-dependencies)
15. [License & Acknowledgements](#-license--acknowledgements)

---

## 📋 Problem Statement & Objectives

**SIH26142** challenges teams to build a deep learning–based **Super-Resolution Mapping (SRM)** system that enhances medium-resolution satellite imagery (Sentinel-2, 10 m spatial resolution) to produce sub-4 m equivalent outputs (4× upscaling → **2.5 m-equivalent** pixel pitch) while satisfying strict remote sensing and operational requirements:

- **Geo-Referencing Preservation**: Maintain exact Coordinate Reference System (CRS) and scaled affine geo-transform matrices across tiled inference and reconstruction.
- **Spectral Fidelity**: Preserve physical reflectance curves across multi-spectral bands (RGB, NIR, Red Edge, SWIR) rather than hallucinating unnatural colors.
- **Uncertainty Quantification**: Provide per-pixel confidence/uncertainty heatmaps so downstream analysts can distinguish high-confidence reconstructions from model guesses.
- **Objective Validation**: Evaluate outputs using standard remote sensing metrics (**PSNR, SSIM, SAM, SRE, ERGAS, UIQ**).
- **Operational Usability**: Deliver a web-based dashboard supporting single-tile analysis, batch ZIP processing, before/after visual comparison, and interactive geo-maps.

---

## ✨ Key Features & Innovations

- **Self-Contained RRDBNet Backbone (`src/rrdbnet.py`)**: 23-block Residual-in-Residual Dense Block architecture with gradient checkpointing support, removing hard runtime dependencies on `basicsr` when unavailable or conflicting with newer Python/torchvision versions.
- **Joint Multi-Band Spectral Modeling (`src/spectral_fusion.py`)**: Inspired by DSen2 (Lanaras et al., 2018), `JointSpectralSR` combines `ChannelSpectralAttention` (dual average + max pooling) and `JointSpectralFusionModule` (1×1 spectral mixing + spatial-spectral convolutions) to couple all bands and enable cross-band gradient flow directly into the Spectral Angle Mapper (SAM) loss.
- **Multi-Temporal Fusion (`src/temporal_fusion.py`)**: Exploits Sentinel-2's 5-day repeat-pass cycle using ORB feature-based homography alignment (`align_temporal_stack`) and temporal median filtering for cloud/haze removal and infrastructure change detection.
- **Ablation-Ready Composite Loss Suite (`src/train.py`)**: Combines L1 pixel loss, VGG16 Perceptual loss, differentiable SAM loss, Sobel edge gradient loss, 2D FFT frequency loss, and Multi-Scale pyramid loss—each incurring zero compute/memory overhead when its weight (`lambda`) is `0.0`.
- **Artifact-Free GeoTIFF Tiling & Blending (`src/preprocessing.py`, `src/inference.py`)**: Uses dtype-based global normalization and per-patch 2D Hann-window blending to eliminate brightness grids and tile-boundary seams.
- **8-Pass TTA Uncertainty Mapping (`src/uncertainty.py`)**: Computes per-pixel variance across 8 dihedral geometric augmentations (D4 group) and renders viridis uncertainty heatmaps.
- **Interactive Streamlit Dashboard (`dashboard/app.py`)**: Upload GeoTIFFs or select samples, run 4× SR with live progress, inspect before/after sliders, view PSNR/SSIM/SAM metric cards, overlay uncertainty heatmaps, verify bounds on a Folium map, and batch-process ZIP archives.

---

## 🏗️ System Architecture

```text
Sentinel-2 L2A GeoTIFF (10 m, Multi-Band)
                 │
                 ▼
      ┌─────────────────────┐
      │  preprocessing.py   │  Validate CRS/transform, tile into overlapping patches
      └──────────┬──────────┘  with global dtype normalization & per-patch affine transforms
                 │
        ┌────────┴────────────────────────────┐
        │ [Training Path]                     │ [Inference Path]
        ▼                                     ▼
┌───────────────────┐               ┌───────────────────────┐
│ pair_generation.py│               │ model.py / rrdbnet.py │
│ Blur → ↓4× →      │               │ Real-ESRGAN x4plus    │
│ Noise → JPEG      │               │ (+ JointSpectralSR)   │
└────────┬──────────┘               └───────────┬───────────┘
         │                                      │
         ▼                               ┌──────┴──────┐
┌───────────────────┐                    ▼             ▼
│     train.py      │              SR Patches   uncertainty.py
│ L1 + VGG16 + SAM  │                    │      (8× TTA Ensemble
│ + Edge + FFT + MS │                    │       Variance Map)
│ → checkpoints/    │                    └──────┬──────┘
└───────────────────┘                           │
                                                ▼
                                     ┌─────────────────────┐
                                     │    inference.py     │
                                     │ Per-patch 2D Hann   │
                                     │ blending + GeoTIFF  │
                                     │ + edge sharpening   │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │     metrics.py      │
                                     │ PSNR, SSIM, SAM,    │
                                     │ SRE, ERGAS, UIQ     │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │  dashboard/app.py   │
                                     │ Streamlit + Folium  │
                                     │ + Plotly + Batch ZIP│
                                     └─────────────────────┘
```

---

## 📁 Repository Structure

```text
Depth-Wizard/
├── .gitignore
├── README.md                              # Project documentation (this file)
└── srm-project/
    ├── BUG_FIXES.md                       # Detailed post-mortems of tiling/blending/metric fixes
    ├── DEPLOY.md                          # Step-by-step Streamlit Community Cloud deployment guide
    ├── PROGRESS.md                        # Phase-by-phase build log and status checklist
    ├── README.md                          # Subproject README
    ├── download_sentinel2_tiles.py        # Helper script for Sentinel-2 tile acquisition
    ├── packages.txt                       # Apt dependencies for Streamlit Cloud (GDAL/PROJ/GEOS)
    ├── requirements.txt                   # Local Python dependencies
    ├── requirements_cloud.txt             # CPU-only PyTorch + GitHub basicsr for cloud deployment
    ├── run_dashboard.bat                  # Windows batch launcher for Streamlit dashboard
    ├── run_demo.py                        # End-to-end demo runner (generate → train → visualize)
    ├── runtime.txt                        # Python 3.11 pin for Streamlit Cloud
    ├── setup.cfg                          # Pytest configuration
    ├── setup_env.ps1                      # Windows PowerShell environment setup script
    ├── setup_gpu.bat                      # Windows GPU environment setup script
    ├── dashboard/
    │   └── app.py                         # Interactive Streamlit web dashboard
    ├── data/
    │   ├── raw/                           # Raw Sentinel-2 L2A GeoTIFF scenes
    │   ├── synthetic_pairs/               # Generated LR/HR .npy patch pairs
    │   └── training_pairs/                # Curated training pairs
    ├── notebooks/
    │   └── colab_finetune.ipynb           # Self-contained Google Colab GPU fine-tuning notebook
    ├── scripts/
    │   ├── evaluate_metrics.py            # Benchmark Bicubic vs Pretrained vs Fine-Tuned models
    │   └── run_ablations.py               # Automated ablation study runner (A1–A6)
    ├── src/
    │   ├── checkpoints/                   # Weights, training curves, and metrics JSON/plots
    │   │   ├── metrics_results.json
    │   │   ├── metrics_barplot.png
    │   │   ├── metrics_table.png
    │   │   ├── training_curves.png
    │   │   └── training_history.json
    │   ├── demo_inference.py              # Demo visualization generator
    │   ├── download_sentinel.py           # Copernicus download instructions & tile validator
    │   ├── inference.py                   # End-to-end GeoTIFF SR inference & Hann blending
    │   ├── metrics.py                     # PSNR, SSIM, SAM, SRE, ERGAS, UIQ implementations
    │   ├── model.py                       # Real-ESRGAN wrapper & multi-band enhancement
    │   ├── pair_generation.py             # Physics-inspired synthetic LR/HR pair generator
    │   ├── preprocessing.py               # GeoTIFF validation, tiling, CRS/transform utilities
    │   ├── rrdbnet.py                     # Standalone RRDBNet architecture + grad checkpointing
    │   ├── spectral_fusion.py             # ChannelSpectralAttention & JointSpectralSR
    │   ├── temporal_fusion.py             # Multi-temporal ORB alignment & median fusion
    │   ├── train.py                       # Unified training & ablation pipeline
    │   └── uncertainty.py                 # 8-pass TTA ensemble & MC Dropout fallback
    └── tests/
        ├── test_blending_fixes.py         # Regression tests for tiling, Hann window & normalization
        ├── test_geo.py                    # CRS, affine transform & bounds preservation tests
        ├── test_metrics.py                # Metric correctness & edge-case tests
        └── test_pipeline.py               # Pair generation, uncertainty & end-to-end smoke tests
```

---

## 🔬 Methodology & Technical Details

### 1. Sentinel-2 Band Stack & Preprocessing

`src/preprocessing.py` expects stacked Sentinel-2 L2A GeoTIFFs in standard band order (while gracefully handling RGB-only or fewer-band GeoTIFFs):

| Band Index | Sentinel-2 Band | Center Wavelength (λ) | Native Resolution |
| :---: | :--- | :---: | :---: |
| `0` | **B02** (Blue) | 490 nm | 10 m |
| `1` | **B03** (Green) | 560 nm | 10 m |
| `2` | **B04** (Red) | 665 nm | 10 m |
| `3` | **B08** (NIR) | 842 nm | 10 m |
| `4` | **B05** (Red Edge 1) | 705 nm | 20 m (resampled to 10 m) |
| `5` | **B06** (Red Edge 2) | 740 nm | 20 m (resampled to 10 m) |
| `6` | **B07** (Red Edge 3) | 783 nm | 20 m (resampled to 10 m) |
| `7` | **B11** (SWIR 1) | 1610 nm | 20 m (resampled to 10 m) |
| `8` | **B12** (SWIR 2) | 2190 nm | 20 m (resampled to 10 m) |

Large scenes are tiled into overlapping patches (default `patch_size=512`, `overlap=64`, `min_valid_fraction=0.5`), preserving each patch's exact `rasterio.transform.Affine` window transform and CRS.

### 2. Synthetic Training Pair Generation

Because paired 10 m Sentinel-2 and sub-4 m commercial imagery is not freely available at scale, `src/pair_generation.py` applies a controlled, physically motivated degradation pipeline to 10 m HR tiles to synthesize 40 m-equivalent LR inputs:

1. **Gaussian Blur** ($\sigma \sim U[0.5, 1.5]$) — simulates sensor Point Spread Function (PSF) blur.
2. **4× Bicubic Downsampling** — models spatial pixel mixing.
3. **Additive Gaussian Noise** ($\sigma \sim U[1, 5] / 255$) — simulates sensor readout noise.
4. **JPEG Compression** ($\text{quality} \sim U[75, 95]$) — models transmission/storage compression artifacts.

### 3. Super-Resolution Backbone & Joint Spectral Modeling

- **Base Generator**: `RRDBNet` (23 Residual-in-Residual Dense Blocks) initialized from `RealESRGAN_x4plus.pth` (auto-downloaded on first run).
- **Independent Multi-Band Fallback**: RGB bands (`B04, B03, B02`) are processed as a 3-channel tensor; non-RGB bands are processed via batched grayscale-to-3-channel passes.
- **Joint Spectral Modeling (`--joint-spectral`)**: `JointSpectralSR` (`src/spectral_fusion.py`) pairs the base generator with `ChannelSpectralAttention` (dual global average + max pooling MLP) and `JointSpectralFusionModule` (1×1 spectral mixing + spatial-spectral residual refinement) so all bands inform each other during reconstruction.

### 4. Multi-Temporal Fusion

`src/temporal_fusion.py` provides multi-pass Sentinel-2 fusion:
- **`align_temporal_stack()`**: Extracts ORB keypoints (up to 5,000 features), computes homography matrices, and warps temporal observations into a unified reference frame.
- **Cloud/Haze Removal & Change Detection**: Combines aligned passes via temporal median filtering to suppress transient clouds/haze and highlight structural changes.

### 5. Composite Loss Suite & Training Optimizations

`src/train.py` unifies training and ablation experiments with the following configurable losses and optimizations:

| Loss Term | CLI Flag | Purpose |
| :--- | :--- | :--- |
| **L1 Pixel Loss** | `--lambda-l1` | Pixel-wise radiometric accuracy |
| **VGG16 Perceptual** | `--lambda-perceptual` | Natural texture & structural realism |
| **Differentiable SAM** | `--lambda-sam` | Preserves inter-band spectral angles ($\arccos\frac{u \cdot v}{\|u\|\|v\|}$) |
| **Sobel Edge Gradient** | `--lambda-edge` | Sharpens roads, rooftops, and linear infrastructure |
| **2D FFT Frequency** | `--lambda-freq` | Recovers high-frequency spatial micro-textures |
| **Multi-Scale Pyramid** | `--lambda-multiscale` | Enforces consistency across spatial scales |

Training features include:
- **`WarmupCosineAnnealingLR`**: Linear warmup followed by cosine decay.
- **D4 Dihedral Augmentation**: 8-way overhead rotation and reflection invariance.
- **Memory & Stability**: AMP FP16 mixed precision, gradient checkpointing, gradient clipping (`max_norm=1.0`), early stopping, and wall-clock time budgeting.

### 6. Geo-Aware Tiled Inference & Seamless Blending

`src/inference.py` reconstructs full scenes without tiling seams:
- Disables `RealESRGANer`'s internal tiling (`tile=0`) to prevent double-seam artifacts.
- Builds a 2D Hann window (`_make_hann_window(h, w)`) from each patch's *actual* output dimensions (handling clipped boundary patches cleanly).
- Supports optional edge-aware post-sharpening (`edge_aware_sharpen` / `unsharp_mask`).
- Scales the output GeoTIFF affine transform by $1 / \text{scale}$ (`1/4`) so every pixel maps to its true geographic coordinate.

### 7. Uncertainty Quantification (TTA Ensemble)

`src/uncertainty.py` applies **8 geometric augmentations** (identity, 90°/180°/270° rotations, horizontal/vertical/diagonal flips), runs SR on each, inverts the transform, and computes:
- **Mean SR prediction** (ensemble-averaged output)
- **Per-pixel variance map** (rendered as a viridis heatmap; high variance highlights regions where the model is hallucinating fine detail)

*(An MC Dropout fallback is also provided in `src/uncertainty.py` for architectures with Dropout layers.)*

### 8. Remote Sensing Evaluation Metrics

`src/metrics.py` implements the full suite of remote sensing quality metrics:

| Metric | Full Name | Direction | Description |
| :--- | :--- | :---: | :--- |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | Higher ↑ | Overall pixel-level fidelity |
| **SSIM** | Structural Similarity Index | Higher ↑ | Per-band structural & luminance similarity |
| **SAM** | Spectral Angle Mapper (°) | Lower ↓ | Angular spectral distortion (float64 with exact-equality short-circuit) |
| **SRE** | Signal-to-Reconstruction Error (dB) | Higher ↑ | Signal energy relative to reconstruction error |
| **ERGAS** | *Erreur Relative Globale Adimensionnelle de Synthèse* | Lower ↓ | Scale-aware global multi-band synthesis error |
| **UIQ** | Universal Image Quality Index | Higher ↑ | Correlation, luminance, and contrast distortion (Wang & Bovik 2002) |

---

## 📊 Benchmark Results

Evaluated via `scripts/evaluate_metrics.py` on **30 validation patches** at **4× scale** (`cuda`), recorded in [`srm-project/src/checkpoints/metrics_results.json`](srm-project/src/checkpoints/metrics_results.json):

| Model | PSNR (dB) ↑ | SSIM ↑ | SAM (°) ↓ | SRE (dB) ↑ | ERGAS ↓ | UIQ ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | **27.63** | **0.5136** | **0.1203** | **17.82** | **3.30** | **0.5997** |
| **Real-ESRGAN Pretrained** | 26.82 | 0.4580 | 0.9923 | 16.97 | 3.66 | 0.4763 |
| **Real-ESRGAN Fine-Tuned (Ours)** | 27.45 | 0.4791 | 0.2577 | 17.67 | 3.36 | 0.5256 |

**Key Takeaway**: Out-of-the-box `RealESRGAN_x4plus` (trained on natural photos) introduces severe spectral distortion on Sentinel-2 imagery (**0.992° SAM**). Fine-tuning on Sentinel-2 patches cuts spectral angle error by **~74%** (down to **0.258° SAM**) while improving PSNR (+0.64 dB), SSIM (+0.021), ERGAS (-0.31), and UIQ (+0.049) over the pretrained weights.

---

## 🛠️ Installation & Environment Setup

All commands below assume you are inside the `srm-project/` directory:

```bash
git clone https://github.com/AjayBora002/Depth-Wizard.git
cd Depth-Wizard/srm-project
```

### Option A: Windows with NVIDIA GPU (Recommended)

```powershell
# Automated setup scripts are provided:
.\setup_env.ps1
# or:
.\setup_gpu.bat
```

Or manually:

```bash
python -m venv .venv
.venv\Scripts\activate

# 1. Install PyTorch with CUDA 12.1 first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install setuptools (distutils shim for Python 3.12+) and requirements
pip install setuptools
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch, rasterio, streamlit; print('OK | CUDA:', torch.cuda.is_available())"
```

### Option B: Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install setuptools
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

Run all commands from `srm-project/`.

### 1. Acquire & Validate Sentinel-2 Data

```bash
# Print step-by-step Copernicus Data Space download instructions
python src/download_sentinel.py

# After placing GeoTIFFs in data/raw/, validate CRS, bands, and dimensions:
python src/download_sentinel.py --validate-only
```

### 2. Generate Synthetic Training Pairs

```bash
python src/pair_generation.py --raw-dir data/raw --out-dir data/synthetic_pairs --scale 4
```

### 3. Fine-Tune the Model

```bash
# Full composite loss (L1 + VGG16 Perceptual + SAM + Edge + FFT + Multi-Scale)
python src/train.py --ablation full --pairs-dir data/synthetic_pairs --epochs 50 --batch-size 4 --crop-size 128

# Full composite loss + Joint Multi-Band Spectral Modeling
python src/train.py --ablation joint_spectral --joint-spectral --pairs-dir data/synthetic_pairs --epochs 50
```

> **No local GPU?** Use the self-contained Google Colab notebook at [`srm-project/notebooks/colab_finetune.ipynb`](srm-project/notebooks/colab_finetune.ipynb).

### 4. Run Ablation Studies

Reproduce controlled loss-term comparisons (A1–A6) individually or sequentially:

```bash
# Individual presets: l1_only | with_perceptual | with_sam | with_edge | with_freq | joint_spectral
python src/train.py --ablation with_sam --epochs 50

# Run all 6 ablations sequentially
python scripts/run_ablations.py --all --epochs 30 --output-root data/ablations/
```

### 5. Run Super-Resolution Inference

```bash
# Using pretrained weights (auto-downloaded on first run)
python src/inference.py --input data/raw/your_tile.tif --output-dir data/outputs/

# Using fine-tuned checkpoint
python src/inference.py \
  --input data/raw/your_tile.tif \
  --checkpoint src/checkpoints/model_finetuned_best.pth \
  --output-dir data/outputs/

# With Ground-Truth reference for full metric evaluation
python src/inference.py \
  --input data/raw/lr_tile.tif \
  --gt data/raw/hr_tile.tif \
  --checkpoint src/checkpoints/model_finetuned_best.pth \
  --output-dir data/outputs/
```

Outputs saved to `data/outputs/`:
- Geo-referenced 4× SR GeoTIFF (`*_sr.tif`)
- TTA uncertainty heatmap PNG (`*_uncertainty.png`)
- JSON metrics & metadata report

### 6. Evaluate Metrics

```bash
python scripts/evaluate_metrics.py --pairs-dir data/training_pairs --max-patches 30
```

### 7. Launch the Web Dashboard

```bash
streamlit run dashboard/app.py
# Or on Windows:
.\run_dashboard.bat
```

Open `http://localhost:8501` in your browser.

### 8. One-Command Demo Workflow

```bash
python run_demo.py --mode all        # Generate pairs → Train → Create visualizations
python run_demo.py --mode generate   # Only generate synthetic pairs
python run_demo.py --mode train      # Only train model
python run_demo.py --mode demo       # Only create demo visualizations
```

---

## 🧪 Running Tests

All unit and integration tests run on synthetic in-memory data (no GPU or external Sentinel-2 tiles required):

```bash
cd srm-project
pytest tests/ -v
```

Test suites:
- [`tests/test_geo.py`](srm-project/tests/test_geo.py) — CRS, affine transform, and geographic bounds preservation
- [`tests/test_metrics.py`](srm-project/tests/test_metrics.py) — PSNR, SSIM, SAM, SRE, ERGAS, UIQ range and precision checks
- [`tests/test_pipeline.py`](srm-project/tests/test_pipeline.py) — Synthetic pair generation, TTA uncertainty, and end-to-end smoke tests
- [`tests/test_blending_fixes.py`](srm-project/tests/test_blending_fixes.py) — Regression tests for Hann-window blending and global normalization

---

## 🌐 Deployment (Streamlit Community Cloud)

See [`srm-project/DEPLOY.md`](srm-project/DEPLOY.md) for the full deployment walkthrough. Summary:

1. Use `srm-project/requirements_cloud.txt` (CPU-only PyTorch + GitHub `basicsr`) as `requirements.txt`.
2. Ensure `packages.txt` (`libgdal-dev`, `libgeos-dev`, `libproj-dev`, `gdal-bin`) and `runtime.txt` (`python-3.11`) are present.
3. On [share.streamlit.io](https://share.streamlit.io/), set the main file path to `srm-project/dashboard/app.py` (or `dashboard/app.py` if deploying `srm-project/` as root).
4. Note: Cloud inference runs on CPU (~30–60 s per 256×256 tile); reduce **Inference tile size** to `128` and **TTA augmentations** to `2` in the sidebar for faster response.

---

## 🔧 Engineering Fixes & Lessons Learned

Documented in [`srm-project/BUG_FIXES.md`](srm-project/BUG_FIXES.md):

1. **Eliminated Double-Seam Artifacts (`BUG-1`)**: Disabled `RealESRGANer`'s internal auto-tiling (`tile=0`) so it does not conflict with the pipeline's Hann-window tiling.
2. **Dimension-Aware Hann Windows (`BUG-2`)**: Built the 2D Hann window from each patch's *actual* SR output dimensions rather than a fixed `crop_size × crop_size`, preventing bright/dark seams on clipped edge tiles.
3. **Global Dtype Normalization (`BUG-3`)**: Replaced per-tile min/max stretching with dtype-based global scaling (`uint16 → /65535.0`, `float32 → 1.0`), eliminating checkerboard brightness grids across scenes.
4. **SAM Float64 Precision Fix**: Fixed float32 `arccos` precision drift where identical images returned `~0.006°` instead of `0.0°` by computing SAM in `float64` with an exact-equality short-circuit.

---

## ⚠️ Known Limitations

1. **Synthetic Training Pairs**: The model is fine-tuned on synthetically degraded (blur + 4× bicubic + noise + JPEG) Sentinel-2 pairs; domain shift may occur on real-world native low-resolution sensors.
2. **Resolution Terminology**: "2.5 m-equivalent" refers to the 4× upscaled pixel grid, not physical 2.5 m optical Ground Sampling Distance (GSD); true resolving power remains bounded by the 10 m input signal.
3. **RGB-Priority Pretrained Weights**: Unless `--joint-spectral` is enabled, non-RGB bands (NIR/SWIR/Red Edge) share RGB-trained weights via grayscale replication.
4. **TTA ≠ Bayesian Posterior**: TTA variance measures geometric augmentation sensitivity rather than true Bayesian posterior uncertainty.
5. **GAN Discriminator**: `train.py` optimizes the generator with composite reconstruction/spectral/perceptual losses rather than a full adversarial discriminator loop.

---

## 🔮 Future Roadmap

- [ ] Benchmark Transformer SR architectures (**SwinIR**, **HAT**) against Real-ESRGAN
- [ ] Incorporate real paired HR reference data (e.g., **CARTOSAT-3** / **Pleiades**)
- [ ] Add full adversarial discriminator training for sharper high-frequency textures
- [ ] Integrate **Sentinel-1 SAR** priors for urban/building height disambiguation
- [ ] Export models to **ONNX / TensorRT** for edge and embedded deployment

---

## 📦 Dependencies

| Package | Version | Role |
| :--- | :--- | :--- |
| `torch` / `torchvision` | ≥ 2.1.0 / ≥ 0.16.0 | Model training & inference |
| `basicsr` / `realesrgan` | ≥ 1.4.2 / ≥ 0.3.0 | RRDBNet & Real-ESRGAN wrapper |
| `rasterio` / `pyproj` / `shapely` | ≥ 1.3.9 / ≥ 3.6.1 / ≥ 2.0.3 | GeoTIFF I/O, CRS & spatial transforms |
| `scikit-image` / `opencv-python-headless` | ≥ 0.22.0 / ≥ 4.9.0 | Image processing, PSNR & SSIM |
| `streamlit` / `folium` / `plotly` | ≥ 1.31.0 / ≥ 0.15.1 / ≥ 5.18.0 | Interactive dashboard, geo-map & charts |
| `pytest` | ≥ 8.0.0 | Automated test suite |

---


