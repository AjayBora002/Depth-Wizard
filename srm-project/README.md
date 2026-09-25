# SRM — Deep Learning Based Super Resolution Mapping
## SIH26142 | Sponsored by NTRO

> **Input**: Sentinel-2 10m multi-band GeoTIFF
> **Output**: 4× super-resolved 2.5m-equivalent GeoTIFF + uncertainty heatmap + spectral metrics

---

## 📋 Problem Statement Summary

**SIH26142** (Smart India Hackathon 2026): Develop a deep learning–based Super Resolution Mapping (SRM) system that enhances medium-resolution satellite imagery (Sentinel-2, 10m) to produce higher-resolution outputs (<4m equivalent), while:
- Preserving geographic coordinates (geo-referencing)
- Maintaining spectral consistency across bands
- Quantifying uncertainty in reconstructed detail
- Providing objective metric validation (PSNR, SSIM, SAM)
- Offering a usable web dashboard

---

## 🏗️ Architecture

```
Sentinel-2 GeoTIFF (10m, multi-band)
           │
           ▼
   preprocessing.py
   Tile into 256×256 patches, preserve CRS + affine transform per patch
           │
           ▼         [Training path (once)]
   pair_generation.py ──────────────────► data/synthetic_pairs/{lr,hr}
   Degradation: blur → bicubic↓4× → noise → JPEG                │
                                                                  ▼
                                                           train.py
                                                    Real-ESRGAN fine-tuning
                                                    (L1 + perceptual loss)
                                                    → src/checkpoints/*.pth
           │
           ▼         [Inference path]
   model.py — RealESRGAN x4plus (pretrained + fine-tuned)
   RGB bands: standard 3-ch ESRGAN
   Other bands: grayscale→3ch→ESRGAN→single channel
           │
    ┌──────┴──────┐
    ▼             ▼
SR output    uncertainty.py
(GeoTIFF)    TTA ensemble (8 augmentations)
             → per-pixel variance map
    │             │
    └──────┬──────┘
           ▼
      metrics.py
  PSNR / SSIM / SAM (vs GT if available)
  Hann-window blending for tile reconstruction
           │
           ▼
  inference.py — saves SR GeoTIFF + uncertainty PNG + JSON report
           │
           ▼
  dashboard/app.py — Streamlit: upload → infer → visualise → geo-map
```

---

## 🚀 Quick Start

### 1. Environment Setup (Windows with GPU)

```bash
git clone https://github.com/<your-username>/srm-project.git
cd srm-project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install -r requirements.txt

# Verify
python -c "import torch, rasterio, streamlit; print('OK, CUDA:', torch.cuda.is_available())"
```

### 2. Get Sentinel-2 Data

```bash
python src/download_sentinel.py   # Prints step-by-step Copernicus download instructions
# Follow instructions, place GeoTIFFs in data/raw/, then:
python src/download_sentinel.py --validate-only
```

### 3. Generate Training Pairs

```bash
python src/pair_generation.py --raw-dir data/raw --out-dir data/synthetic_pairs
```

### 4. Fine-Tune (optional but recommended)

```bash
python src/train.py \
  --pairs-dir data/synthetic_pairs \
  --epochs 50 \
  --batch-size 4 \
  --crop-size 128
```

### 5. Run Inference

```bash
# Single tile (pretrained weights auto-downloaded on first run)
python src/inference.py --input data/raw/your_tile.tif --output-dir data/outputs/

# With fine-tuned checkpoint
python src/inference.py \
  --input data/raw/your_tile.tif \
  --checkpoint src/checkpoints/model_finetuned_best.pth \
  --output-dir data/outputs/

# With ground-truth for metrics
python src/inference.py \
  --input data/raw/lr_tile.tif \
  --gt data/raw/hr_tile.tif \
  --output-dir data/outputs/
```

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
```
Navigate to `http://localhost:8501`

### 7. Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Metrics Results
 
> **Note**: Evaluated on 4× super-resolution vs high-resolution reference tiles.
> Includes full remote sensing metric suite required by literature (RS-ESRGAN, DSen2).
 
| Model | PSNR (dB) ↑ | SSIM ↑ | SAM (°) ↓ | SRE (dB) ↑ | ERGAS ↓ | UIQ ↑ |
|-------|------------|--------|-----------|------------|---------|-------|
| Bicubic baseline | — | — | — | — | — | — |
| Real-ESRGAN pretrained | — | — | — | — | — | — |
| Real-ESRGAN fine-tuned | — | — | — | — | — | — |
| + Joint Spectral Modeling | — | — | — | — | — | — |
 
---

## 🔬 SAM-Loss & Loss Suite Ablation Study

Use `scripts/run_ablations.py` or `src/train.py --ablation <preset>` to reproduce controlled comparisons:

```bash
# A1: Pure pixel reconstruction baseline (L1 only)
python src/train.py --ablation l1_only --epochs 50

# A2: + Perceptual realism (L1 + VGG16)
python src/train.py --ablation with_perceptual --epochs 50

# A3: + Spectral preservation (L1 + Perceptual + SAM)
python src/train.py --ablation with_sam --epochs 50

# A4: + Linear infrastructure gradients (L1 + Perceptual + SAM + Sobel Edge)
python src/train.py --ablation with_edge --epochs 50

# A5: + High-frequency micro-textures (L1 + Perceptual + SAM + Edge + FFT Frequency)
python src/train.py --ablation with_freq --epochs 50

# A6: Full Composite Loss + Joint Multi-Band Spectral Modeling
python src/train.py --ablation joint_spectral --joint-spectral --epochs 50

# Or run the entire ablation suite sequentially:
python scripts/run_ablations.py --all --epochs 30
```

Setting any `--lambda-*` to 0.0 disables that loss term with zero compute and memory overhead. All run configurations are logged in `training_history.json`.

---

## 🧪 Methodology

### Super-Resolution Model
**Real-ESRGAN x4plus** (xinntao, MIT License)
- Architecture: RRDB (Residual-in-Residual Dense Block), 23 blocks
- Input: 3-channel uint8/float32 image, Output: 4× upscaled 3-channel
- Gradient Checkpointing: Supported via `src/rrdbnet.py` to train larger patches in VRAM.

### Multi-Band & Joint Spectral Modeling (`src/spectral_fusion.py`)
Standard photo-SR processes bands independently. This repository provides:
1. **Independent Fallback**: RGB processed as 3-ch, other bands processed via batched grayscale passes.
2. **Joint Spectral Modeling (`--joint-spectral`)**: Inspired by DSen2 (Lanaras et al., 2018), `JointSpectralRefiner` couples all spectral bands together using dual-pooling spatial-spectral attention and 1×1 spectral mixing convolutions, enabling cross-band gradient flow directly into the SAM loss.

### Synthetic Training Pairs (Disclosed Design Decision)
> ⚠️ **This system uses synthetic, not real-sensor, LR/HR pairs.**
>
> True paired low-resolution / high-resolution satellite datasets at Sentinel-2 / sub-4m
> resolution are not freely available at scale. We apply a degradation model to existing
> Sentinel-2 10m tiles:
>
> 1. **Gaussian blur** — σ ~ U[0.5, 1.5] — simulates PSF blur
> 2. **4× bicubic downscale** — pixel mixing to ~40m equivalent
> 3. **Gaussian noise** — σ ~ U[1, 5]/255 — sensor noise
> 4. **JPEG compression** — quality ~ U[75, 95] — compression artifacts

### Uncertainty Quantification
**Batched Test-Time Augmentation (TTA) Ensemble (`src/uncertainty.py`)**:
- 8 geometric augmentations (identity, 90°/180°/270° rotations, flips)
- Full multi-band support (RGB + batched grayscale passes) without dimension mismatch
- Per-pixel variance across outputs = uncertainty estimate heatmap

### Remote Sensing Evaluation Metrics (`src/metrics.py`)
| Metric | Description | Target |
|--------|-------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | Higher ↑ (>30 dB) |
| **SSIM** | Structural Similarity Index Measure | Higher ↑ (>0.85) |
| **SAM** | Spectral Angle Mapper (degrees, Yuhas et al. 1992) | Lower ↓ (<2.0°) |
| **SRE** | Signal to Reconstruction Error Ratio (dB) | Higher ↑ |
| **ERGAS** | Relative Dimensionless Global Error of Synthesis (Wald et al.) | Lower ↓ (<3.0) |
| **UIQ** | Universal Image Quality Index (Wang & Bovik 2002) | Higher ↑ (~1.0) |

---

## 🌐 Dashboard Deployment (Streamlit Community Cloud)

1. Push to GitHub: `git push origin main`
2. Go to: https://share.streamlit.io/
3. Click "New app" → select your repo
4. Main file: `dashboard/app.py`
5. Set secrets if needed (none required for this project)
6. Click "Deploy" — free, public URL generated

> **Note**: Model weights (>60MB) are downloaded at first run. On Streamlit Cloud,
> use `@st.cache_resource` (already implemented) to avoid re-downloading each session.
> Consider hosting weights on Hugging Face Hub for faster cloud cold starts.

---

## ⚠️ Known Limitations

1. **Synthetic training data**: Model fine-tuned on synthetic (bicubic-downsampled) pairs; real-world performance on genuine low-resolution sensors may differ.
2. **Limited validation scope**: Tested on Sentinel-2 10m tiles from specific geographic regions; generalisation to other biomes or sensors not validated.
3. **RGB-priority fine-tuning**: Non-RGB bands use the same RGB-trained weights; spectral accuracy on SWIR/NIR may be lower than on visible bands.
4. **TTA ≠ Bayesian uncertainty**: TTA variance measures augmentation sensitivity, not true posterior uncertainty. A proper Bayesian approach requires Dropout or ensembling distinct models.
5. **Resolution terminology**: "2.5m-equivalent" refers to the pixel pitch after 4× upscaling, not the ground sampling distance of a physical 2.5m sensor. True resolving power is bounded by the original 10m imagery.
6. **Fine-tuning scope**: 50 epochs on synthetic pairs — not exhaustive. More data, more epochs, and discriminator-based adversarial training would improve results.

---

## 🔮 Future Work

- [ ] Benchmark SwinIR and HAT (Hybrid Attention Transformer) against Real-ESRGAN
- [ ] Acquire real paired training data from CARTOSAT-3 (0.3m) / Pleiades as HR reference
- [ ] Add discriminator adversarial training (full ESRGAN training loop) for sharper textures
- [ ] Multi-temporal super-resolution (exploit multiple passes over same area)
- [ ] Tile-level confidence scores for automatic quality control
- [ ] ONNX export for deployment on edge devices / embedded systems
- [ ] Sentinel-1 SAR integration for building height / urban density priors

---

## 📦 Dependencies (key)

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.1.0 | Model inference + training |
| basicsr | ≥1.4.2 | RRDB architecture |
| realesrgan | ≥0.3.0 | RealESRGANer wrapper |
| rasterio | ≥1.3.9 | GeoTIFF I/O (bundles GDAL) |
| scikit-image | ≥0.22.0 | PSNR, SSIM |
| streamlit | ≥1.31.0 | Dashboard |
| folium | ≥0.15.1 | Geo map |
| plotly | ≥5.18.0 | Training charts |

---

## 📄 License

MIT License. Pretrained Real-ESRGAN weights: BSD 3-Clause (xinntao).
Sentinel-2 data: Copernicus Open Access Hub (free, CC BY-SA 3.0 IGO).

---

## 🙏 Acknowledgements

- xinntao/Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN)
- ESA Copernicus Programme (Sentinel-2 free data)
- SIH26142 problem statement (NTRO, Smart India Hackathon 2026)
