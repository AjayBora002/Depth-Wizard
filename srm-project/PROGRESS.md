# PROGRESS.md — SIH26142 Super-Resolution Mapping
## Running build status (updated after each phase)

---

## ✅ Phase 0 — Project Scaffolding  COMPLETE

**Done:**
- Full directory structure created: `data/raw`, `data/synthetic_pairs/{lr,hr}`, `data/outputs`, `src/checkpoints`, `dashboard/`, `notebooks/`, `tests/`
- `requirements.txt` with pinned dependencies (torch, rasterio, realesrgan, basicsr, streamlit, folium, scikit-image, pytest, etc.)
- `.gitignore` configured (large files, weights, raw data excluded)
- All `__init__.py` and `.gitkeep` stubs created

**Environment setup (run once):**
```bash
cd srm-project
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## ✅ Phase 1 — Data Acquisition  COMPLETE (instructions ready, tiles needed)

**Done:**
- `src/download_sentinel.py` — prints complete manual signup + download instructions
- `src/preprocessing.py` — `validate_tile()`, `tile_image()`, `extract_rgb_preview()` implemented
- `src/preprocessing.py` — `validate_all_tiles()` scans `data/raw/` and reports each tile

**Waiting on user action:**
- Create free Copernicus Data Space account at https://dataspace.copernicus.eu/
- Download 5-10 Sentinel-2 L2A tiles (urban, agricultural, coastal) as GeoTIFF
- Place tiles in `data/raw/`
- Run: `python src/download_sentinel.py --validate-only` to confirm

**IMPORTANT — Before continuing phases 2-3:**
Tiles must be present in `data/raw/` for pair generation and training to proceed.
The remaining phases can be code-verified without tiles but cannot produce real outputs.

---

## ✅ Phase 2 — Synthetic Pair Generation  COMPLETE (code done, needs tiles)

**Done:**
- `src/pair_generation.py` — full degradation pipeline:
  1. Gaussian blur (σ ~ U[0.5, 1.5]) — PSF simulation
  2. 4× bicubic downscale
  3. Gaussian noise (σ/255 ~ U[1, 5]) — sensor noise
  4. JPEG compression artifacts (quality ~75-95)
- `SyntheticPairDataset` PyTorch Dataset with train/val split
- Output: `.npy` files in `data/synthetic_pairs/lr/` and `data/synthetic_pairs/hr/`

**Design decision (explicitly disclosed):**
True paired LR/HR satellite data is not freely available at scale. Controlled downsampling
of existing Sentinel-2 10m tiles creates synthetic training pairs. This is a documented
limitation, not a hidden shortcut. The original 10m tile is the "high-resolution ground truth."

**To run (after tiles are in data/raw/):**
```bash
python src/pair_generation.py --raw-dir data/raw --out-dir data/synthetic_pairs --scale 4
```

---

## ✅ Phase 3 — Model Setup & Fine-Tuning  COMPLETE (code done)

**Done:**
- `src/model.py` — Real-ESRGAN wrapper:
  - Auto-downloads `RealESRGAN_x4plus.pth` from GitHub releases on first run
  - `enhance_multiband()` handles Sentinel-2 multi-band (RGB via standard 3-ch ESRGAN, other bands via batched grayscale-to-3ch passes)
  - `joint_spectral` option integrated with `src/spectral_fusion.py`
  - `load_generator_for_training()` extracts bare RRDB generator for fine-tuning
  - `save_generator_checkpoint()` saves in RealESRGAN-compatible format
- `src/spectral_fusion.py` — **Joint Multi-Band Spectral Modeling**:
  - `ChannelSpectralAttention` dual-pooling inter-band correlation modeling
  - `JointSpectralRefiner` post-SR cross-spectral residual refinement block
  - `JointSpectralSR` wrapper pairing base generator with multi-band joint modeling
- `src/train.py` — **Unified training script** (consolidated with train_enhanced.py):
  - L1 + VGG16 Perceptual + SAM + Sobel Edge + FFT Frequency + Multi-Scale Pyramid loss
  - Linear warmup + Cosine Annealing learning rate schedule
  - D4 dihedral symmetry data augmentation
  - Memory-efficient gradient checkpointing support
  - Mixed precision training (AMP) + gradient clipping
  - SSIM + PSNR + SAM validation and early stopping
  - Clean ablation configuration system (`--ablation` presets) with zero overhead when lambda=0
- `scripts/run_ablations.py` — automated runner for controlled comparison studies
- `notebooks/colab_finetune.ipynb` — self-contained Colab backup

---

## ✅ Phase 4 — Metrics  COMPLETE

**Done:**
- `src/metrics.py`:
  - `psnr()` — scikit-image `peak_signal_noise_ratio`
  - `ssim()` — scikit-image `structural_similarity` (per-band averaged)
  - `sam()` — custom Spectral Angle Mapper:
    - `per-pixel SAM = arccos(dot(u,v) / (||u|| * ||v||))`
    - `scene SAM = mean over valid pixels (degrees)`
  - `sre()` — Signal to Reconstruction Error ratio (dB)
  - `ergas()` — Erreur Relative Globale Adimensionnelle de Synthèse
  - `uiq()` — Universal Image Quality Index (Wang & Bovik 2002)
  - `evaluate_pair()` — returns all 6 metrics
  - `evaluate_directory()` — batch evaluation over directories
- All metrics verified with unit tests (identical pairs, degraded pairs, range checks)

---

## ✅ Phase 5 — Uncertainty Quantification  COMPLETE

**Done:**
- `src/uncertainty.py`:
  - **Primary: TTA ensemble** (8 geometric augmentations: identity, 3 rotations, 4 flips)
  - `tta_ensemble()` — returns `(mean_sr, uncertainty_map)` as float32 arrays
  - `uncertainty_to_heatmap()` — viridis colormap, percentile-stretched for display
  - `save_uncertainty_heatmap()` — saves 3-panel figure (SR | blended | heatmap)
  - **Fallback: MC Dropout** — provided but not default (RRDBNet has no Dropout layers)

---

## ✅ Phase 6 — Dashboard  COMPLETE (functional locally)

**Done:**
- `dashboard/app.py` — full Streamlit app:
  - Upload GeoTIFF / select from samples
  - Run inference with progress bar
  - Before/after slider (`streamlit-image-comparison`)
  - PSNR/SSIM/SAM metric cards (with status colour: green/orange/grey)
  - Uncertainty heatmap display
  - Folium geo-map showing tile bounds on basemap
  - Batch upload (ZIP → process all → download ZIP)
  - Training history Plotly chart
  - About tab with full methodology disclosure

**To run:**
```bash
streamlit run dashboard/app.py
```

---

## ✅ Phase 7 — Tests  COMPLETE

**Test files:**
- `tests/test_geo.py` — geo-referencing preservation (CRS, affine, bounds)
- `tests/test_metrics.py` — PSNR/SSIM/SAM sanity (identical → perfect, noisy → degraded)
- `tests/test_pipeline.py` — pair generation, uncertainty, end-to-end smoke test

**All tests use synthetic in-memory data — no GPU or real tiles required.**

**To run:**
```bash
pytest tests/ -v
```

---

## ✅ Phase 8 — Documentation  COMPLETE

**Done:**
- `README.md` — complete with setup, run instructions, architecture, methodology, limitations
- `PROGRESS.md` — this file

---

## ✅ Phase 9 — Deployment Prep  COMPLETE

**Done:**
- `packages.txt` — system apt packages for Streamlit Cloud (libgdal-dev, libgeos-dev, libproj-dev, gdal-bin)
- `runtime.txt` — pins Python 3.11 for Streamlit Cloud stability
- `requirements_cloud.txt` — CPU-only PyTorch + basicsr from GitHub (avoids Python 3.12+ exec() issue)
- `DEPLOY.md` — complete step-by-step deployment guide
- `git init` + first commit: 29 files, 4802 insertions

**Bug fixes applied:**
- `setup.cfg`: `[pytest]` → `[tool:pytest]` (pytest 9+ compatibility)
- `tests/__init__.py`, `src/__init__.py`, `dashboard/__init__.py`: re-encoded UTF-16 → UTF-8 (null bytes fixed)
- `src/metrics.py` SAM: float32 precision bug fixed (identical images returned ~0.006° instead of 0.0); now uses float64 with exact equality short-circuit
- `src/pair_generation.py` `SyntheticPairDataset.__getitem__`: cross-platform hr_path using `Path` methods
- `tests/test_pipeline.py` `test_evaluate_pair_on_pipeline_output`: sinusoidal test signal for guaranteed >15dB PSNR after bicubic

---

## 🔵 OVERALL STATUS

| Component | Status | Notes |
|-----------|--------|-------|
| Project structure | ✅ Done | |
| Sentinel-2 download instructions | ✅ Done | Waiting for user to download tiles |
| Preprocessing (geo-aware tiling) | ✅ Done | |
| Synthetic pair generation | ✅ Done | Needs tiles in data/raw/ |
| Model (Real-ESRGAN wrapper) | ✅ Done | Auto-downloads weights |
| Local fine-tuning script | ✅ Done | GPU required |
| Colab fine-tuning notebook | ✅ Done | Backup option |
| PSNR/SSIM/SAM metrics | ✅ Done | All 40 tests passing |
| Uncertainty (TTA ensemble) | ✅ Done | |
| Streamlit dashboard | ✅ Running | http://localhost:8501 |
| Tests | ✅ **40/40 PASSING** | pytest 9.1.1, Python 3.14.5 |
| README | ✅ Done | |
| Git repository | ✅ Done | Initial commit: 29 files |
| Streamlit Cloud deploy files | ✅ Done | packages.txt, runtime.txt, requirements_cloud.txt |
| DEPLOY.md guide | ✅ Done | Step-by-step with troubleshooting |

---

## 📋 USER ACTION ITEMS (Remaining)

### To deploy to Streamlit Community Cloud:
1. Create GitHub repo and push: `git remote add origin https://github.com/YOUR_USERNAME/srm-project.git && git push -u origin main`
2. Copy cloud requirements: `cp requirements_cloud.txt requirements.txt && git add requirements.txt && git commit -m "cloud requirements" && git push`
3. Go to https://share.streamlit.io → New App → set Main file: `dashboard/app.py`
4. See `DEPLOY.md` for full instructions

### To train with real satellite data:
1. **Create Copernicus account + download tiles** → `data/raw/`
2. **Validate tiles**: `python src/download_sentinel.py --validate-only`
3. **Generate pairs**: `python src/pair_generation.py`
4. **Fine-tune**: `python src/train.py --epochs 50`
5. **Run inference**: `python src/inference.py --input data/raw/yourfile.tif`
6. **Launch dashboard**: `streamlit run dashboard/app.py`
7. **Run tests**: `pytest tests/ -v`

---

## 🏷️ Known Limitations (disclosed)

1. **Synthetic training pairs**: Not real sensor-paired LR/HR data. Real-world SR quality may differ from test-set metrics.
2. **Limited training epochs**: 50 epochs may not fully converge; more epochs → better fine-tuning with more compute.
3. **RGB-only fine-tuning**: train.py uses RGB bands (3-ch) for compatibility with RRDBNet; non-RGB bands use the same weights (may be sub-optimal for SWIR etc.).
4. **TTA uncertainty**: Not the same as full Bayesian uncertainty; it measures augmentation sensitivity, not posterior variance.
5. **Validation scope**: Metrics computed on synthetic test set (held-out from same distribution as training); real-world generalization not yet validated on independent sensor data.
6. **Cloud inference speed**: Streamlit Community Cloud has no GPU — CPU-only inference takes ~30–60s per tile.
