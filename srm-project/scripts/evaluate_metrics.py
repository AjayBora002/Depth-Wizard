"""
evaluate_metrics.py
===================
Full metrics evaluation for SIH26142 SRM project.

Run after training to compute PSNR, SSIM, SAM, ERGAS, SRE, UIQ
for three model variants: Bicubic / Pretrained ESRGAN / Fine-Tuned ESRGAN.

Usage (Colab cell):
    !python scripts/evaluate_metrics.py

Usage (local):
    python scripts/evaluate_metrics.py --pairs-dir data/training_pairs --max-patches 30
"""

from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import psnr, ssim, sam, sre, ergas, uiq
from src.model import load_realesrgan, enhance_multiband


# ─── helpers ─────────────────────────────────────────────────────────────────

def bicubic_upsample(lr: np.ndarray, scale: int = 4) -> np.ndarray:
    c, h, w = lr.shape
    out = []
    for b in range(c):
        band = (lr[b].clip(0, 1) * 255).astype(np.uint8)
        up = cv2.resize(band, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        out.append(up.astype(np.float32) / 255.0)
    return np.stack(out, axis=0)


def sr_with_model(model, lr: np.ndarray) -> np.ndarray:
    try:
        from src.model import _generator_from_upsampler
        gen = _generator_from_upsampler(model) if hasattr(model, 'model') else model
    except Exception:
        gen = model
    return enhance_multiband(model, lr.clip(0, 1), generator=gen)


def match_shape(sr: np.ndarray, hr: np.ndarray) -> np.ndarray:
    if sr.shape == hr.shape:
        return sr
    c, h, w = hr.shape
    out = []
    for b in range(c):
        src = sr[b] if b < sr.shape[0] else sr[0]
        out.append(cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, axis=0)


METRIC_DISPLAY = {
    'psnr':  ('PSNR (dB) up',  '{:.2f}'),
    'ssim':  ('SSIM up',        '{:.4f}'),
    'sam':   ('SAM (deg) dn',   '{:.3f}'),
    'ergas': ('ERGAS dn',       '{:.2f}'),
    'sre':   ('SRE (dB) up',   '{:.2f}'),
    'uiq':   ('UIQ up',         '{:.4f}'),
}
ROW_COLORS = ['#f8f9fa', '#e9ecef', '#d0f0c0']


def evaluate_model(name, val_lr, val_hr, model=None, scale=4):
    accum = {k: [] for k in METRIC_DISPLAY}
    for lr_f, hr_f in tqdm(zip(val_lr, val_hr), total=len(val_lr), desc=f'  [{name}]'):
        lr_np = np.load(str(lr_f)).astype(np.float32).clip(0, 1)
        hr_np = np.load(str(hr_f)).astype(np.float32).clip(0, 1)
        try:
            sr_np = bicubic_upsample(lr_np, scale) if model is None else sr_with_model(model, lr_np)
            sr_np = match_shape(sr_np, hr_np)
            for k, fn in [('psnr', psnr), ('ssim', ssim), ('sam', sam),
                           ('ergas', lambda a, b: ergas(a, b, scale=scale)),
                           ('sre', sre), ('uiq', uiq)]:
                v = float(fn(hr_np, sr_np))
                if not (np.isnan(v) or np.isinf(v)):
                    accum[k].append(v)
        except Exception as exc:
            print(f'    Warning: {lr_f.name}: {exc}')
    return {k: float(np.mean(v)) if v else float('nan') for k, v in accum.items()}


def save_table(results, out_path, n_patches, scale):
    models = list(results['models'].keys())
    headers = ['Model'] + [METRIC_DISPLAY[m][0] for m in METRIC_DISPLAY]
    rows = []
    for m_name, m_vals in results['models'].items():
        row = [m_name]
        for mk, (_, fmt) in METRIC_DISPLAY.items():
            v = m_vals.get(mk, float('nan'))
            row.append(fmt.format(v) if not np.isnan(v) else '--')
        rows.append(row)

    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor('#1a1a2e')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(models)):
        for j in range(len(headers)):
            tbl[i+1, j].set_facecolor(ROW_COLORS[min(i, 2)])
            if i == len(models)-1:
                tbl[i+1, j].set_text_props(fontweight='bold')
    plt.title(f'SIH26142 Sentinel-2 {scale}x SR Results  ({n_patches} val patches)',
              fontsize=13, fontweight='bold', pad=12)
    leg = [mpatches.Patch(facecolor=ROW_COLORS[i], label=(models[i] + (' (*Ours*)' if i==2 else '')))
           for i in range(len(models))]
    ax.legend(handles=leg, loc='lower right', fontsize=9, bbox_to_anchor=(1.0, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Table  -> {out_path}')


def save_barplot(results, out_path):
    models = list(results['models'].keys())
    mk_list = list(METRIC_DISPLAY.keys())
    x = np.arange(len(mk_list))
    width = 0.25
    colors = ['#6c757d', '#4a90d9', '#28a745']
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (m_name, m_vals) in enumerate(results['models'].items()):
        vals = [m_vals.get(mk, 0) for mk in mk_list]
        ax.bar(x + i * width, vals, width, label=m_name, color=colors[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([METRIC_DISPLAY[m][0] for m in mk_list], fontsize=11)
    ax.set_title('SIH26142 Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=600, bbox_inches='tight')
    plt.close()
    print(f'  Barplot-> {out_path}')


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs-dir',   default='data/training_pairs')
    parser.add_argument('--ckpt-best',   default='src/checkpoints/model_finetuned_best.pth')
    parser.add_argument('--ckpt-base',   default='src/checkpoints/RealESRGAN_x4plus.pth')
    parser.add_argument('--out-dir',     default='src/checkpoints')
    parser.add_argument('--max-patches', type=int, default=30)
    parser.add_argument('--scale',       type=int, default=4)
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    lr_dir = Path(args.pairs_dir) / 'lr'
    hr_dir = Path(args.pairs_dir) / 'hr'
    if not lr_dir.exists():
        print(f'ERROR: {lr_dir} not found. Run pair generation first.'); sys.exit(1)

    lr_files = sorted(lr_dir.glob('*.npy'))
    hr_files = sorted(hr_dir.glob('*.npy'))
    if not lr_files:
        print(f'ERROR: No .npy files in {lr_dir}'); sys.exit(1)

    n_total = len(lr_files)
    n_val   = max(1, int(n_total * 0.2))
    val_lr  = lr_files[-n_val:][:args.max_patches]
    val_hr  = hr_files[-n_val:][:args.max_patches]
    print(f'Val patches: {len(val_lr)} / {n_total}\n')

    # 1. Bicubic
    print('1/3  Bicubic Baseline')
    m_bic = evaluate_model('Bicubic', val_lr, val_hr, scale=args.scale)

    # 2. Pretrained
    print('\n2/3  Pretrained ESRGAN')
    m_pre_model = load_realesrgan('x4plus', args.ckpt_base, device, half=torch.cuda.is_available())
    m_pre = evaluate_model('Pretrained', val_lr, val_hr, model=m_pre_model, scale=args.scale)
    del m_pre_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 3. Fine-tuned
    print('\n3/3  Fine-Tuned ESRGAN')
    ckpt = args.ckpt_best if Path(args.ckpt_best).exists() else args.ckpt_base
    if not Path(args.ckpt_best).exists():
        print(f'  WARNING: fine-tuned ckpt not found, using pretrained.')
    m_ft_model = load_realesrgan('x4plus', ckpt, device, half=torch.cuda.is_available())
    m_ft = evaluate_model('Fine-Tuned', val_lr, val_hr, model=m_ft_model, scale=args.scale)
    del m_ft_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    results = {
        'n_val_patches': len(val_lr),
        'scale': args.scale,
        'device': str(device),
        'models': {
            'Bicubic Baseline':              m_bic,
            'Real-ESRGAN Pretrained':        m_pre,
            'Real-ESRGAN Fine-Tuned (Ours)': m_ft,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / 'metrics_results.json'
    out_json.write_text(json.dumps(results, indent=2))
    print(f'\n JSON   -> {out_json}')

    # Console table
    mk_list = list(METRIC_DISPLAY.keys())
    w = 38 + 12 * len(mk_list)
    print('\n' + '=' * w)
    header = f"{'Model':<38}" + ''.join(f'{METRIC_DISPLAY[m][0]:>12}' for m in mk_list)
    print(header)
    print('-' * w)
    for m_name, m_vals in results['models'].items():
        row = f'{m_name:<38}' + ''.join(
            METRIC_DISPLAY[mk][1].format(m_vals.get(mk, float('nan'))).rjust(12)
            for mk in mk_list)
        print(row)
    print('=' * w)

    print('\nGenerating figures...')
    save_table(results, out_dir / 'metrics_table.png', len(val_lr), args.scale)
    save_barplot(results, out_dir / 'metrics_barplot.png')

    if not args.no_download:
        try:
            from google.colab import files as colab_files
            print('\nDownloading from Colab...')
            for f in [out_json, out_dir / 'metrics_table.png', out_dir / 'metrics_barplot.png']:
                if f.exists():
                    print(f'  down {f.name}')
                    colab_files.download(str(f))
        except ImportError:
            print('(Not in Colab - results in src/checkpoints/)')

    print('\nDone! Use metrics_results.json to fill your paper table.')


if __name__ == '__main__':
    main()
