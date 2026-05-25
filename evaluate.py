"""
evaluate.py — Full-dataset evaluation for SIFT+IPVO watermarking.

Usage:
    python evaluate.py                          # all images, all attacks
    python evaluate.py --subset 5              # quick smoke test (first N images)
    python evaluate.py --attacks rotate_15 scale_0.75
    python evaluate.py --no-baseline           # skip baseline (faster)

Outputs (saved to output/evaluation/):
    results_sift.csv        – per-image, per-attack metrics (SIFT pipeline)
    results_baseline.csv    – per-image, per-attack metrics (baseline)
    summary_bar.png         – grouped bar: SIFT vs Baseline BER per attack (avg across images)
    summary_heatmap.png     – heatmap: BER per image × attack (SIFT pipeline)
    summary_heatmap_bl.png  – heatmap: BER per image × attack (baseline)
    stats.json              – aggregate stats (mean, std, min, max per metric × attack)
    failed_images.txt       – images that failed to embed/extract
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from src.watermark_embed    import embed_watermark, save_embed_data
from src.watermark_extract  import extract_watermark
from src.watermark_baseline import embed_baseline, extract_baseline
from utils.attacks  import run_all_attacks, ATTACK_SUITE
from utils.metrics  import evaluate_all, psnr, ssim

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

PATCH_SIZE     = 32
N_KEYPOINTS    = 20
WM_BITS        = 16
RANDOM_SEED    = 42
RNG            = np.random.default_rng(seed=RANDOM_SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def run_one_image(
    img_name: str,
    img: np.ndarray,
    attack_names: list[str],
    attacks: list,
    run_baseline: bool,
    n_keypoints: int,
    wm_bits: int,
) -> dict:
    """
    Run the full pipeline (SIFT+IPVO + optionally Baseline) for one image.

    Returns:
        dict with keys 'sift', 'baseline', 'psnr', 'ssim', 'n_kps', 'elapsed_ms'
    """
    t0 = time.time()
    watermark = RNG.integers(0, 2, wm_bits, dtype=np.uint8)

    # ── SIFT+IPVO ────────────────────────────────────────────────────────────
    sift_rows = []
    try:
        img_wm, embed_data = embed_watermark(img, watermark, n_keypoints)
        psnr_val = psnr(img, img_wm)
        ssim_val = ssim(img, img_wm)
    except Exception as e:
        return {
            'sift': [], 'baseline': [],
            'psnr': np.nan, 'ssim': np.nan,
            'n_kps': 0, 'elapsed_ms': (time.time() - t0) * 1000,
            'error': str(e),
        }

    for (atk_name, img_attacked) in zip(attack_names, attacks):
        wm_ext, surviving_kps, lost_idx, n_located = extract_watermark(
            img_attacked, embed_data)
        m = evaluate_all(img, img_attacked, watermark, wm_ext)
        m['image']       = img_name
        m['attack']      = atk_name
        m['survived']    = len(surviving_kps)
        m['total_kps']   = len(embed_data['keypoints'])
        m['located']     = n_located
        sift_rows.append(m)

    # ── Baseline ─────────────────────────────────────────────────────────────
    bl_rows = []
    if run_baseline:
        try:
            img_bl, embed_bl = embed_baseline(img, watermark, n_keypoints)
        except Exception as e:
            return {
                'sift': sift_rows, 'baseline': [],
                'psnr': psnr_val, 'ssim': ssim_val,
                'n_kps': len(embed_data['keypoints']),
                'elapsed_ms': (time.time() - t0) * 1000,
                'error': str(e),
            }

        for (atk_name, img_attacked) in zip(attack_names, attacks):
            wm_ext_bl = extract_baseline(img_attacked, embed_bl)
            m = evaluate_all(img, img_attacked, watermark, wm_ext_bl)
            m['image']     = img_name
            m['attack']    = atk_name
            bl_rows.append(m)

    return {
        'sift':      sift_rows,
        'baseline':  bl_rows,
        'psnr':      psnr_val,
        'ssim':      ssim_val,
        'n_kps':     len(embed_data['keypoints']),
        'elapsed_ms': (time.time() - t0) * 1000,
    }


def aggregate_stats(df_sift: pd.DataFrame, df_bl: Optional[pd.DataFrame]
                    ) -> dict:
    """
    Compute aggregate statistics (mean, std, min, max) for each attack,
    across all images.
    """
    stats = {}

    for name, df in [('SIFT+IPVO', df_sift), ('Baseline', df_bl)]:
        if df is None or df.empty:
            continue
        by_atk = {}
        for atk in df['attack'].unique():
            sub = df[df['attack'] == atk]
            by_atk[atk] = {
                f'{met}_mean':  round(sub[met].mean(),  4)
                if not sub[met].isna().all() else np.nan
                for met in ('PSNR', 'SSIM', 'BER', 'NC')
            }
            by_atk[atk].update({
                f'{met}_std':   round(sub[met].std(),   4)
                if not sub[met].isna().all() else np.nan
                for met in ('PSNR', 'SSIM', 'BER', 'NC')
            })
        stats[name] = by_atk

    return stats


def plot_summary_bar(df_sift: pd.DataFrame, df_bl: Optional[pd.DataFrame],
                      save_path: Path) -> None:
    """Grouped bar chart: mean BER per attack, SIFT vs Baseline."""
    import matplotlib.pyplot as plt

    attacks = sorted(df_sift['attack'].unique())
    sift_ber   = [df_sift[df_sift['attack']==a]['BER'].mean() for a in attacks]
    bl_ber     = ([df_bl[df_bl['attack']==a]['BER'].mean() for a in attacks]
                  if df_bl is not None else [0] * len(attacks))

    x = np.arange(len(attacks))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(attacks) * 1.2), 5))
    b1 = ax.bar(x - w/2, sift_ber, w, label='SIFT+IPVO', color='#2e7d32', alpha=0.85)
    b2 = ax.bar(x + w/2, bl_ber,   w, label='Baseline',  color='#ef6c00', alpha=0.85)

    ax.set_ylabel('BER (avg across images)')
    ax.set_title('BER Comparison: SIFT+IPVO vs Baseline IPVO')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_', ' ') for a in attacks], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, min(1.0, max(max(sift_ber), max(bl_ber)) * 1.2 + 0.05))
    ax.grid(axis='y', alpha=0.3)

    for bar in b1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in b2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_heatmap(df: pd.DataFrame, metric: str, save_path: Path,
                 cmap: str = 'RdYlGn_r', vmin: float = 0, vmax: float = 1,
                 title: str = '') -> None:
    """Heatmap: images × attacks, cell = mean metric value."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if df.empty:
        return

    pivot = df.pivot_table(index='image', columns='attack', values=metric,
                            aggfunc='mean')
    # sort attacks in fixed order
    atk_order = [a for a, _ in ATTACK_SUITE if a in pivot.columns]
    pivot = pivot.reindex(columns=atk_order)

    # shorten image names for display
    pivot.index = [n.replace('.tiff', '').replace('.512', '') for n in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(atk_order) * 1.1),
                                    max(6, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(atk_order)))
    ax.set_xticklabels([a.replace('_', '\n') for a in atk_order],
                       rotation=0, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(title or f'{metric} per Image × Attack', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(atk_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if (val > vmax * 0.6) else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_psnr_ssim_bars(df_sift: pd.DataFrame, df_bl: Optional[pd.DataFrame],
                         save_path: Path) -> None:
    """Two-grouped bars: PSNR and SSIM per attack (avg across images)."""
    import matplotlib.pyplot as plt

    attacks = sorted(df_sift['attack'].unique())
    psnr_sift = [df_sift[df_sift['attack']==a]['PSNR'].mean() for a in attacks]
    ssim_sift = [df_sift[df_sift['attack']==a]['SSIM'].mean() for a in attacks]

    psnr_bl = ([df_bl[df_bl['attack']==a]['PSNR'].mean() for a in attacks]
               if df_bl is not None else [np.nan] * len(attacks))
    ssim_bl = ([df_bl[df_bl['attack']==a]['SSIM'].mean() for a in attacks]
                if df_bl is not None else [np.nan] * len(attacks))

    x = np.arange(len(attacks))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(attacks) * 1.4), 4))

    # PSNR
    axes[0].bar(x - w, psnr_sift, w, label='SIFT+IPVO', color='#1565c0', alpha=0.85)
    axes[0].bar(x,     psnr_bl,   w, label='Baseline',  color='#f57c00', alpha=0.85)
    axes[0].set_title('PSNR (dB) — image quality')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=7)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    # SSIM
    axes[1].bar(x - w, ssim_sift, w, label='SIFT+IPVO', color='#1565c0', alpha=0.85)
    axes[1].bar(x,     ssim_bl,   w, label='Baseline',  color='#f57c00', alpha=0.85)
    axes[1].set_title('SSIM — image quality')
    axes[1].set_ylabel('SSIM')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=7)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle('Image Quality Metrics per Attack (avg across dataset)', fontsize=11)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full-dataset watermarking evaluation")
    parser.add_argument("--data_dir",  default="data/gray",
                        help="Directory containing grayscale test images")
    parser.add_argument("--subset",    type=int, default=0,
                        help="Run on first N images only (0 = all)")
    parser.add_argument("--attacks",   nargs='+',
                        help="Run only these attacks (default: all)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline IPVO (faster)")
    parser.add_argument("--output_dir", default="output/evaluation",
                        help="Output directory for results")
    parser.add_argument("--n_keypoints", type=int, default=N_KEYPOINTS)
    parser.add_argument("--wm_bits",     type=int, default=WM_BITS)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Select attacks ────────────────────────────────────────────────────────
    if args.attacks:
        selected = [(name, fn) for (name, fn) in ATTACK_SUITE if name in args.attacks]
        if not selected:
            print("No matching attacks found.")
            return
        attack_names = [n for n, _ in selected]
        attacks_fn  = [fn for _, fn in selected]
    else:
        attack_names = [n for n, _ in ATTACK_SUITE]
        attacks_fn  = [fn for _, fn in ATTACK_SUITE]

    # ── Find images ───────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    img_paths = sorted(data_dir.glob("*.tiff"))
    if args.subset > 0:
        img_paths = img_paths[:args.subset]

    print(f"Images found: {len(img_paths)}")
    print(f"Attacks: {attack_names}")
    print(f"Baseline: {'ON' if not args.no_baseline else 'OFF'}")
    print(f"Output: {out}\n")

    # ── Run pipeline per image ────────────────────────────────────────────────
    all_sift_rows   = []
    all_bl_rows     = []
    failed_images   = []

    for idx, img_path in enumerate(img_paths):
        img_name = img_path.name
        img = load_image(img_path)
        if img is None:
            print(f"  [{idx+1}/{len(img_paths)}] SKIP {img_name} — cannot load")
            failed_images.append((img_name, 'load_failed'))
            continue

        # Pre-compute attacked images (same for SIFT and Baseline)
        attacked_imgs = [fn(img.copy()) for fn in attacks_fn]

        result = run_one_image(
            img_name, img,
            attack_names, attacked_imgs,
            run_baseline=not args.no_baseline,
            n_keypoints=args.n_keypoints,
            wm_bits=args.wm_bits,
        )

        if 'error' in result:
            print(f"  [{idx+1}/{len(img_paths)}] FAIL {img_name}: {result['error']}")
            failed_images.append((img_name, result.get('error', 'unknown')))
        else:
            all_sift_rows.extend(result['sift'])
            all_bl_rows.extend(result['baseline'])
            print(
                f"  [{idx+1}/{len(img_paths)}] {img_name}: "
                f"PSNR={result['psnr']:.2f}  SSIM={result['ssim']:.4f}  "
                f"kps={result['n_kps']}  {result['elapsed_ms']:.0f}ms"
            )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df_sift = pd.DataFrame(all_sift_rows) if all_sift_rows else pd.DataFrame()
    df_bl   = pd.DataFrame(all_bl_rows)   if all_bl_rows   else pd.DataFrame()

    if not df_sift.empty:
        df_sift.to_csv(out / "results_sift.csv", index=False)
        print(f"\nSaved: {out / 'results_sift.csv'}  ({len(df_sift)} rows)")

    if not df_bl.empty:
        df_bl.to_csv(out / "results_baseline.csv", index=False)
        print(f"Saved: {out / 'results_baseline.csv'}  ({len(df_bl)} rows)")

    # ── Aggregate stats ───────────────────────────────────────────────────────
    if not df_sift.empty:
        stats = aggregate_stats(df_sift, df_bl if not args.no_baseline else None)
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        print(f"Saved: {out / 'stats.json'}")

        # ── Plots ──────────────────────────────────────────────────────────────
        print("Generating plots...")

        plot_summary_bar(
            df_sift,
            df_bl if not args.no_baseline else None,
            out / "summary_ber_bar.png",
        )
        print(f"  saved: {out / 'summary_ber_bar.png'}")

        plot_psnr_ssim_bars(
            df_sift,
            df_bl if not args.no_baseline else None,
            out / "summary_quality_bar.png",
        )
        print(f"  saved: {out / 'summary_quality_bar.png'}")

        # BER heatmap: SIFT
        plot_heatmap(
            df_sift, 'BER', out / "summary_heatmap_ber.png",
            cmap='RdYlGn_r', vmin=0, vmax=1,
            title='BER — SIFT+IPVO (per image × attack)',
        )
        print(f"  saved: {out / 'summary_heatmap_ber.png'}")

        # BER heatmap: Baseline
        if not args.no_baseline and not df_bl.empty:
            plot_heatmap(
                df_bl, 'BER', out / "summary_heatmap_ber_baseline.png",
                cmap='RdYlGn_r', vmin=0, vmax=1,
                title='BER — Baseline IPVO (per image × attack)',
            )
            print(f"  saved: {out / 'summary_heatmap_ber_baseline.png'}")

        # NC heatmap
        plot_heatmap(
            df_sift, 'NC', out / "summary_heatmap_nc.png",
            cmap='RdYlGn', vmin=0, vmax=1,
            title='NC — SIFT+IPVO (per image × attack)',
        )
        print(f"  saved: {out / 'summary_heatmap_nc.png'}")

    # ── Failed images ──────────────────────────────────────────────────────────
    if failed_images:
        (out / "failed_images.txt").write_text(
            '\n'.join(f'{name}: {err}' for name, err in failed_images)
        )
        print(f"\nFailed images ({len(failed_images)}): {out / 'failed_images.txt'}")

    print(f"\nDone! All results in: {out.resolve()}")


if __name__ == "__main__":
    main()
