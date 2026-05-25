"""
evaluate.py — Đánh giá toàn bộ dataset cho SIFT+SS watermarking.

Usage:
    python evaluate.py                          # tất cả ảnh, tất cả tấn công
    python evaluate.py --subset 5              # thử nhanh 5 ảnh đầu
    python evaluate.py --attacks rotate_15 scale_0.75
    python evaluate.py --no-baseline           # bỏ qua Baseline SS (nhanh hơn)

Outputs (saved to output/evaluation/):
    results_sift.csv        – metrics theo từng ảnh, từng tấn công (SIFT+SS)
    results_baseline.csv    – metrics theo từng ảnh, từng tấn công (Baseline SS)
    stats.json              – thống kê tổng hợp (mean, std per attack)
    summary_ber_bar.png     – biểu đồ cột BER: SIFT+SS vs Baseline SS
    summary_heatmap_ber.png – heatmap BER: image × attack (SIFT+SS)
    summary_quality_bar.png – biểu đồ PSNR/SSIM
    failed_images.txt       – ảnh lỗi (nếu có)
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

from src.watermark_embed    import embed_watermark
from src.watermark_extract  import extract_watermark
from src.watermark_baseline import embed_baseline, extract_baseline
from utils.attacks  import run_all_attacks, ATTACK_SUITE
from utils.metrics  import evaluate_all, psnr, ssim

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

N_KEYPOINTS = 20
WM_BITS     = 16
RANDOM_SEED = 42
RNG         = np.random.default_rng(seed=RANDOM_SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def run_one_image(
    img_name: str,
    img: np.ndarray,
    attack_names: list[str],
    attacks_fn: list,
    run_baseline: bool,
    n_keypoints: int,
    wm_bits: int,
) -> dict:
    """
    Chạy pipeline SIFT+SS (+ tuỳ chọn Baseline SS) cho một ảnh.

    NOTE: attacks_fn là list callables — attack được áp dụng SAU khi embed,
    tức là trên ảnh đã nhúng watermark (img_wm / img_bl), không phải img gốc.

    Returns:
        dict với keys 'sift', 'baseline', 'psnr', 'ssim', 'n_kps', 'elapsed_ms'
    """
    t0        = time.time()
    watermark = RNG.integers(0, 2, wm_bits, dtype=np.uint8)

    # ── SIFT + SS ─────────────────────────────────────────────────────────────
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

    # Attack trên ảnh ĐÃ watermark (img_wm)
    for atk_name, atk_fn in zip(attack_names, attacks_fn):
        img_attacked = atk_fn(img_wm.copy())
        wm_ext, surviving_kps, _, n_located = extract_watermark(
            img_attacked, embed_data)
        m = evaluate_all(img, img_attacked, watermark, wm_ext)
        m['image']     = img_name
        m['attack']    = atk_name
        m['survived']  = len(surviving_kps)
        m['total_kps'] = len(embed_data['keypoints'])
        m['located']   = n_located
        sift_rows.append(m)

    # ── Baseline SS (không có SIFT) ───────────────────────────────────────────
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

        # Attack trên ảnh baseline ĐÃ watermark (img_bl)
        for atk_name, atk_fn in zip(attack_names, attacks_fn):
            img_attacked_bl = atk_fn(img_bl.copy())
            wm_ext_bl = extract_baseline(img_attacked_bl, embed_bl)
            m = evaluate_all(img, img_attacked_bl, watermark, wm_ext_bl)
            m['image']  = img_name
            m['attack'] = atk_name
            bl_rows.append(m)

    return {
        'sift':       sift_rows,
        'baseline':   bl_rows,
        'psnr':       psnr_val,
        'ssim':       ssim_val,
        'n_kps':      len(embed_data['keypoints']),
        'elapsed_ms': (time.time() - t0) * 1000,
    }


def aggregate_stats(df_sift: pd.DataFrame,
                    df_bl: Optional[pd.DataFrame]) -> dict:
    """Tính mean/std cho từng metric × từng attack."""
    stats = {}
    for name, df in [('SIFT+SS', df_sift), ('Baseline SS', df_bl)]:
        if df is None or df.empty:
            continue
        by_atk = {}
        for atk in df['attack'].unique():
            sub = df[df['attack'] == atk]
            by_atk[atk] = {
                f'{met}_mean': round(sub[met].mean(), 4) if not sub[met].isna().all() else np.nan
                for met in ('PSNR', 'SSIM', 'BER', 'NC')
            }
            by_atk[atk].update({
                f'{met}_std': round(sub[met].std(), 4) if not sub[met].isna().all() else np.nan
                for met in ('PSNR', 'SSIM', 'BER', 'NC')
            })
        stats[name] = by_atk
    return stats


def plot_summary_bar(df_sift: pd.DataFrame, df_bl: Optional[pd.DataFrame],
                     save_path: Path) -> None:
    """Biểu đồ cột: BER trung bình mỗi attack, SIFT+SS vs Baseline SS."""
    import matplotlib.pyplot as plt

    attacks  = sorted(df_sift['attack'].unique())
    sift_ber = [df_sift[df_sift['attack'] == a]['BER'].mean() for a in attacks]
    bl_ber   = ([df_bl[df_bl['attack'] == a]['BER'].mean() for a in attacks]
                if df_bl is not None else [0] * len(attacks))

    x, w = np.arange(len(attacks)), 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(attacks) * 1.2), 5))
    b1 = ax.bar(x - w/2, sift_ber, w, label='SIFT+SS',     color='#1976d2', alpha=0.85)
    b2 = ax.bar(x + w/2, bl_ber,   w, label='Baseline SS', color='#e64a19', alpha=0.85)

    ax.set_ylabel('BER (trung bình qua tất cả ảnh)')
    ax.set_title('So sánh BER: SIFT+SS vs Baseline SS')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_', ' ') for a in attacks], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.4, label='BER=0.5 (random)')
    ax.grid(axis='y', alpha=0.3)

    for bars in (b1, b2):
        for bar in bars:
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
    """Heatmap: ảnh × tấn công, cell = giá trị metric."""
    import matplotlib.pyplot as plt

    if df.empty:
        return
    pivot     = df.pivot_table(index='image', columns='attack',
                               values=metric, aggfunc='mean')
    atk_order = [a for a, _ in ATTACK_SUITE if a in pivot.columns]
    pivot     = pivot.reindex(columns=atk_order)
    pivot.index = [n.replace('.tiff', '').replace('.512', '') for n in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(atk_order) * 1.1),
                                    max(6, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(atk_order)))
    ax.set_xticklabels([a.replace('_', '\n') for a in atk_order], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(title or f'{metric} per Image × Attack', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02).ax.tick_params(labelsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(atk_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_quality_bars(df_sift: pd.DataFrame, df_bl: Optional[pd.DataFrame],
                      save_path: Path) -> None:
    """Biểu đồ PSNR và SSIM trung bình mỗi attack."""
    import matplotlib.pyplot as plt

    attacks   = sorted(df_sift['attack'].unique())
    psnr_sift = [df_sift[df_sift['attack'] == a]['PSNR'].mean() for a in attacks]
    ssim_sift = [df_sift[df_sift['attack'] == a]['SSIM'].mean() for a in attacks]
    psnr_bl   = ([df_bl[df_bl['attack'] == a]['PSNR'].mean() for a in attacks]
                 if df_bl is not None else [np.nan] * len(attacks))
    ssim_bl   = ([df_bl[df_bl['attack'] == a]['SSIM'].mean() for a in attacks]
                 if df_bl is not None else [np.nan] * len(attacks))

    x, w  = np.arange(len(attacks)), 0.2
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(attacks) * 1.4), 4))

    axes[0].bar(x - w, psnr_sift, w, label='SIFT+SS',     color='#1565c0', alpha=0.85)
    axes[0].bar(x,     psnr_bl,   w, label='Baseline SS', color='#f57c00', alpha=0.85)
    axes[0].set_title('PSNR (dB)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=7)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x - w, ssim_sift, w, label='SIFT+SS',     color='#1565c0', alpha=0.85)
    axes[1].bar(x,     ssim_bl,   w, label='Baseline SS', color='#f57c00', alpha=0.85)
    axes[1].set_title('SSIM')
    axes[1].set_ylabel('SSIM')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=7)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle('Chất lượng ảnh watermarked (trung bình qua dataset)', fontsize=11)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Đánh giá SIFT+SS trên toàn dataset")
    parser.add_argument("--data_dir",    default="data/gray")
    parser.add_argument("--subset",      type=int, default=0,
                        help="Chỉ dùng N ảnh đầu (0 = tất cả)")
    parser.add_argument("--attacks",     nargs='+',
                        help="Chọn các tấn công cụ thể (mặc định: tất cả)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Bỏ qua Baseline SS")
    parser.add_argument("--output_dir",  default="output/evaluation")
    parser.add_argument("--n_keypoints", type=int, default=N_KEYPOINTS)
    parser.add_argument("--wm_bits",     type=int, default=WM_BITS)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Chọn attacks
    if args.attacks:
        selected = [(n, fn) for n, fn in ATTACK_SUITE if n in args.attacks]
        if not selected:
            print("Không tìm thấy attack nào khớp.")
            return
    else:
        selected = list(ATTACK_SUITE)
    attack_names = [n for n, _ in selected]
    attacks_fn   = [fn for _, fn in selected]

    # Tìm ảnh
    data_dir  = Path(args.data_dir)
    img_paths = sorted(data_dir.glob("*.tiff"))
    if args.subset > 0:
        img_paths = img_paths[:args.subset]

    print(f"Ảnh: {len(img_paths)}  |  Attacks: {attack_names}")
    print(f"Baseline: {'TẮT' if args.no_baseline else 'BẬT'}  |  Output: {out}\n")

    all_sift_rows, all_bl_rows, failed = [], [], []

    for idx, img_path in enumerate(img_paths):
        img_name = img_path.name
        img = load_image(img_path)
        if img is None:
            print(f"  [{idx+1}/{len(img_paths)}] SKIP {img_name} — không đọc được")
            failed.append((img_name, 'load_failed'))
            continue

        result = run_one_image(
            img_name, img, attack_names, attacks_fn,
            run_baseline=not args.no_baseline,
            n_keypoints=args.n_keypoints,
            wm_bits=args.wm_bits,
        )

        if 'error' in result:
            print(f"  [{idx+1}/{len(img_paths)}] FAIL {img_name}: {result['error']}")
            failed.append((img_name, result.get('error', 'unknown')))
        else:
            all_sift_rows.extend(result['sift'])
            all_bl_rows.extend(result['baseline'])
            print(f"  [{idx+1}/{len(img_paths)}] {img_name}: "
                  f"PSNR={result['psnr']:.2f}  SSIM={result['ssim']:.4f}  "
                  f"kps={result['n_kps']}  {result['elapsed_ms']:.0f}ms")

    # Lưu CSV
    df_sift = pd.DataFrame(all_sift_rows) if all_sift_rows else pd.DataFrame()
    df_bl   = pd.DataFrame(all_bl_rows)   if all_bl_rows   else pd.DataFrame()

    if not df_sift.empty:
        df_sift.to_csv(out / "results_sift.csv", index=False)
        print(f"\nSaved: {out/'results_sift.csv'}  ({len(df_sift)} rows)")

    if not df_bl.empty:
        df_bl.to_csv(out / "results_baseline.csv", index=False)
        print(f"Saved: {out/'results_baseline.csv'}  ({len(df_bl)} rows)")

    if not df_sift.empty:
        stats = aggregate_stats(df_sift, df_bl if not args.no_baseline else None)
        (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"Saved: {out/'stats.json'}")

        print("Generating plots...")
        plot_summary_bar(df_sift, df_bl if not args.no_baseline else None,
                         out / "summary_ber_bar.png")
        plot_quality_bars(df_sift, df_bl if not args.no_baseline else None,
                          out / "summary_quality_bar.png")
        plot_heatmap(df_sift, 'BER', out / "summary_heatmap_ber.png",
                     cmap='RdYlGn_r', vmin=0, vmax=1,
                     title='BER — SIFT+SS (per image × attack)')
        plot_heatmap(df_sift, 'NC', out / "summary_heatmap_nc.png",
                     cmap='RdYlGn', vmin=0, vmax=1,
                     title='NC — SIFT+SS (per image × attack)')
        print(f"Plots saved to {out}")

    if failed:
        (out / "failed_images.txt").write_text(
            '\n'.join(f'{n}: {e}' for n, e in failed))
        print(f"\nFailed: {len(failed)} ảnh -> {out/'failed_images.txt'}")

    print(f"\nDone! Kết quả trong: {out.resolve()}")


if __name__ == "__main__":
    main()
