"""
main.py — CLI runner for SIFT + IPVO watermarking demo.

Usage:
    python main.py
    python main.py --image data/gray/7.1.01.tiff --n_keypoints 15

All outputs are saved to the output/ folder.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

from src.watermark_embed    import embed_watermark, save_embed_data
from src.watermark_extract  import extract_watermark
from src.watermark_baseline import embed_baseline, extract_baseline, grid_patch_keypoints, PATCH_SIZE
from utils.attacks  import run_all_attacks, ATTACK_SUITE
from utils.metrics  import evaluate_all, psnr, ssim
from utils.visualize import (
    draw_keypoints,
    draw_embedded_patches,
    draw_extracted_patches,
    save_attacked_image,
    draw_metrics_table,
    draw_comparison_table,
    export_watermark_debug_artifacts,
)


def run(image_path: str, n_keypoints: int = 20, wm_bits: int = 64,
        output_dir: str = "output", dump_sift_steps: bool = False,
        dump_watermark_debug: bool = False):

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # ── 1. Load image ─────────────────────────────────────────────────────
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    print(f"[1] Image loaded: {image_path}  size={img.shape}")

    # ── 2. Generate watermark ──────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    watermark = rng.integers(0, 2, wm_bits, dtype=np.uint8)
    print(f"[2] Watermark ({wm_bits} bits): {watermark}")

    if dump_sift_steps:
        from src.sift_utils import export_sift_debug_artifacts
        dbg = export_sift_debug_artifacts(img, str(out), n_keypoints)
        print(
            f"[2.5] SIFT debug exported -> {dbg['output_dir']} "
            f"(selected={dbg['num_selected']}, raw_oriented={dbg['num_raw_oriented']})"
        )

    # ── 3. Detect keypoints + visualize ───────────────────────────────────
    from src.sift_utils import detect_keypoints
    kps = detect_keypoints(img, n_keypoints)
    draw_keypoints(img, kps, save_path=str(out / "1_keypoints.png"))
    print(f"[3] Detected {len(kps)} keypoints -> output/1_keypoints.png")

    # ── 4. Embed ───────────────────────────────────────────────────────────
    img_wm, embed_data = embed_watermark(img, watermark, n_keypoints)
    save_embed_data(embed_data, str(out / "embed_data.pkl"))

    draw_embedded_patches(
        img_wm,
        embed_data['keypoints'],
        save_clean  = str(out / "2_watermarked.png"),
        save_marked = str(out / "3_embed_marked.png"),
    )

    p = psnr(img, img_wm)
    s = ssim(img, img_wm)
    print(f"[4] Watermark embedded -> PSNR={p:.2f} dB  SSIM={s:.4f}")
    print(f"    output/2_watermarked.png  output/3_embed_marked.png")

    if dump_watermark_debug:
        export_watermark_debug_artifacts(
            img, img_wm, embed_data, str(out), n_samples=6)
        print(f"    watermark debug -> output/watermark_debug/")

    # ── 5. Attack + Extract ────────────────────────────────────────────────
    results = []
    attacks = run_all_attacks(img_wm)

    for attack_name, img_attacked in attacks:
        save_attacked_image(img_attacked, attack_name, output_dir)

        wm_ext, surviving_kps, lost_idx, n_located = extract_watermark(img_attacked, embed_data)

        draw_extracted_patches(
            img_attacked,
            embed_data['keypoints'],
            surviving_kps,
            lost_idx,
            n_located   = n_located,
            save_path   = str(out / f"5_extract_{attack_name}.png"),
        )

        metrics = evaluate_all(img, img_attacked, watermark, wm_ext)
        metrics['attack']   = attack_name
        metrics['survived'] = f"{len(surviving_kps)}/{len(embed_data['keypoints'])}"
        results.append(metrics)

        print(f"  [{attack_name:20s}]  BER={metrics['BER']:.4f}  "
              f"NC={metrics['NC']:.4f}  survived={metrics['survived']}")

    # ── 6. Metrics table (SIFT+IPVO) ──────────────────────────────────────
    draw_metrics_table(results, save_path=str(out / "6_metrics_table.png"))
    print(f"\n[6] Metrics table -> output/6_metrics_table.png")

    # ── 7. Baseline: IPVO only (no SIFT) ──────────────────────────────────
    print("\n--- Baseline: IPVO only (no SIFT) ---")
    img_bl, embed_bl = embed_baseline(img, watermark, n_keypoints)
    bl_kps = grid_patch_keypoints(embed_bl['positions'])
    draw_embedded_patches(
        img_bl, bl_kps,
        save_clean             = str(out / "7_baseline_watermarked.png"),
        save_marked            = str(out / "7_baseline_embed_marked.png"),
        show_keypoint_circles  = False,
    )
    p_bl = psnr(img, img_bl)
    s_bl = ssim(img, img_bl)
    print(f"    PSNR={p_bl:.2f} dB  SSIM={s_bl:.4f}")
    print(f"    output/7_baseline_watermarked.png  output/7_baseline_embed_marked.png")

    baseline_results = []
    for attack_name, img_attacked in run_all_attacks(img_bl):
        wm_ext_bl = extract_baseline(img_attacked, embed_bl)

        # Baseline visualization: green = patch content mostly intact (low MSE),
        # red = patch content heavily changed by attack (high MSE) even though
        # coordinates are in-bounds.  This reveals geometric distortion clearly.
        h_att, w_att = img_attacked.shape
        surviving_bl = []   # green: low pixel change
        lost_bl      = []   # red:   high pixel change (content corrupted)
        _MSE_THRESH = 30.0  # mean squared pixel error threshold
        for i, (r, c) in enumerate(embed_bl['positions']):
            r_c = min(r, h_att - PATCH_SIZE)
            c_c = min(c, w_att - PATCH_SIZE)
            if r_c < 0 or c_c < 0:
                lost_bl.append(i)
                continue
            patch_orig = img_bl[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
            patch_att  = img_attacked[r_c:r_c + PATCH_SIZE, c_c:c_c + PATCH_SIZE]
            if patch_att.shape != (PATCH_SIZE, PATCH_SIZE):
                lost_bl.append(i)
                continue
            mse = float(np.mean((patch_orig.astype(np.float32) -
                                 patch_att.astype(np.float32)) ** 2))
            kp = cv2.KeyPoint(x=float(c_c + PATCH_SIZE / 2),
                              y=float(r_c + PATCH_SIZE / 2),
                              size=float(PATCH_SIZE))
            if mse <= _MSE_THRESH:
                surviving_bl.append(kp)
            else:
                lost_bl.append(i)

        draw_extracted_patches(
            img_attacked,
            bl_kps,
            surviving_bl,
            lost_bl,
            n_located             = len(surviving_bl) + len(lost_bl),
            show_keypoint_circles = False,
            save_path             = str(out / f"7_baseline_extract_{attack_name}.png"),
        )

        metrics_bl = evaluate_all(img, img_attacked, watermark, wm_ext_bl)
        metrics_bl['attack'] = attack_name
        baseline_results.append(metrics_bl)
        print(f"  [{attack_name:20s}]  BER={metrics_bl['BER']:.4f}  NC={metrics_bl['NC']:.4f}")

    # ── 8. Comparison table ────────────────────────────────────────────────
    draw_comparison_table(
        results, baseline_results,
        save_path=str(out / "8_comparison_table.png"),
    )
    print(f"\n[8] Comparison table -> output/8_comparison_table.png")
    print("\nDone! All outputs saved to output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIFT + IPVO Watermarking Demo")
    parser.add_argument("--image",       default="data/gray/boat.512.tiff",
                        help="Path to grayscale test image")
    parser.add_argument("--n_keypoints", type=int, default=20,
                        help="Number of SIFT keypoints to use")
    parser.add_argument("--wm_bits",     type=int, default=16,
                        help="Watermark length in bits")
    parser.add_argument("--output_dir",  default="output",
                        help="Output directory")
    parser.add_argument("--dump_sift_steps", action="store_true",
                        help="Export intermediate SIFT step images + README.md")
    parser.add_argument("--dump_watermark_debug", action="store_true",
                        help="Export watermark embedding debug images (diff, bit heatmap, loc map)")
    args = parser.parse_args()

    run(args.image, args.n_keypoints, args.wm_bits, args.output_dir,
        args.dump_sift_steps, args.dump_watermark_debug)
