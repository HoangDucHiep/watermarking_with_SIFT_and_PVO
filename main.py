"""
main.py — Demo SIFT + Spread Spectrum watermarking.

So sánh hai phương pháp:
  • SIFT + SS  : nhúng vào canonical patch (chuẩn hoá hình học bằng SIFT)
  • Baseline SS: nhúng vào patch cố định trên lưới (không SIFT)

Usage:
    python main.py
    python main.py --image data/gray/7.1.01.tiff --n_keypoints 15
    python main.py --no-baseline    # chỉ chạy SIFT+SS
    python main.py --dump_sift_steps

Outputs (saved to output/):
    1_keypoints.png              – SIFT keypoints được chọn
    2_watermarked.png            – ảnh SIFT+SS watermarked
    3_embed_marked.png           – ảnh với patch boundaries
    4_attacked_<atk>.png         – các ảnh bị tấn công
    5_extract_<atk>.png          – kết quả extract sau tấn công
    6_metrics_table.png          – bảng BER/NC của SIFT+SS
    7_baseline_*.png             – ảnh baseline
    8_comparison_table.png       – bảng so sánh SIFT+SS vs Baseline
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

from src.watermark_embed    import embed_watermark, save_embed_data
from src.watermark_extract  import extract_watermark
from src.watermark_baseline import embed_baseline, extract_baseline, grid_patch_keypoints, PATCH_SIZE
from utils.attacks   import run_all_attacks, ATTACK_SUITE
from utils.metrics   import evaluate_all, psnr, ssim
from utils.visualize import (
    draw_keypoints,
    draw_embedded_patches,
    draw_extracted_patches,
    save_attacked_image,
    draw_metrics_table,
    draw_comparison_table,
)


def run(image_path: str, n_keypoints: int = 20, wm_bits: int = 16,
        output_dir: str = "output", run_baseline: bool = True,
        dump_sift_steps: bool = False):

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # ── 1. Load ảnh ───────────────────────────────────────────────────────────
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    print(f"[1] Ảnh: {image_path}  size={img.shape}")

    # ── 2. Sinh watermark ngẫu nhiên ──────────────────────────────────────────
    rng       = np.random.default_rng(seed=42)
    watermark = rng.integers(0, 2, wm_bits, dtype=np.uint8)
    print(f"[2] Watermark ({wm_bits} bits): {watermark}")

    # ── 2.5 Export SIFT debug (tuỳ chọn) ─────────────────────────────────────
    if dump_sift_steps:
        from src.sift_utils import export_sift_debug_artifacts
        dbg = export_sift_debug_artifacts(img, str(out), n_keypoints)
        print(f"[2.5] SIFT debug -> {dbg['output_dir']}  "
              f"(selected={dbg['num_selected']})")

    # ── 3. Phát hiện keypoints & visualise ────────────────────────────────────
    from src.sift_utils import detect_keypoints
    kps = detect_keypoints(img, n_keypoints)
    draw_keypoints(img, kps, save_path=str(out / "1_keypoints.png"))
    print(f"[3] {len(kps)} keypoints -> output/1_keypoints.png")

    # ── 4. Nhúng watermark (SIFT + SS) ────────────────────────────────────────
    img_wm, embed_data = embed_watermark(img, watermark, n_keypoints)
    save_embed_data(embed_data, str(out / "embed_data.pkl"))

    draw_embedded_patches(
        img_wm, embed_data['keypoints'],
        save_clean  = str(out / "2_watermarked.png"),
        save_marked = str(out / "3_embed_marked.png"),
    )

    p = psnr(img, img_wm)
    s = ssim(img, img_wm)
    print(f"[4] SIFT+SS embedded -> PSNR={p:.2f} dB  SSIM={s:.4f}")
    print(f"    output/2_watermarked.png  output/3_embed_marked.png")

    # ── 5. Tấn công + Extract (SIFT + SS) ─────────────────────────────────────
    print("\n--- SIFT + SS ---")
    sift_results = []

    for attack_name, img_attacked in run_all_attacks(img_wm):
        save_attacked_image(img_attacked, attack_name, output_dir)

        wm_ext, surviving_kps, lost_idx, n_located = extract_watermark(
            img_attacked, embed_data)

        draw_extracted_patches(
            img_attacked, embed_data['keypoints'], surviving_kps, lost_idx,
            n_located=n_located,
            save_path=str(out / f"5_extract_{attack_name}.png"),
        )

        m = evaluate_all(img, img_attacked, watermark, wm_ext)
        m['attack']   = attack_name
        m['survived'] = f"{len(surviving_kps)}/{len(embed_data['keypoints'])}"
        sift_results.append(m)
        print(f"  [{attack_name:20s}]  BER={m['BER']:.4f}  "
              f"NC={m['NC']:.4f}  survived={m['survived']}")

    draw_metrics_table(sift_results, save_path=str(out / "6_metrics_table.png"))
    print(f"\n[6] Metrics table -> output/6_metrics_table.png")

    # ── 7. Baseline (SS không có SIFT) ────────────────────────────────────────
    baseline_results = []
    if run_baseline:
        print("\n--- Baseline SS (không có SIFT) ---")
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

        for attack_name, img_attacked_bl in run_all_attacks(img_bl):
            wm_ext_bl = extract_baseline(img_attacked_bl, embed_bl)
            m_bl      = evaluate_all(img, img_attacked_bl, watermark, wm_ext_bl)
            m_bl['attack'] = attack_name
            baseline_results.append(m_bl)
            print(f"  [{attack_name:20s}]  BER={m_bl['BER']:.4f}  NC={m_bl['NC']:.4f}")

    # ── 8. Bảng so sánh ───────────────────────────────────────────────────────
    draw_comparison_table(
        sift_results,
        baseline_results if run_baseline else [],
        save_path=str(out / "8_comparison_table.png"),
    )
    print(f"\n[8] Comparison table -> output/8_comparison_table.png")
    print("\nDone! Tất cả output trong output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIFT + Spread Spectrum Watermarking")
    parser.add_argument("--image",        default="data/gray/boat.512.tiff")
    parser.add_argument("--n_keypoints",  type=int, default=20)
    parser.add_argument("--wm_bits",      type=int, default=16)
    parser.add_argument("--output_dir",   default="output")
    parser.add_argument("--no-baseline",  action="store_true",
                        help="Bỏ qua Baseline SS")
    parser.add_argument("--dump_sift_steps", action="store_true",
                        help="Export ảnh debug các bước SIFT")
    args = parser.parse_args()

    run(args.image, args.n_keypoints, args.wm_bits, args.output_dir,
        run_baseline=not args.no_baseline,
        dump_sift_steps=args.dump_sift_steps)
