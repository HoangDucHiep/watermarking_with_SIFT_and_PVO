"""
Visualization utilities for the SIFT+IPVO watermarking pipeline.

Saved outputs:
  1_keypoints.png      – original image + rich SIFT keypoint circles
  2_watermarked.png    – clean watermarked image (no annotation)
  3_embed_marked.png   – watermarked image + highlighted patch regions
  4_attacked_<name>.png– attacked image
  5_extract_marked.png – attacked image + green/red circles at keypoints
  6_metrics_table.png  – table of PSNR/SSIM/BER/NC for each attack
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.sift_utils import PATCH_SIZE, SCALE_FACTOR


# ── colour palette (BGR for OpenCV, RGB for matplotlib) ──────────────────────
_GREEN  = (0, 200,  50)
_RED    = (0,  50, 220)
_BLUE   = (220, 80,  0)
_YELLOW = (0, 220, 220)
_ALPHA  = 0.35          # overlay transparency for patch highlights


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale to BGR for colored annotations."""
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray.copy()


def _kp_radius(kp: cv2.KeyPoint) -> int:
    """Circle radius proportional to keypoint scale (importance)."""
    return max(int(kp.size / 2), 5)


def _patch_half(kp: cv2.KeyPoint) -> int:
    """Half-side of the square representing the embedded patch region."""
    sigma = kp.size / 2.0
    return max(int(sigma * SCALE_FACTOR / 2), 8)


# Keep old name as alias so existing callers don't break
def _patch_radius(kp: cv2.KeyPoint) -> int:
    return _patch_half(kp)


# ─────────────────────────────────────────────────────────────────────────────

def draw_keypoints(img: np.ndarray,
                   keypoints: list[cv2.KeyPoint],
                   save_path: str | None = None) -> np.ndarray:
    """
    Draw rich SIFT keypoints (circle + orientation line) on the image.

    Args:
        img:       Grayscale uint8 image
        keypoints: List of cv2.KeyPoint
        save_path: If given, save PNG to this path

    Returns:
        Annotated BGR image
    """
    vis = _to_bgr(img)
    vis = cv2.drawKeypoints(
        vis, keypoints, None,
        color=_GREEN,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Number each keypoint
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.putText(vis, str(i + 1), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _YELLOW, 1, cv2.LINE_AA)

    # Legend
    cv2.putText(vis, f"SIFT keypoints: {len(keypoints)}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 1, cv2.LINE_AA)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, vis)

    return vis


def draw_embedded_patches(img_watermarked: np.ndarray,
                           keypoints: list[cv2.KeyPoint],
                           save_clean: str | None = None,
                           save_marked: str | None = None,
                           show_keypoint_circles: bool = True,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Save two versions of the watermarked image:
      - clean  (2_watermarked.png): no annotations
      - marked (3_embed_marked.png): semi-transparent squares showing each
                embedded patch region, with keypoint circle inside, numbered

    Keypoints  → small circles  (orientation-aware, from SIFT)
    Patch area → squares        (half-side = patch radius)
    """
    clean = _to_bgr(img_watermarked)
    marked = clean.copy()

    # Overlay: filled semi-transparent squares for patch regions
    overlay = marked.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        h = _patch_half(kp)
        cv2.rectangle(overlay, (x - h, y - h), (x + h, y + h), _BLUE, -1)

    cv2.addWeighted(overlay, _ALPHA, marked, 1 - _ALPHA, 0, marked)

    # Draw: square = patch region (blue), circle = keypoint importance (yellow, SIFT only)
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        h    = _patch_half(kp)
        cv2.rectangle(marked, (x - h, y - h), (x + h, y + h), _BLUE, 2)
        if show_keypoint_circles:
            kr = _kp_radius(kp)
            cv2.circle(marked, (x, y), kr, _YELLOW, 2)
        cv2.putText(marked, str(i + 1), (x + h + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _YELLOW, 1, cv2.LINE_AA)

    cv2.putText(marked, f"Embedded patches: {len(keypoints)}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BLUE, 1, cv2.LINE_AA)

    if save_clean:
        Path(save_clean).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_clean, clean)
    if save_marked:
        Path(save_marked).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_marked, marked)

    return clean, marked


def draw_extracted_patches(img_attacked: np.ndarray,
                            original_keypoints: list[cv2.KeyPoint],
                            surviving_kps: list[cv2.KeyPoint],
                            lost_orig_indices: list[int],
                            n_located: int | None = None,
                            show_keypoint_circles: bool = True,
                            save_path: str | None = None) -> np.ndarray:
    """
    Visualize which patches survived after a geometric attack.

    Green circles  – patches successfully detected & extracted
    Red   circles  – original patch locations that were lost

    Args:
        img_attacked:       Grayscale attacked image
        original_keypoints: All original keypoints (for showing lost ones)
        surviving_kps:      Re-detected keypoints that matched originals
        lost_orig_indices:  Indices into original_keypoints that were lost
        save_path:          If given, save PNG

    Returns:
        Annotated BGR image
    """
    vis = _to_bgr(img_attacked)

    # Surviving: green square (patch region), + green circle if SIFT
    for kp in surviving_kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        h    = _patch_half(kp)
        cv2.rectangle(vis, (x - h, y - h), (x + h, y + h), _GREEN, 2)
        if show_keypoint_circles:
            kr = _kp_radius(kp)
            cv2.circle(vis, (x, y), kr, _GREEN, 2)

    # Lost: red square (patch region), + red circle + X if SIFT
    for idx in lost_orig_indices:
        kp   = original_keypoints[idx]
        x, y = int(kp.pt[0]), int(kp.pt[1])
        h    = _patch_half(kp)
        cv2.rectangle(vis, (x - h, y - h), (x + h, y + h), _RED, 2)
        if show_keypoint_circles:
            kr = _kp_radius(kp)
            cv2.circle(vis, (x, y), kr, _RED, 2)
            cv2.line(vis, (x - kr, y - kr), (x + kr, y + kr), _RED, 1)
            cv2.line(vis, (x + kr, y - kr), (x - kr, y + kr), _RED, 1)
        else:
            # For baseline: just draw an X inside the square
            cv2.line(vis, (x - h + 4, y - h + 4), (x + h - 4, y + h - 4), _RED, 1)
            cv2.line(vis, (x + h - 4, y - h + 4), (x - h + 4, y + h - 4), _RED, 1)

    n_total = len(original_keypoints)
    n_surv  = len(surviving_kps)
    n_loc   = n_located if n_located is not None else n_surv

    cv2.putText(vis, f"Located:   {n_loc}/{n_total} patches",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1, cv2.LINE_AA)
    cv2.putText(vis, f"Extracted: {n_surv}/{n_total} patches",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 1, cv2.LINE_AA)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, vis)

    return vis


def save_attacked_image(img_attacked: np.ndarray,
                        attack_name: str,
                        output_dir: str = "output") -> None:
    """Save an attacked image with a clear filename."""
    path = Path(output_dir) / f"4_attacked_{attack_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_attacked)


def draw_metrics_table(results: list[dict],
                       save_path: str | None = None) -> None:
    """
    Draw a formatted table of metrics for multiple attacks.

    Args:
        results: List of dicts, each with keys:
                 'attack', 'PSNR', 'SSIM', 'BER', 'NC', 'survived'
        save_path: If given, save PNG
    """
    if not results:
        return

    col_labels = ['Attack', 'PSNR (dB)', 'SSIM', 'BER', 'NC', 'Patches']
    rows = []
    for r in results:
        rows.append([
            r.get('attack', '—'),
            f"{r['PSNR']:.2f}"  if r.get('PSNR') is not None else '—',
            f"{r['SSIM']:.4f}"  if r.get('SSIM') is not None else '—',
            f"{r['BER']:.4f}"   if r.get('BER')  is not None else '—',
            f"{r['NC']:.4f}"    if r.get('NC')    is not None else '—',
            str(r.get('survived', '—')),
        ])

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.45 * (len(rows) + 1)))
    ax.axis('off')

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    # Header styling
    for col in range(len(col_labels)):
        tbl[(0, col)].set_facecolor('#2c5f8a')
        tbl[(0, col)].set_text_props(color='white', fontweight='bold')

    # Alternating row colours
    for row in range(1, len(rows) + 1):
        color = '#f0f4f8' if row % 2 == 0 else 'white'
        for col in range(len(col_labels)):
            tbl[(row, col)].set_facecolor(color)

    plt.title('Watermark Robustness — SIFT + IPVO', fontsize=12, pad=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def draw_comparison_table(
    sift_results: list[dict],
    baseline_results: list[dict],
    save_path: str | None = None,
) -> None:
    """
    Side-by-side BER / NC comparison: SIFT+IPVO vs. Baseline (IPVO only).

    Args:
        sift_results:     List of metric dicts from the SIFT+IPVO pipeline.
        baseline_results: List of metric dicts from the baseline pipeline.
        save_path:        If given, save PNG.
    """
    if not sift_results or not baseline_results:
        return

    # Build lookup by attack name
    bl_map = {r['attack']: r for r in baseline_results}

    col_labels = [
        'Attack',
        'SIFT BER', 'SIFT NC',
        'Baseline BER', 'Baseline NC',
        'BER improve',
    ]
    rows = []
    for r in sift_results:
        name = r['attack']
        bl   = bl_map.get(name, {})
        ber_s  = r.get('BER', float('nan'))
        ber_b  = bl.get('BER', float('nan'))
        nc_s   = r.get('NC',  float('nan'))
        nc_b   = bl.get('NC', float('nan'))
        delta  = ber_b - ber_s          # positive → SIFT is better
        rows.append([
            name,
            f"{ber_s:.4f}", f"{nc_s:.4f}",
            f"{ber_b:.4f}", f"{nc_b:.4f}",
            f"{delta:+.4f}",
        ])

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(13, 0.5 + 0.45 * (n_rows + 1)))
    ax.axis('off')

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    # Header
    for col in range(len(col_labels)):
        tbl[(0, col)].set_facecolor('#2c5f8a')
        tbl[(0, col)].set_text_props(color='white', fontweight='bold')

    # Row colouring: highlight cells where SIFT BER < Baseline BER (SIFT wins)
    for row_idx, r in enumerate(sift_results, start=1):
        name = r['attack']
        bl   = bl_map.get(name, {})
        ber_s = r.get('BER', 0.5)
        ber_b = bl.get('BER', 0.5)
        bg = '#f0f4f8' if row_idx % 2 == 0 else 'white'
        for col in range(len(col_labels)):
            tbl[(row_idx, col)].set_facecolor(bg)
        if ber_s < ber_b:
            tbl[(row_idx, 1)].set_facecolor('#c8f0c8')   # SIFT BER — green
            tbl[(row_idx, 5)].set_facecolor('#c8f0c8')   # delta — green

    plt.title(
        'SIFT+IPVO  vs.  Baseline IPVO (no SIFT) — BER / NC per Attack',
        fontsize=12, pad=8,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
