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


# ─────────────────────────────────────────────────────────────────────────────
# Watermark debug / visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bit_heatmap(patch_size: int, location_map: list[tuple[int, int]],
                 bits: np.ndarray) -> np.ndarray:
    """
    For each 3×3 block, record the embedded bit value (0, 1) or 2 (shift/skip).

    Returns a float32 image (patch_size × patch_size) where each 3×3 block
    cell is filled with the bit value * 85 (so 0→0, 1→85, 2→170).
    """
    out = np.full((patch_size, patch_size), 2.0, dtype=np.float32)
    bit_idx = 0
    for r in range(0, patch_size - 2, 3):
        for c in range(0, patch_size - 2, 3):
            loc_max, loc_min = location_map[bit_idx] if bit_idx < len(location_map) else (2, 2)
            val = 2.0
            consumed = False
            for loc in (loc_max, loc_min):
                if loc == 3:
                    continue
                if loc == 2:
                    continue
                if bit_idx < len(bits) and bits[bit_idx] in (0, 1):
                    val = float(bits[bit_idx])
                    bit_idx += 1
                    consumed = True
            if not consumed:
                bit_idx += 1
            out[r:r + 3, c:c + 3] = val * 85.0
    return out


def export_watermark_debug_artifacts(
    img_original: np.ndarray,
    img_watermarked: np.ndarray,
    embed_data: dict,
    output_dir: str,
    n_samples: int = 4,
) -> None:
    """
    Export visualisations showing how and where watermark bits were embedded.

    Outputs (saved to <output_dir>/watermark_debug/):
      wm1_diff_full.png     – full-image difference (watermarked − original), colormap
      wm2_patch_grid.png    – for each selected patch: orig / watermarked / diff / bit-heatmap
      wm3_loc_map.png       – location map grid for each selected patch
      wm4_patch_comparison.png – row of original patches vs watermarked patches
      README.md             – description of each file + stats per keypoint

    Args:
        img_original:    uint8 grayscale original image
        img_watermarked: uint8 grayscale watermarked image
        embed_data:     dict from embed_watermark()
        output_dir:     base output directory
        n_samples:       how many keypoint patches to visualise in detail (default 4)
    """
    from src.sift_utils import extract_canonical_patch

    dbg = Path(output_dir) / "watermark_debug"
    dbg.mkdir(parents=True, exist_ok=True)

    keypoints = embed_data['keypoints']
    loc_maps  = embed_data['location_maps']
    n_embeds  = embed_data['n_embedded']
    watermark = embed_data['watermark']
    P         = embed_data['patch_size']

    # ── 1. Full-image difference ─────────────────────────────────────────────
    diff = cv2.absdiff(img_watermarked, img_original).astype(np.float32)
    diff_u8 = (np.clip(diff / (diff.max() + 1e-12) * 255, 0, 255)
               .astype(np.uint8))
    diff_color = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(dbg / "wm1_diff_full.png"), diff_color)

    # ── 2. Per-patch grids (orig / watermarked / diff / bit heatmap) ────────
    sample_kps = list(keypoints[:min(n_samples, len(keypoints))])
    ncols = 4
    nrows = len(sample_kps)

    fig, axes = plt.subplots(nrows=nrows + 1, ncols=ncols,
                             figsize=(ncols * 2.5, (nrows + 1) * 2.5))
    for ax, title in zip(axes[0], ['Original patch', 'Watermarked patch',
                                    'Difference |Δ|', 'Bit heatmap (0/1/shift)']):
        ax.axis('off')
        ax.set_title(title, fontsize=9, pad=4)

    for row_idx, kp in enumerate(sample_kps):
        kp_idx   = keypoints.index(kp)
        patch_orig = extract_canonical_patch(img_original, kp)
        patch_wm   = extract_canonical_patch(img_watermarked, kp)
        loc_map    = loc_maps[kp_idx]

        if patch_orig is None or patch_wm is None:
            for col in range(ncols):
                axes[row_idx + 1][col].axis('off')
            continue

        diff_patch = np.abs(
            patch_wm.astype(np.int16) - patch_orig.astype(np.int16)
        ).astype(np.uint8)

        bit_hm = (_bit_heatmap(P, loc_map, watermark)
                  if loc_map else np.full((P, P), 170.0))

        axes[row_idx + 1][0].imshow(patch_orig, cmap='gray')
        axes[row_idx + 1][0].axis('off')

        axes[row_idx + 1][1].imshow(patch_wm, cmap='gray')
        axes[row_idx + 1][1].axis('off')

        axes[row_idx + 1][2].imshow(diff_patch, cmap='hot')
        axes[row_idx + 1][2].axis('off')

        hm = axes[row_idx + 1][3].imshow(bit_hm, cmap='coolwarm', vmin=0, vmax=170)
        axes[row_idx + 1][3].axis('off')
        plt.colorbar(hm, ax=axes[row_idx + 1][3], fraction=0.046, pad=0.04,
                     ticks=[0, 85, 170])
        axes[row_idx + 1][3].set_title('bit=0 | bit=1 | shift', fontsize=7)

    plt.suptitle('Watermark embedding per patch (IPVO)', fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(str(dbg / "wm2_patch_grid.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── 3. Location-map grid (colour-coded per block) ───────────────────────
    nloc = len(sample_kps)
    fig2, axes2 = plt.subplots(1, nloc, figsize=(nloc * 2.5, 2.5))
    if nloc == 1:
        axes2 = [axes2]

    for ax, kp in zip(axes2, sample_kps):
        kp_idx  = keypoints.index(kp)
        loc_map = loc_maps[kp_idx]
        n_emb   = n_embeds[kp_idx]

        grid = np.zeros((P // 3, P // 3), dtype=np.float32)
        bit_idx = 0
        for r in range(P // 3):
            for c in range(P // 3):
                if bit_idx >= len(loc_map):
                    break
                loc_max, loc_min = loc_map[bit_idx]
                # 0=skip, 1=shift, 2=bit0, 3=bit1
                code = 0
                for loc, b_idx in ((loc_max, bit_idx), (loc_min, bit_idx + 1)):
                    if loc == 3:
                        continue
                    if loc == 2:
                        code = max(code, 1)
                    else:
                        if b_idx < len(watermark):
                            code = 3 if watermark[b_idx] == 1 else 2
                grid[r, c] = code * (255.0 / 3.0)
                bit_idx += 2

        ax.imshow(grid, cmap='viridis', vmin=0, vmax=255)
        ax.set_title(f'kp#{kp_idx+1}\n{n_emb}/{len(watermark)} bits', fontsize=8)
        ax.axis('off')

    cbar_ax = fig2.add_axes([0.92, 0.35, 0.015, 0.3])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    cb = matplotlib.colorbar.ColorbarBase(
        cbar_ax, cmap=plt.cm.viridis, norm=norm, orientation='vertical')
    cb.set_ticks([255 * k / 3 for k in range(4)])
    cb.set_ticklabels(['skip', 'shift', 'bit=0', 'bit=1'])
    cb.ax.tick_params(labelsize=7)

    plt.suptitle('Location Map per Patch (IPVO block decisions)', fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(str(dbg / "wm3_loc_map.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── 4. Big comparison: original patches vs watermarked patches ───────────
    n_comp = min(n_samples, len(sample_kps))
    fig3, axes3 = plt.subplots(2, n_comp, figsize=(n_comp * 2.5, 5))
    if n_comp == 1:
        axes3 = [axes3[0], axes3[1]]

    for col_idx, kp in enumerate(sample_kps[:n_comp]):
        patch_orig = extract_canonical_patch(img_original, kp)
        patch_wm   = extract_canonical_patch(img_watermarked, kp)

        axes3[0][col_idx].imshow(patch_orig, cmap='gray')
        axes3[0][col_idx].set_title(f'kp#{col_idx+1} original', fontsize=8)
        axes3[0][col_idx].axis('off')

        axes3[1][col_idx].imshow(patch_wm, cmap='gray')
        axes3[1][col_idx].set_title(f'kp#{col_idx+1} watermarked', fontsize=8)
        axes3[1][col_idx].axis('off')

    plt.suptitle('Patch Comparison: Original (top) vs Watermarked (bottom)', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(dbg / "wm4_patch_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── 5. Summary markdown ──────────────────────────────────────────────────
    summary_lines = [
        "# Watermark Debug Artifacts",
        "",
        "This folder contains intermediate visualisations of the watermark embedding.",
        "",
        "## wm1_diff_full.png",
        "Full-image difference: |watermarked − original|, colormap JET.",
        "Brighter = larger pixel change. Shows WHERE in the image the watermark",
        "was written (i.e. the canonical patches around SIFT keypoints).",
        "",
        "## wm2_patch_grid.png",
        f"Top {n_samples} keypoint patches, 4 columns:",
        "  - Column 1: original canonical patch",
        "  - Column 2: watermarked canonical patch",
        "  - Column 3: pixel difference |after − before| (hot colormap)",
        "  - Column 4: bit heatmap: blue=bit0, red=bit1, gray=shift/skip",
        "",
        "## wm3_loc_map.png",
        "Per-patch location map. Each 3×3 block = 1 cell. Colour encodes what",
        "IPVO did: skip / shift / embedded bit=0 / embedded bit=1.",
        "",
        "## wm4_patch_comparison.png",
        "Direct side-by-side original (top row) vs watermarked (bottom row) patches.",
        "",
        "## Per-keypoint stats",
        f"- Total keypoints: {len(keypoints)}",
        f"- Watermark length: {len(watermark)} bits",
        f"- Watermark bits: {watermark.tolist()}",
        "",
        "| # | Position (x,y) | σ | θ (°) | Bits embedded | Loc map len |",
        "|---|-----------------|-------|--------|---------------|-------------|",
    ]
    for i, (kp, lm, ne) in enumerate(zip(keypoints, loc_maps, n_embeds)):
        if lm is not None:
            summary_lines.append(
                f"| {i+1} | ({kp.pt[0]:.1f}, {kp.pt[1]:.1f}) "
                f"| {kp.size/2:.1f} | {kp.angle:.1f} "
                f"| {ne}/{len(watermark)} "
                f"| {len(lm)} |"
            )

    (dbg / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")
