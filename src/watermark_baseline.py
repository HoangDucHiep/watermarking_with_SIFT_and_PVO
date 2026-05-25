"""
Baseline watermarking — Spread Spectrum KHÔNG có SIFT.

Nhúng watermark vào N patch cố định trên lưới đều (không dùng keypoint).
Dùng cùng thuật toán SS với SIFT+SS để so sánh công bằng:
  → Điểm khác biệt duy nhất là KHÔNG có SIFT geometric normalisation.

Kỳ vọng:
  - No-attack:   BER ≈ 0   (patch content giữ nguyên)
  - Translation: BER ≈ 0.5 (lưới cố định, pixel dịch ra khỏi vùng patch)
  - Rotation / Scale / Crop / Affine: BER ≈ 0.5 (patch content bị biến đổi)
"""

import numpy as np
import cv2

from src import ss

PATCH_SIZE = 32      # phải khớp với PATCH_SIZE trong sift_utils.py


# ─────────────────────────────────────────────────────────────────────────────

def _grid_positions(img_shape: tuple[int, int], n: int) -> list[tuple[int, int]]:
    """
    Trả về n vị trí (row, col) góc trên-trái của các patch trên lưới đều.
    """
    h, w    = img_shape
    cols_n  = max(1, int(np.ceil(np.sqrt(n * w / h))))
    rows_n  = max(1, int(np.ceil(n / cols_n)))
    step_c  = max(PATCH_SIZE, w // cols_n)
    step_r  = max(PATCH_SIZE, h // rows_n)

    positions: list[tuple[int, int]] = []
    for ri in range(rows_n):
        for ci in range(cols_n):
            r = ri * step_r + (step_r - PATCH_SIZE) // 2
            c = ci * step_c + (step_c - PATCH_SIZE) // 2
            r = min(r, h - PATCH_SIZE)
            c = min(c, w - PATCH_SIZE)
            positions.append((r, c))
            if len(positions) == n:
                return positions
    return positions[:n]


# ─────────────────────────────────────────────────────────────────────────────

def embed_baseline(
    img: np.ndarray,
    watermark: np.ndarray,
    n_patches: int = 20,
    alpha: float = ss.ALPHA,
    seed: int = ss.SS_SEED,
) -> tuple[np.ndarray, dict]:
    """
    Nhúng watermark vào n_patches patch cố định trên lưới dùng Spread Spectrum.

    Returns:
        img_wm:     ảnh đã nhúng
        embed_data: dict với keys 'positions', 'watermark', 'patch_size', 'alpha', 'seed'
    """
    assert img.ndim == 2, "Grayscale only"
    positions = _grid_positions(img.shape, n_patches)
    img_out   = img.copy()

    for (r, c) in positions:
        patch    = img_out[r:r + PATCH_SIZE, c:c + PATCH_SIZE].copy()
        patch_wm = ss.embed(patch, watermark, seed=seed, alpha=alpha)
        img_out[r:r + PATCH_SIZE, c:c + PATCH_SIZE] = patch_wm

    return img_out, {
        'positions':  positions,
        'watermark':  watermark.copy(),
        'patch_size': PATCH_SIZE,
        'alpha':      alpha,
        'seed':       seed,
    }


def extract_baseline(
    img_attacked: np.ndarray,
    embed_data: dict,
) -> np.ndarray:
    """
    Giải mã watermark từ các patch cố định bằng SS correlation + majority voting.

    Returns:
        1-D uint8 array — watermark đã giải mã
    """
    assert img_attacked.ndim == 2, "Grayscale only"
    h, w      = img_attacked.shape
    positions = embed_data['positions']
    wm_length = len(embed_data['watermark'])
    seed      = embed_data['seed']

    votes = np.zeros(wm_length, dtype=np.int32)
    count = 0

    for (r, c) in positions:
        r = min(r, h - PATCH_SIZE)
        c = min(c, w - PATCH_SIZE)
        if r < 0 or c < 0:
            continue
        patch = img_attacked[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            continue

        bits   = ss.extract(patch, wm_length, seed=seed)
        votes += bits.astype(np.int32)
        count += 1

    if count == 0:
        return np.zeros(wm_length, dtype=np.uint8)
    return (votes > count / 2).astype(np.uint8)


def grid_patch_keypoints(positions: list[tuple[int, int]]) -> list:
    """Chuyển vị trí lưới → fake cv2.KeyPoint để dùng hàm visualise."""
    half = PATCH_SIZE / 2
    return [
        cv2.KeyPoint(x=float(c + half), y=float(r + half), size=float(PATCH_SIZE))
        for (r, c) in positions
    ]
