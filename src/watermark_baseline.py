"""
Baseline watermarking — IPVO only, NO SIFT.

Embeds the same watermark into N fixed 32×32 patches placed on a regular
grid across the image (no keypoint detection, no canonical orientation).
Uses the same majority-voting extraction as the SIFT pipeline so the ONLY
difference from SIFT+IPVO is the absence of geometric normalisation.

Expected behaviour:
  - No-attack   → BER ≈ 0  (same as SIFT pipeline)
  - Translation → BER ≈ 0.5  (fixed coords shift out of patch content)
  - Rotation / Scale / Crop / Affine → BER ≈ 0.5
This contrasts with SIFT+IPVO, which handles translation well.
"""

import numpy as np
import cv2

from src import pvo

PATCH_SIZE = 32          # must match sift_utils.PATCH_SIZE


# ─────────────────────────────────────────────────────────────────────────────

def _grid_positions(img_shape: tuple[int, int], n: int) -> list[tuple[int, int]]:
    """
    Return top-left (row, col) corners for n non-overlapping patches
    placed on an evenly-spaced grid.  Patches are clipped to stay inside.
    """
    h, w = img_shape
    cols_n = max(1, int(np.ceil(np.sqrt(n * w / h))))
    rows_n = max(1, int(np.ceil(n / cols_n)))

    step_c = max(PATCH_SIZE, w // cols_n)
    step_r = max(PATCH_SIZE, h // rows_n)

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
) -> tuple[np.ndarray, dict]:
    """
    Embed watermark into n_patches fixed grid locations using IPVO.

    Returns:
        (img_watermarked, embed_data_baseline)
    """
    assert img.ndim == 2, "Grayscale only"
    positions   = _grid_positions(img.shape, n_patches)
    img_out     = img.copy()
    location_maps: list = []
    n_embedded_list: list[int] = []

    for (r, c) in positions:
        patch = img_out[r:r + PATCH_SIZE, c:c + PATCH_SIZE].copy()
        patch_wm, loc_map, n_emb = pvo.embed(patch, watermark)
        img_out[r:r + PATCH_SIZE, c:c + PATCH_SIZE] = patch_wm
        location_maps.append(loc_map)
        n_embedded_list.append(n_emb)

    embed_data = {
        'positions':     positions,
        'location_maps': location_maps,
        'watermark':     watermark.copy(),
        'patch_size':    PATCH_SIZE,
        'n_embedded':    n_embedded_list,
    }
    return img_out, embed_data


def extract_baseline(
    img_attacked: np.ndarray,
    embed_data: dict,
) -> np.ndarray:
    """
    Extract watermark from fixed grid positions using majority voting.

    Returns:
        1-D uint8 array of extracted watermark bits.
    """
    assert img_attacked.ndim == 2, "Grayscale only"
    h, w = img_attacked.shape

    positions    = embed_data['positions']
    location_maps= embed_data['location_maps']
    wm_length    = len(embed_data['watermark'])

    votes = np.zeros(wm_length, dtype=np.int32)
    count = 0

    for (r, c), loc_map in zip(positions, location_maps):
        # Patch may be partially out of bounds after crop/affine attack
        r = min(r, h - PATCH_SIZE)
        c = min(c, w - PATCH_SIZE)
        if r < 0 or c < 0:
            continue
        patch = img_attacked[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            continue
        bits = pvo.extract(patch, loc_map, wm_length)
        votes += bits.astype(np.int32)
        count += 1

    if count == 0:
        return np.zeros(wm_length, dtype=np.uint8)
    return (votes > count / 2).astype(np.uint8)


def grid_patch_keypoints(positions: list[tuple[int, int]]) -> list:
    """
    Convert grid (row, col) positions to fake cv2.KeyPoint objects so we can
    reuse draw_embedded_patches / draw_extracted_patches visualisations.
    """
    kps = []
    half = PATCH_SIZE / 2
    for (r, c) in positions:
        kp = cv2.KeyPoint(
            x=float(c + half),
            y=float(r + half),
            size=float(PATCH_SIZE),
        )
        kps.append(kp)
    return kps
