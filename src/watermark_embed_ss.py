"""
Watermark embedding — SIFT (cv2) + Spread Spectrum.

Giống hệt watermark_embed.py nhưng dùng ss.embed thay vì pvo.embed.
Không cần lưu location_map (SS chỉ cần seed để tái tạo PN sequences).
"""

import cv2
import numpy as np
import pickle
from pathlib import Path

from src.sift_utils import (
    extract_canonical_patch,
    put_canonical_patch,
    PATCH_SIZE,
    SCALE_FACTOR,
    MIN_KP_DIST,
)
from src import ss


def embed_watermark_ss(
    img: np.ndarray,
    watermark: np.ndarray,
    n_keypoints: int = 20,
    alpha: float = ss.ALPHA,
    seed: int = ss.SS_SEED,
) -> tuple[np.ndarray, dict]:
    """
    Nhúng watermark vào ảnh dùng SIFT keypoints + Spread Spectrum.

    Args:
        img:         Ảnh xám uint8
        watermark:   Mảng nhị phân 1-D (0/1)
        n_keypoints: Số SIFT keypoints dùng để nhúng
        alpha:       Cường độ nhúng SS
        seed:        Seed tạo PN sequences (phải khớp khi extract)

    Returns:
        (img_wm, embed_data_ss)
        embed_data_ss keys:
            'keypoints'   – list cv2.KeyPoint
            'descriptors' – ndarray (n, 128) float32
            'watermark'   – mảng watermark gốc
            'patch_size'  – PATCH_SIZE
            'alpha'       – alpha dùng khi embed
            'seed'        – seed dùng khi embed
    """
    assert img.ndim == 2, "Input phải là ảnh xám"

    # ── Detect SIFT keypoints ──────────────────────────────────────────────────
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10)
    _raw, all_descs = sift.detectAndCompute(img, None)
    raw_kps: list = list(_raw) if _raw is not None else []

    if not raw_kps or all_descs is None:
        raise RuntimeError("SIFT không tìm được keypoint nào")

    for i, kp in enumerate(raw_kps):
        kp.class_id = i

    # ── Chọn N keypoint không chồng lấp ───────────────────────────────────────
    raw_kps.sort(key=lambda kp: kp.response, reverse=True)
    h_img, w_img = img.shape[:2]
    selected: list[cv2.KeyPoint] = []

    for kp in raw_kps:
        x, y   = kp.pt
        sigma  = kp.size / 2.0
        src_r  = sigma * SCALE_FACTOR + 2

        if x - src_r < 0 or x + src_r >= w_img or y - src_r < 0 or y + src_r >= h_img:
            continue
        if any(np.hypot(x - s.pt[0], y - s.pt[1]) < MIN_KP_DIST for s in selected):
            continue

        selected.append(kp)
        if len(selected) >= n_keypoints:
            break

    if not selected:
        raise RuntimeError(f"Không đủ keypoint ({n_keypoints} requested)")

    # ── Nhúng SS vào mỗi canonical patch ──────────────────────────────────────
    img_out = img.copy()

    for kp in selected:
        patch = extract_canonical_patch(img, kp)
        if patch is None:
            continue
        patch_wm = ss.embed(patch, watermark, seed=seed, alpha=alpha)
        img_out  = put_canonical_patch(img_out, kp, patch_wm)

    # ── Lấy descriptors cho keypoints đã chọn ─────────────────────────────────
    selected_indices = [kp.class_id for kp in selected]
    selected_descs   = all_descs[selected_indices]

    embed_data = {
        'keypoints':   selected,
        'descriptors': selected_descs,
        'watermark':   watermark.copy(),
        'patch_size':  PATCH_SIZE,
        'alpha':       alpha,
        'seed':        seed,
    }
    return img_out, embed_data


def save_embed_data_ss(embed_data: dict, path: str) -> None:
    """Lưu embed_data SS ra file pickle."""
    kp_serial = [
        (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in embed_data['keypoints']
    ]
    data = embed_data.copy()
    data['keypoints'] = kp_serial
    with open(path, 'wb') as f:
        pickle.dump(data, f)
