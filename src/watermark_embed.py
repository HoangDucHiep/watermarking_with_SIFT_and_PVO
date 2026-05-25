"""
Watermark embedding — SIFT (cv2) + Spread Spectrum.

Các bước:
  1. Phát hiện N SIFT keypoints mạnh nhất, không chồng lấp
  2. Mỗi keypoint → canonical patch (32×32, chuẩn hoá góc quay & scale)
  3. Nhúng toàn bộ watermark vào patch bằng Spread Spectrum (ss.embed)
  4. Ghi patch đã sửa trở lại ảnh
  5. Lưu keypoints + descriptors + seed → dùng lại khi extract
"""

import cv2
import numpy as np
import pickle

from src.sift_utils import (
    extract_canonical_patch,
    put_canonical_patch,
    PATCH_SIZE,
    SCALE_FACTOR,
    MIN_KP_DIST,
)
from src import ss


def embed_watermark(
    img: np.ndarray,
    watermark: np.ndarray,
    n_keypoints: int = 20,
    alpha: float = ss.ALPHA,
    seed: int = ss.SS_SEED,
) -> tuple[np.ndarray, dict]:
    """
    Nhúng watermark vào ảnh dùng SIFT keypoints + Spread Spectrum.

    Args:
        img:         Ảnh xám uint8 (H × W)
        watermark:   Mảng nhị phân 1-D (0/1), tối đa 200 bits với patch 32×32
        n_keypoints: Số SIFT keypoints (mỗi keypoint nhúng 1 bản watermark)
        alpha:       Cường độ nhúng (cao → robust hơn nhưng PSNR thấp hơn)
        seed:        Seed sinh PN sequences (phải giống khi extract)

    Returns:
        img_wm:      Ảnh đã nhúng watermark, uint8
        embed_data:  Dict chứa thông tin cần thiết để extract:
                       'keypoints'   – list cv2.KeyPoint
                       'descriptors' – ndarray (n, 128) float32
                       'watermark'   – watermark gốc (để đánh giá BER)
                       'patch_size'  – kích thước canonical patch
                       'alpha'       – alpha đã dùng
                       'seed'        – seed đã dùng
    """
    assert img.ndim == 2, "Input phải là ảnh xám (2-D)"

    # ── Phát hiện SIFT keypoints ───────────────────────────────────────────────
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10)
    _raw, all_descs = sift.detectAndCompute(img, None)
    raw_kps: list = list(_raw) if _raw is not None else []

    if not raw_kps or all_descs is None:
        raise RuntimeError("SIFT không tìm được keypoint nào trong ảnh")

    # Lưu index gốc vào class_id trước khi sắp xếp
    for i, kp in enumerate(raw_kps):
        kp.class_id = i

    # ── Chọn N keypoints mạnh nhất, không chồng lấp ───────────────────────────
    raw_kps.sort(key=lambda kp: kp.response, reverse=True)
    h_img, w_img = img.shape[:2]
    selected: list[cv2.KeyPoint] = []

    for kp in raw_kps:
        x, y  = kp.pt
        sigma = kp.size / 2.0
        src_r = sigma * SCALE_FACTOR + 2          # bán kính vùng lấy patch

        # Patch phải nằm hoàn toàn trong ảnh
        if x - src_r < 0 or x + src_r >= w_img or y - src_r < 0 or y + src_r >= h_img:
            continue
        # Cách ít nhất MIN_KP_DIST so với keypoints đã chọn
        if any(np.hypot(x - s.pt[0], y - s.pt[1]) < MIN_KP_DIST for s in selected):
            continue

        selected.append(kp)
        if len(selected) >= n_keypoints:
            break

    if not selected:
        raise RuntimeError(f"Không đủ keypoints hợp lệ (yêu cầu {n_keypoints})")

    # ── Nhúng SS vào mỗi canonical patch ──────────────────────────────────────
    img_out = img.copy()
    for kp in selected:
        patch = extract_canonical_patch(img, kp)
        if patch is None:
            continue
        patch_wm = ss.embed(patch, watermark, seed=seed, alpha=alpha)
        img_out  = put_canonical_patch(img_out, kp, patch_wm)

    # ── Lưu descriptors cho các keypoints đã chọn ─────────────────────────────
    selected_descs = all_descs[[kp.class_id for kp in selected]]

    embed_data = {
        'keypoints':   selected,
        'descriptors': selected_descs,
        'watermark':   watermark.copy(),
        'patch_size':  PATCH_SIZE,
        'alpha':       alpha,
        'seed':        seed,
    }
    return img_out, embed_data


def save_embed_data(embed_data: dict, path: str) -> None:
    """Lưu embed_data ra file pickle."""
    data = embed_data.copy()
    data['keypoints'] = [
        (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in embed_data['keypoints']
    ]
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_embed_data(path: str) -> dict:
    """Load embed_data từ file pickle."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    data['keypoints'] = [
        cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle,
                     response=response, octave=octave, class_id=class_id)
        for pt, size, angle, response, octave, class_id in data['keypoints']
    ]
    return data
