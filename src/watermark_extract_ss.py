"""
Watermark extraction — SIFT descriptor matching + Spread Spectrum correlation.

Giống watermark_extract.py nhưng dùng ss.extract thay vì pvo.extract.
Không cần location_map — chỉ cần seed (giống lúc embed).
"""

import numpy as np
import cv2

from src.sift_utils import extract_canonical_patch
from src import ss

_LOWE_RATIO = 0.75


def extract_watermark_ss(
    img_attacked: np.ndarray,
    embed_data: dict,
) -> tuple[np.ndarray, list, list, int]:
    """
    Giải mã watermark từ ảnh bị tấn công dùng SIFT matching + SS correlation.

    Steps:
        1. Re-detect SIFT keypoints trên ảnh bị tấn công
        2. Match với descriptors gốc (BFMatcher + Lowe ratio test)
        3. Với mỗi keypoint khớp: extract canonical patch → ss.extract
        4. Majority voting trên tất cả patch → watermark cuối

    Returns:
        (wm_final, surviving_kps, lost_orig_indices, n_located)
    """
    assert img_attacked.ndim == 2, "Input phải là ảnh xám"

    original_kps  = embed_data['keypoints']
    orig_descs    = embed_data['descriptors']
    wm_length     = len(embed_data['watermark'])
    seed          = embed_data['seed']

    # ── Re-detect trên ảnh bị tấn công ───────────────────────────────────────
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01, edgeThreshold=20)
    _new_kps, new_descs = sift.detectAndCompute(img_attacked, None)
    new_kps: list = list(_new_kps) if _new_kps is not None else []

    if not new_kps or new_descs is None or orig_descs is None:
        return (
            np.zeros(wm_length, dtype=np.uint8),
            [], list(range(len(original_kps))), 0,
        )

    # ── Descriptor matching (BFMatcher + Lowe) ────────────────────────────────
    bf         = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(orig_descs, new_descs, k=2)

    matched_new_kps: list[cv2.KeyPoint] = []
    orig_indices: list[int] = []
    used_new: set[int] = set()

    for orig_idx, matches in enumerate(raw_matches):
        if len(matches) < 2:
            continue
        m, n = matches[0], matches[1]
        if m.distance < _LOWE_RATIO * n.distance:
            new_idx = m.trainIdx
            if new_idx not in used_new:
                matched_new_kps.append(new_kps[new_idx])
                orig_indices.append(orig_idx)
                used_new.add(new_idx)

    n_located = len(matched_new_kps)

    # ── SS correlation + majority voting ──────────────────────────────────────
    votes   = np.zeros(wm_length, dtype=np.int32)
    count   = 0
    surviving_kps: list[cv2.KeyPoint] = []
    lost_set: set[int] = set(range(len(original_kps)))

    for new_kp, orig_idx in zip(matched_new_kps, orig_indices):
        patch = extract_canonical_patch(img_attacked, new_kp)
        if patch is None:
            continue

        bits = ss.extract(patch, wm_length, seed=seed)
        votes += bits.astype(np.int32)
        count += 1
        surviving_kps.append(new_kp)
        lost_set.discard(orig_idx)

    # ── Majority voting ────────────────────────────────────────────────────────
    if count == 0:
        wm_final = np.zeros(wm_length, dtype=np.uint8)
    else:
        wm_final = (votes > count / 2).astype(np.uint8)

    return wm_final, surviving_kps, sorted(lost_set), n_located
