"""
Watermark extraction pipeline — cv2.SIFT descriptor matching + majority voting.

Extraction steps:
  1. Re-detect SIFT keypoints on the attacked image using cv2.SIFT
  2. Match them to the original descriptors via BFMatcher + Lowe ratio test
  3. For each matched pair: extract canonical patch from attacked image using
     the RE-DETECTED keypoint parameters (invariant to the geometric attack)
  4. IPVO extract using the location map stored during embedding
  5. Majority voting over all surviving patches → final watermark

No fallback is used. If a keypoint cannot be matched by descriptor distance,
it is simply lost. This maintains full consistency between embed and extract
pipelines (both use cv2.SIFT).
"""

import numpy as np
import cv2

from src.sift_utils import extract_canonical_patch
from src import pvo


# ─── Ratio threshold for Lowe's ratio test ──────────────────────────────────
_LOWE_RATIO = 0.75


def extract_watermark(
    img_attacked: np.ndarray,
    embed_data: dict,
) -> tuple[np.ndarray, list[cv2.KeyPoint], list[int]]:
    """
    Extract watermark from an attacked image using majority voting.

    Both detection and descriptor computation use cv2.SIFT at extract time,
    ensuring consistency with the cv2 SIFT descriptors stored during embedding.

    Extraction steps:
      1. Re-detect SIFT keypoints on the attacked image using cv2.SIFT
      2. Match them to the original descriptors via BFMatcher + Lowe ratio test
      3. For each matched pair: extract canonical patch using the new keypoint
      4. IPVO extract using the location map stored during embedding
      5. Majority voting over all surviving patches → final watermark

    No fallback is used. If a keypoint cannot be matched, it is simply lost.
    """
    assert img_attacked.ndim == 2, "Input must be a grayscale image"

    original_kps   = embed_data['keypoints']
    orig_descs     = embed_data['descriptors']
    location_maps  = embed_data['location_maps']
    wm_length      = len(embed_data['watermark'])

    # ── Detect keypoints on attacked image using cv2 SIFT ──────────────────────
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01, edgeThreshold=20)
    _new_kps, new_descs = sift.detectAndCompute(img_attacked, None)
    # Ensure keypoints are a list (not tuple) so we can index safely
    new_kps: list = list(_new_kps) if _new_kps is not None else []

    if not new_kps or new_descs is None or orig_descs is None:
        # No keypoints detected → cannot extract
        return (
            np.zeros(wm_length, dtype=np.uint8),
            [],                      # surviving_kps
            list(range(len(original_kps))),  # all original keypoints are lost
            0,                      # n_located
        )

    # ── Match descriptors via BFMatcher + Lowe ratio test ────────────────────
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(orig_descs, new_descs, k=2)

    matched_new_kps: list[cv2.KeyPoint] = []
    orig_indices: list[int] = []
    used_new: set[int] = set()   # avoid matching two originals to the same new kp

    for orig_idx, matches in enumerate(raw_matches):
        if len(matches) < 2:
            continue
        m, n = matches[0], matches[1]
        # Lowe ratio test: best match must be sufficiently better than second best
        if m.distance < _LOWE_RATIO * n.distance:
            new_idx = m.trainIdx
            if new_idx not in used_new:
                matched_new_kps.append(new_kps[new_idx])
                orig_indices.append(orig_idx)
                used_new.add(new_idx)

    n_located = len(matched_new_kps)

    # ── Extract watermark from each matched keypoint ───────────────────────────
    votes   = np.zeros(wm_length, dtype=np.int32)
    count   = 0
    surviving_kps: list[cv2.KeyPoint] = []
    lost_set: set[int] = set(range(len(original_kps)))

    for new_kp, orig_idx in zip(matched_new_kps, orig_indices):
        loc_map = location_maps[orig_idx]
        if loc_map is None:
            continue

        patch = extract_canonical_patch(img_attacked, new_kp)
        if patch is None:
            continue

        bits = pvo.extract(patch, loc_map, wm_length)
        votes += bits.astype(np.int32)
        count += 1
        surviving_kps.append(new_kp)
        lost_set.discard(orig_idx)

    # ── Majority voting ────────────────────────────────────────────────────────
    if count == 0:
        wm_final = np.zeros(wm_length, dtype=np.uint8)
    else:
        wm_final = (votes > count / 2).astype(np.uint8)

    lost_orig_indices = sorted(lost_set)
    return wm_final, surviving_kps, lost_orig_indices, n_located
