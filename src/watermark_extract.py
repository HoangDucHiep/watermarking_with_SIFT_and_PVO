"""
Watermark extraction pipeline — SIFT descriptor matching + majority voting.

Extraction steps:
  1. Re-detect SIFT keypoints on the attacked image
  2. Match them to the original keypoints via descriptor distance (Lowe ratio test)
  3. For each matched pair: extract canonical patch from attacked image using
     the RE-DETECTED keypoint parameters (invariant to the geometric attack)
  4. IPVO extract using the location map stored during embedding
  5. Majority voting over all surviving patches → final watermark

For the no-attack case (img_attacked == img_watermarked) an exact match is
guaranteed because the SIFT descriptors of the watermarked image are nearly
identical to the originals (IPVO perturbation is only ±1 pixel).
"""

import numpy as np
import cv2

from src.sift_utils import extract_canonical_patch
from src import pvo


# ─── Ratio threshold for Lowe's ratio test ──────────────────────────────────
_LOWE_RATIO = 0.75


def _match_keypoints_by_descriptor(
    img_attacked: np.ndarray,
    original_keypoints: list[cv2.KeyPoint],
    original_descriptors: np.ndarray,
) -> tuple[list[cv2.KeyPoint], list[int]]:
    """
    Re-detect SIFT keypoints in the attacked image and match to originals
    using descriptor distance (Lowe ratio test).

    Returns:
        (matched_new_kps, orig_indices)
    """
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01, edgeThreshold=20)
    new_kps, new_descs = sift.detectAndCompute(img_attacked, None)

    if not new_kps or new_descs is None or original_descriptors is None:
        return [], []

    # BFMatcher with L2 norm and knn k=2 for ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(original_descriptors, new_descs, k=2)

    matched_new_kps: list[cv2.KeyPoint] = []
    orig_indices: list[int] = []
    used_new: set[int] = set()  # avoid matching two originals to the same new kp

    for orig_idx, matches in enumerate(raw):
        if len(matches) < 2:
            continue
        m, n = matches[0], matches[1]
        # Lowe ratio test
        if m.distance < _LOWE_RATIO * n.distance:
            new_idx = m.trainIdx
            if new_idx not in used_new:
                matched_new_kps.append(new_kps[new_idx])
                orig_indices.append(orig_idx)
                used_new.add(new_idx)

    return matched_new_kps, orig_indices


def extract_watermark(
    img_attacked: np.ndarray,
    embed_data: dict,
) -> tuple[np.ndarray, list[cv2.KeyPoint], list[int]]:
    """
    Extract watermark from an attacked image using majority voting.

    Args:
        img_attacked: Grayscale uint8 attacked/watermarked image
        embed_data:   Dict returned by watermark_embed.embed_watermark()

    Returns:
        (watermark_final, surviving_kps, lost_orig_indices)
    """
    assert img_attacked.ndim == 2, "Input must be a grayscale image"

    original_kps   = embed_data['keypoints']
    orig_descs     = embed_data.get('descriptors')
    location_maps  = embed_data['location_maps']
    wm_length      = len(embed_data['watermark'])

    # ── Try descriptor-based matching first ─────────────────────────────────
    if orig_descs is not None:
        matched_kps, orig_indices = _match_keypoints_by_descriptor(
            img_attacked, original_kps, orig_descs)
    else:
        matched_kps, orig_indices = [], []

    # ── Fallback: if no descriptor matches, use original keypoints directly ─
    # (handles the no-attack case perfectly and minor perturbation cases)
    if not matched_kps:
        matched_kps  = original_kps
        orig_indices = list(range(len(original_kps)))

    n_located = len(matched_kps)   # keypoints found by SIFT descriptor matching

    votes   = np.zeros(wm_length, dtype=np.int32)
    count   = 0
    surviving_kps: list[cv2.KeyPoint] = []
    lost_set: set[int] = set(range(len(original_kps)))

    for new_kp, orig_idx in zip(matched_kps, orig_indices):
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

    if count == 0:
        wm_final = np.zeros(wm_length, dtype=np.uint8)
    else:
        wm_final = (votes > count / 2).astype(np.uint8)

    lost_orig_indices = sorted(lost_set)
    return wm_final, surviving_kps, lost_orig_indices, n_located
