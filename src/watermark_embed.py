"""
Watermark embedding pipeline — SIFT (cv2) + IPVO.

Both detection and descriptor computation use cv2.SIFT at embed time,
ensuring consistency with cv2.SIFT used at extract time.

Steps:
  1. Detect N stable SIFT keypoints using cv2.SIFT
  2. Select top-N non-overlapping keypoints by response
  3. For each keypoint: extract canonical patch → embed watermark via IPVO
  4. Write modified patch back into image
  5. Save keypoints + descriptors + location maps to a sidecar file

Outputs:
  - watermarked image (uint8 ndarray)
  - embed_data dict (keypoints, descriptors, location_maps, watermark, patch_size)
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
from src import pvo


def embed_watermark(
    img: np.ndarray,
    watermark: np.ndarray,
    n_keypoints: int = 20,
) -> tuple[np.ndarray, dict]:
    """
    Embed a binary watermark into the image using cv2.SIFT + IPVO.

    Both detection and descriptor computation use cv2.SIFT, ensuring
    descriptor consistency between embed and extract pipelines.

    Args:
        img:         Grayscale uint8 image
        watermark:   1D binary array (0/1), length ≤ capacity per patch
        n_keypoints: Number of SIFT keypoints to use

    Returns:
        (watermarked_img, embed_data)

        embed_data keys:
          'keypoints'     – list of cv2.KeyPoint (from cv2.SIFT)
          'descriptors'   – ndarray (n, 128), float32, from cv2.SIFT
          'location_maps' – list of location maps, one per keypoint
          'watermark'     – original watermark bits
          'patch_size'    – PATCH_SIZE constant
          'n_embedded'    – bits actually embedded per patch (list)
    """
    assert img.ndim == 2, "Input must be a grayscale image"
    capacity = pvo.capacity_for_patch(PATCH_SIZE)
    assert len(watermark) <= capacity, (
        f"Watermark length {len(watermark)} exceeds patch capacity {capacity}"
    )

    # ── Detect keypoints using cv2 SIFT ─────────────────────────────────────────
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10)
    _raw, all_descs = sift.detectAndCompute(img, None)
    # Ensure keypoints are a list (not tuple) so we can sort in-place
    raw_kps: list = list(_raw) if _raw is not None else []

    if not raw_kps or all_descs is None:
        raise RuntimeError("cv2.SIFT found no keypoints in image")

    # Save original indices before sorting (descriptors[i] corresponds to raw_kps[i])
    for i, kp in enumerate(raw_kps):
        kp.class_id = i

    # ── Select top-N non-overlapping keypoints ──────────────────────────────────
    # Sort by response (strength) descending
    raw_kps.sort(key=lambda kp: kp.response, reverse=True)

    h_img, w_img = img.shape[:2]
    selected: list[cv2.KeyPoint] = []

    for kp in raw_kps:
        x, y = kp.pt
        sigma = kp.size / 2.0
        src_r = sigma * SCALE_FACTOR + 2

        # Patch must fit entirely inside the image
        if x - src_r < 0 or x + src_r >= w_img or y - src_r < 0 or y + src_r >= h_img:
            continue

        # Must be at least MIN_KP_DIST away from already-selected keypoints
        if any(np.hypot(x - s.pt[0], y - s.pt[1]) < MIN_KP_DIST for s in selected):
            continue

        selected.append(kp)
        if len(selected) >= n_keypoints:
            break

    if not selected:
        raise RuntimeError(
            f"No stable keypoints found after filtering (requested {n_keypoints})"
        )

    # ── Embed watermark into each selected keypoint's canonical patch ────────────
    img_out = img.copy()
    location_maps = []
    n_embedded_list = []

    for kp in selected:
        patch = extract_canonical_patch(img, kp)
        if patch is None:
            location_maps.append(None)
            n_embedded_list.append(0)
            continue

        modified_patch, loc_map, n_emb = pvo.embed(patch, watermark)
        location_maps.append(loc_map)
        n_embedded_list.append(n_emb)
        img_out = put_canonical_patch(img_out, kp, modified_patch)

    # Extract descriptors for the selected keypoints using saved original indices
    selected_indices = [kp.class_id for kp in selected]
    selected_descs = all_descs[selected_indices]

    embed_data = {
        'keypoints':     selected,
        'descriptors':   selected_descs,   # (n_selected, 128), float32
        'location_maps': location_maps,
        'watermark':     watermark.copy(),
        'patch_size':    PATCH_SIZE,
        'n_embedded':    n_embedded_list,
    }

    return img_out, embed_data


def save_embed_data(embed_data: dict, path: str) -> None:
    """Serialize embed_data to a pickle file."""
    # Convert KeyPoint objects to serializable tuples
    kp_serial = [
        (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in embed_data['keypoints']
    ]
    data = embed_data.copy()
    data['keypoints'] = kp_serial
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_embed_data(path: str) -> dict:
    """Load embed_data from a pickle file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    data['keypoints'] = [
        cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle,
                     response=response, octave=octave, class_id=class_id)
        for pt, size, angle, response, octave, class_id in data['keypoints']
    ]
    return data
