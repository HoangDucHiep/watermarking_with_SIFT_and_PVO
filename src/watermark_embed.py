"""
Watermark embedding pipeline.

Steps:
  1. Detect N stable SIFT keypoints in the host image
  2. For each keypoint: extract canonical patch → embed watermark via IPVO
  3. Write modified patch back into image
  4. Save keypoints + location maps to a sidecar file

Outputs:
  - watermarked image (uint8 ndarray)
  - embed_data dict (keypoints, location_maps, watermark, patch_size)
"""

import cv2
import numpy as np
import pickle
from pathlib import Path

from src.sift_utils import detect_keypoints, extract_canonical_patch, put_canonical_patch, PATCH_SIZE
from src import pvo


def embed_watermark(
    img: np.ndarray,
    watermark: np.ndarray,
    n_keypoints: int = 20,
) -> tuple[np.ndarray, dict]:
    """
    Embed a binary watermark into the image using SIFT + IPVO.

    Args:
        img:         Grayscale uint8 image
        watermark:   1D binary array (0/1), length ≤ capacity per patch
        n_keypoints: Number of SIFT keypoints to use

    Returns:
        (watermarked_img, embed_data)

        embed_data keys:
          'keypoints'     – list of cv2.KeyPoint
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

    keypoints = detect_keypoints(img, n_keypoints)
    if not keypoints:
        raise RuntimeError("No stable SIFT keypoints found in image")

    # Compute SIFT descriptors for original keypoints (used for matching at extraction)
    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(img, keypoints)   # shape (n, 128)

    img_out = img.copy()
    location_maps = []
    n_embedded_list = []

    for kp in keypoints:
        patch = extract_canonical_patch(img, kp)
        if patch is None:
            location_maps.append(None)
            n_embedded_list.append(0)
            continue

        modified_patch, loc_map, n_emb = pvo.embed(patch, watermark)
        location_maps.append(loc_map)
        n_embedded_list.append(n_emb)

        img_out = put_canonical_patch(img_out, kp, modified_patch)

    embed_data = {
        'keypoints':     keypoints,
        'descriptors':   descriptors,        # float32 (n, 128)
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
