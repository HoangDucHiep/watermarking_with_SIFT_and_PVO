"""
Geometric attack simulations for watermarking robustness testing.

All functions accept a uint8 grayscale image and return a uint8 image
of the SAME size as the input (resize/pad back as needed).
"""

import cv2
import numpy as np


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by `angle` degrees (counter-clockwise).
    Fills gaps with reflected border to avoid black regions.

    Args:
        img:   Grayscale uint8 image
        angle: Rotation angle in degrees

    Returns:
        Rotated image, same size as input
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return rotated


def scale(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale the image by `factor` then resize back to original dimensions.

    A factor > 1 zooms in (upscale then crop center).
    A factor < 1 shrinks (downscale then pad).

    Args:
        img:    Grayscale uint8 image
        factor: Scale factor (e.g. 0.8 = 80%, 1.2 = 120%)

    Returns:
        Scaled image, same size as input
    """
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)

    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if factor >= 1.0:
        # Crop centre
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        result = scaled[y0:y0 + h, x0:x0 + w]
    else:
        # Pad with reflection to fill back to original size
        pad_h = h - new_h
        pad_w = w - new_w
        top  = pad_h // 2;  bot  = pad_h - top
        left = pad_w // 2;  right = pad_w - left
        result = cv2.copyMakeBorder(scaled, top, bot, left, right,
                                    borderType=cv2.BORDER_REFLECT)

    # Ensure exact original size
    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    return result


def translate(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """
    Translate image by (tx, ty) pixels.
    Rolls the image so no black border appears (circular shift).

    Args:
        img: Grayscale uint8 image
        tx:  Horizontal shift in pixels (positive = right)
        ty:  Vertical shift in pixels   (positive = down)

    Returns:
        Translated image, same size as input
    """
    result = np.roll(img, ty, axis=0)
    result = np.roll(result, tx, axis=1)
    return result


def crop(img: np.ndarray, ratio: float) -> np.ndarray:
    """
    Crop `ratio` fraction from each side, then resize back to original.

    Args:
        img:   Grayscale uint8 image
        ratio: Fraction of each side to remove (e.g. 0.1 = 10%)

    Returns:
        Cropped-and-resized image, same size as input
    """
    h, w = img.shape[:2]
    dy = int(h * ratio)
    dx = int(w * ratio)
    cropped = img[dy:h - dy, dx:w - dx]
    result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return result


def affine(img: np.ndarray,
           shear: float = 0.1,
           angle: float = 5.0,
           scale_x: float = 1.0,
           scale_y: float = 1.0) -> np.ndarray:
    """
    Apply a general affine transformation (shear + rotation + scale).

    Args:
        img:     Grayscale uint8 image
        shear:   Shear factor along x axis
        angle:   Additional rotation in degrees
        scale_x: X-axis scale factor
        scale_y: Y-axis scale factor

    Returns:
        Transformed image, same size as input
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    # Build affine matrix: rotate + shear + scale, centered
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    M = np.array([
        [scale_x * cos_a + shear * sin_a, -scale_x * sin_a + shear * cos_a, 0],
        [scale_y * sin_a,                  scale_y * cos_a,                  0],
    ], dtype=np.float64)

    # Adjust translation so center maps to center
    M[0, 2] = cx - M[0, 0] * cx - M[0, 1] * cy
    M[1, 2] = cy - M[1, 0] * cx - M[1, 1] * cy

    result = cv2.warpAffine(img, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    return result


# ─── Convenience: run all attacks with default parameters ────────────────────

ATTACK_SUITE = [
    ("rotate_15",    lambda img: rotate(img, 15)),
    ("rotate_45",    lambda img: rotate(img, 45)),
    ("scale_0.75",   lambda img: scale(img, 0.75)),
    ("scale_1.25",   lambda img: scale(img, 1.25)),
    ("translate_50", lambda img: translate(img, 50, 50)),
    ("crop_10pct",   lambda img: crop(img, 0.10)),
    ("crop_20pct",   lambda img: crop(img, 0.20)),
    ("affine",       lambda img: affine(img, shear=0.15, angle=12)),
]


def run_all_attacks(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Apply all attacks in ATTACK_SUITE to the image.

    Returns:
        List of (attack_name, attacked_image) tuples
    """
    return [(name, fn(img)) for name, fn in ATTACK_SUITE]
