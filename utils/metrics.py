"""
Image quality and watermark integrity metrics.

  PSNR  – Peak Signal-to-Noise Ratio          (image quality, dB)
  SSIM  – Structural Similarity Index         (image quality, 0-1)
  BER   – Bit Error Rate                      (watermark integrity, 0-1)
  NC    – Normalized Correlation              (watermark similarity, -1 to 1)
"""

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Compute PSNR between two uint8 images.

    Returns:
        PSNR in dB. Returns inf if images are identical.
    """
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Compute SSIM between two uint8 grayscale images.

    Returns:
        SSIM value in [−1, 1], higher is better (1 = identical).
    """
    return float(_ssim(original, modified, data_range=255))


def ber(watermark_orig: np.ndarray, watermark_extracted: np.ndarray) -> float:
    """
    Compute Bit Error Rate between original and extracted watermarks.

    Args:
        watermark_orig:      1D array of original bits (0/1)
        watermark_extracted: 1D array of extracted bits (0/1)

    Returns:
        BER in [0, 1]. 0 = perfect extraction, 1 = all bits wrong.
    """
    w1 = np.asarray(watermark_orig,      dtype=np.uint8)
    w2 = np.asarray(watermark_extracted, dtype=np.uint8)
    n = max(len(w1), len(w2))
    # Pad shorter array with zeros
    if len(w1) < n:
        w1 = np.pad(w1, (0, n - len(w1)))
    if len(w2) < n:
        w2 = np.pad(w2, (0, n - len(w2)))
    return float(np.sum(w1 != w2) / n)


def nc(watermark_orig: np.ndarray, watermark_extracted: np.ndarray) -> float:
    """
    Compute Normalized Correlation between original and extracted watermarks.

    Maps bits {0,1} to {-1,+1} before computing correlation.

    Returns:
        NC in [-1, 1]. 1 = perfect match.
    """
    w1 = 2 * np.asarray(watermark_orig,      dtype=np.float64) - 1
    w2 = 2 * np.asarray(watermark_extracted, dtype=np.float64) - 1

    n = min(len(w1), len(w2))
    w1, w2 = w1[:n], w2[:n]

    denom = np.sqrt(np.sum(w1 ** 2) * np.sum(w2 ** 2))
    if denom == 0:
        return 0.0
    return float(np.dot(w1, w2) / denom)


def evaluate_all(
    img_original: np.ndarray,
    img_watermarked: np.ndarray,
    wm_original: np.ndarray,
    wm_extracted: np.ndarray,
) -> dict:
    """
    Compute all four metrics at once.

    Returns:
        Dict with keys 'PSNR', 'SSIM', 'BER', 'NC'
    """
    return {
        'PSNR': psnr(img_original, img_watermarked),
        'SSIM': ssim(img_original, img_watermarked),
        'BER':  ber(wm_original, wm_extracted),
        'NC':   nc(wm_original, wm_extracted),
    }
