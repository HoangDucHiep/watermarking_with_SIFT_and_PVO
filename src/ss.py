"""
Spread Spectrum (SS) watermarking — spatial domain.

Ý tưởng cơ bản
──────────────
Thay vì chỉnh ±1 pixel như IPVO, SS cộng/trừ một nhiễu giả ngẫu nhiên
(Pseudo-Noise, PN) vào toàn bộ patch.  Vì PN trải đều trên mọi pixel,
năng lượng watermark được "trải" (spread) khắp patch → robust với noise.

Embedding (mỗi patch)
─────────────────────
Với mỗi bit b_i ∈ {0,1}, sinh chuỗi PN p_i ∈ ℝ^(H×W), chuẩn hoá ||p_i||=1.
Dấu s_i = 2*b_i - 1 ∈ {-1, +1}.

    patch_wm = patch + ALPHA * Σ_i (s_i * p_i)

Extraction (mỗi patch)
──────────────────────
Tương quan patch với từng p_i:

    corr_i = <patch_wm , p_i>
           ≈ ALPHA * s_i  +  <patch_orig, p_i>  +  noise_i

Vì p_i ngẫu nhiên chuẩn hoá, <patch_orig, p_i> ≈ 0 (noise nhỏ ~std/√N).
Sau geometric attack, noise_i tăng nhưng vẫn nhỏ hơn ALPHA → sign vẫn đúng.

    b_i = 1 if corr_i > 0 else 0

Tại sao robust hơn IPVO với geometric attacks
─────────────────────────────────────────────
• IPVO signal = ±1 pixel → bị chìm khi interpolation thay đổi pixel ±3-7
• SS  signal  = ALPHA * p_i, corr trên N=1024 pixels
  → noise_attack / signal = (noise_per_pixel / sqrt(N)) / ALPHA
                           = (5 / 32) / 10 = 0.016  (rất nhỏ)
"""

import numpy as np

# ─── Tham số ──────────────────────────────────────────────────────────────────

ALPHA      = 10.0   # cường độ nhúng watermark (trade-off: PSNR vs robustness)
                    # ALPHA=10 → PSNR ≈ 46 dB với 16 bits / 32×32 patch
SS_SEED    = 99999  # seed toàn cục để tạo PN sequences (embed = extract)


# ─── Core API ─────────────────────────────────────────────────────────────────

def _pn_matrix(num_bits: int, n_pixels: int, seed: int) -> np.ndarray:
    """
    Tạo ma trận PN (num_bits × n_pixels), mỗi hàng là vector đơn vị.

    Args:
        num_bits: số watermark bits
        n_pixels: tổng số pixels của patch (= H × W)
        seed:     seed cố định để embed và extract dùng cùng PN

    Dùng cùng seed ở cả embed và extract để đảm bảo PN giống nhau.
    """
    rng = np.random.default_rng(seed)
    pn  = rng.standard_normal((num_bits, n_pixels))
    norms = np.linalg.norm(pn, axis=1, keepdims=True)
    pn /= (norms + 1e-12)
    return pn   # shape: (num_bits, n_pixels)


def embed(patch: np.ndarray, bits: np.ndarray,
          seed: int = SS_SEED, alpha: float = ALPHA) -> np.ndarray:
    """
    Nhúng watermark bits vào patch bằng Spread Spectrum.

    Args:
        patch:  2-D uint8 array (H × W)
        bits:   1-D array nhị phân (0/1), độ dài num_bits
        seed:   seed để tạo PN sequences (phải giống lúc extract)
        alpha:  cường độ nhúng

    Returns:
        patch_wm: uint8 array cùng kích thước, đã nhúng watermark
    """
    h, w = patch.shape
    n    = len(bits)

    pn     = _pn_matrix(n, h * w, seed)          # (n, h*w)
    signs  = 2.0 * bits.astype(np.float64) - 1.0  # {-1, +1}
    delta  = (signs @ pn).reshape(h, w)            # tổng đóng góp của mọi bit

    patch_f = patch.astype(np.float64) + alpha * delta
    return np.clip(patch_f, 0, 255).astype(np.uint8)


def extract(patch: np.ndarray, num_bits: int,
            seed: int = SS_SEED) -> np.ndarray:
    """
    Giải mã watermark bits từ patch bằng tương quan (correlation).

    Args:
        patch:    2-D uint8 array (H × W), có thể đã qua geometric attack
        num_bits: số bits cần giải mã
        seed:     seed để tạo PN sequences (phải giống lúc embed)

    Returns:
        bits: 1-D uint8 array độ dài num_bits  (0 hoặc 1 mỗi phần tử)
    """
    h, w = patch.shape
    pn   = _pn_matrix(num_bits, h * w, seed)   # (num_bits, h*w)
    flat = patch.astype(np.float64).ravel()    # (h*w,)

    corr = pn @ flat                           # (num_bits,) — tương quan mỗi bit
    return (corr > 0).astype(np.uint8)


def psnr_estimate(num_bits: int, patch_h: int, patch_w: int,
                  alpha: float = ALPHA) -> float:
    """
    Ước tính lý thuyết PSNR (dB) của việc nhúng num_bits vào patch.

    MSE ≈ alpha² × num_bits / (patch_h × patch_w)
    PSNR = 10 × log10(255² / MSE)
    """
    mse  = alpha ** 2 * num_bits / (patch_h * patch_w)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)
