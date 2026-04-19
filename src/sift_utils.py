"""
SIFT-based utilities for keypoint detection and canonical patch extraction.

A canonical patch at keypoint (x, y, σ, θ) is extracted by:
  1. Building a 2×3 affine matrix M that:
       - rotates the image by -θ around (x, y)  → aligns orientation
       - scales so that σ·SCALE_FACTOR pixels → PATCH_SIZE/2 pixels
       - translates so the keypoint lands at (PATCH_SIZE/2, PATCH_SIZE/2)
  2. Applying warpAffine(img, M, full_size, INTER_NEAREST)
  3. Cutting a PATCH_SIZE × PATCH_SIZE window centred at (PATCH_SIZE/2, PATCH_SIZE/2)

INTER_NEAREST is critical: it maps each patch pixel to EXACTLY one source
pixel, preserving the ±1 integer changes made by IPVO.

put_canonical_patch is the exact inverse: for each destination image pixel
that falls inside the patch region (determined by the same M), we overwrite
it with the corresponding modified patch value.

detect_keypoints implements the full SIFT detection pipeline from scratch:
  1. Scale-space construction  – Gaussian pyramid + Difference-of-Gaussians
  2. Discrete extrema search   – 3×3×3 neighbourhood in (x, y, scale)
  3. Sub-pixel refinement      – Taylor expansion, contrast + edge filters
  4. Orientation assignment    – gradient histogram with parabolic peak interpolation
"""

import cv2
import numpy as np
import pickle


PATCH_SIZE   = 32    # pixels per side of the canonical patch
SCALE_FACTOR = 6.0   # patch half-width covers SCALE_FACTOR * σ source pixels
MIN_KP_DIST  = PATCH_SIZE * 1.2  # minimum centre-to-centre distance (no overlap)


# ─────────────────────────────────────────────────────────────────────────────
# SIFT detection hyper-parameters  (Lowe 2004)
# ─────────────────────────────────────────────────────────────────────────────

_SIFT_N_OCTAVES       = 4      # maximum number of octaves
_SIFT_N_SCALES        = 3      # intervals per octave  (s in the paper)
_SIFT_SIGMA0          = 1.6    # base-octave target blur sigma
_SIFT_SIGMA_CAMERA    = 0.5    # assumed lens blur already in the input image
_SIFT_CONTRAST_THRESH = 0.04   # |D| threshold (image normalised to [0, 1])
_SIFT_EDGE_THRESH_R   = 10.0   # principal-curvature ratio threshold
_SIFT_ORI_BINS        = 36     # bins in the orientation histogram
_SIFT_ORI_SIGMA_FAC   = 1.5    # orientation window: σ_w = factor × scale_σ
_SIFT_ORI_PEAK_RATIO  = 0.8    # retain auxiliary peaks above this × max-peak


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_M(kp: cv2.KeyPoint, img_shape: tuple[int, int]) -> np.ndarray:
    """
    Build the 2×3 affine matrix that maps image coords → warped patch coords.

    For warpAffine(img, M, dsize):
        dst_pixel(px, py) = img[M * (px, py, 1)^T]
    So M maps (patch pixel coord) → (source image coord).

    After the warp, keypoint (x, y) appears at (PATCH_SIZE/2, PATCH_SIZE/2).
    """
    x, y  = kp.pt
    sigma = kp.size / 2.0   # half-diameter ≈ scale parameter
    angle = kp.angle         # orientation in degrees

    # scale: maps SCALE_FACTOR * σ source pixels → PATCH_SIZE/2 patch pixels
    scale = (PATCH_SIZE / 2.0) / (sigma * SCALE_FACTOR)

    rad   = np.deg2rad(-angle)   # rotate by -θ to align orientation
    cos_a = np.cos(rad) * scale
    sin_a = np.sin(rad) * scale

    # M maps patch coords (px,py) → image coords (ix,iy):
    #   [ix]   [cos_a  -sin_a  tx] [px]
    #   [iy] = [sin_a   cos_a  ty] [py]
    # Solve for tx,ty so that (PATCH_SIZE/2, PATCH_SIZE/2) → (x, y):
    #   x = cos_a*(P/2) - sin_a*(P/2) + tx  →  tx = x - (cos_a - sin_a)*(P/2)
    #   y = sin_a*(P/2) + cos_a*(P/2) + ty  →  ty = y - (sin_a + cos_a)*(P/2)
    half = PATCH_SIZE / 2.0
    tx = x - (cos_a - sin_a) * half
    ty = y - (sin_a + cos_a) * half

    M = np.array([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
    ], dtype=np.float64)
    return M


# ─────────────────────────────────────────────────────────────────────────────
# SIFT step 1 – Scale-space construction
# ─────────────────────────────────────────────────────────────────────────────

def _sift_build_scale_space(gray01: np.ndarray):
    """
    Build the Gaussian and Difference-of-Gaussians (DoG) pyramids.

    Parameters
    ----------
    gray01 : float32 ndarray normalised to [0, 1]

    Returns
    -------
    gaussians  : list[list[ndarray]]
        gaussians[o][s] – blurred image at octave o, scale index s
        Each octave contains _SIFT_N_SCALES + 3 images.
    dogs       : list[list[ndarray]]
        dogs[o][s] = gaussians[o][s+1] − gaussians[o][s]
        Each octave contains _SIFT_N_SCALES + 2 DoG images.
    factors    : list[float]
        factors[o] = 2^o  — multiply octave-frame coords to get original coords.
    sigmas_oct : list[list[float]]
        sigmas_oct[o][s] – sigma in the octave-frame coordinate system at (o, s).
    """
    S  = _SIFT_N_SCALES
    k  = 2.0 ** (1.0 / S)
    h, w = gray01.shape

    n_oct = min(_SIFT_N_OCTAVES, int(np.log2(min(h, w))) - 2)
    n_oct = max(n_oct, 1)

    # Pre-blur: bring input from assumed camera sigma to target sigma0
    sig_diff = np.sqrt(max(_SIFT_SIGMA0 ** 2 - _SIFT_SIGMA_CAMERA ** 2, 1e-8))
    base = cv2.GaussianBlur(gray01, (0, 0), sig_diff)

    gaussians, dogs, factors, sigmas_oct = [], [], [], []

    for o in range(n_oct):
        factor = float(1 << o)   # 2^o

        # Sigma at each of S+3 levels in the octave frame
        sigs = [_SIFT_SIGMA0 * (k ** s) for s in range(S + 3)]

        # Build S+3 progressively-blurred images
        imgs = [base]
        for s in range(1, S + 3):
            sig_inc = np.sqrt(max(sigs[s] ** 2 - sigs[s - 1] ** 2, 1e-8))
            imgs.append(cv2.GaussianBlur(imgs[-1], (0, 0), sig_inc))

        gaussians.append(imgs)
        dogs.append([imgs[i + 1] - imgs[i] for i in range(S + 2)])
        factors.append(factor)
        sigmas_oct.append(sigs)

        # Seed next octave: subsample the image blurred to sigma0*2  (= imgs[S])
        nxt = imgs[S]
        nh, nw = nxt.shape[0] // 2, nxt.shape[1] // 2
        if nh < 4 or nw < 4:
            break
        base = cv2.resize(nxt, (nw, nh), interpolation=cv2.INTER_LINEAR)

    return gaussians, dogs, factors, sigmas_oct


# ─────────────────────────────────────────────────────────────────────────────
# SIFT step 2 – Sub-pixel / sub-scale keypoint refinement
# ─────────────────────────────────────────────────────────────────────────────

def _sift_refine_extremum(dogs_oct: list, s0: int, y0: int, x0: int):
    """
    Refine a candidate extremum to sub-pixel / sub-scale accuracy using the
    quadratic Taylor expansion of D around (s, y, x).

    Iterates up to 5 times; if the offset converges (|δ| ≤ 0.5 in every
    dimension) the keypoint passes contrast and edge-response filters.

    Returns (s_f, y_f, x_f, response) on success, or None if rejected.
    """
    n_dogs   = len(dogs_oct)
    h, w     = dogs_oct[0].shape
    S        = _SIFT_N_SCALES
    s, y, x  = s0, y0, x0

    for _ in range(5):
        if (s < 1 or s >= n_dogs - 1 or
                y < 1 or y >= h - 1 or
                x < 1 or x >= w - 1):
            return None

        d_prev, d_curr, d_next = dogs_oct[s - 1], dogs_oct[s], dogs_oct[s + 1]

        # First-order partial derivatives (central differences)
        dx = 0.5 * (d_curr[y, x + 1]     - d_curr[y, x - 1])
        dy = 0.5 * (d_curr[y + 1, x]     - d_curr[y - 1, x])
        ds = 0.5 * (d_next[y, x]          - d_prev[y, x])
        grad = np.array([ds, dy, dx], dtype=np.float64)

        # Second-order partial derivatives → 3×3 Hessian
        dxx = d_curr[y, x + 1] - 2.0 * d_curr[y, x] + d_curr[y, x - 1]
        dyy = d_curr[y + 1, x] - 2.0 * d_curr[y, x] + d_curr[y - 1, x]
        dss = d_next[y, x]     - 2.0 * d_curr[y, x] + d_prev[y, x]
        dxy = 0.25 * (d_curr[y + 1, x + 1] - d_curr[y + 1, x - 1]
                    - d_curr[y - 1, x + 1] + d_curr[y - 1, x - 1])
        dxs = 0.25 * (d_next[y, x + 1]     - d_next[y, x - 1]
                    - d_prev[y, x + 1]      + d_prev[y, x - 1])
        dys = 0.25 * (d_next[y + 1, x]     - d_next[y - 1, x]
                    - d_prev[y + 1, x]      + d_prev[y - 1, x])

        H = np.array([[dss, dys, dxs],
                      [dys, dyy, dxy],
                      [dxs, dxy, dxx]], dtype=np.float64)

        try:
            offset = np.linalg.solve(H, -grad)   # δ = −H⁻¹ ∇D
        except np.linalg.LinAlgError:
            return None

        if np.max(np.abs(offset)) <= 0.5:
            # ── Converged: apply contrast filter ──────────────────────────
            response = float(d_curr[y, x] + 0.5 * grad.dot(offset))
            if abs(response) < _SIFT_CONTRAST_THRESH / S:
                return None

            # ── Edge filter: Tr(H_xy)² / Det(H_xy) < (r+1)²/r ───────────
            tr  = dxx + dyy
            det = dxx * dyy - dxy * dxy
            r   = _SIFT_EDGE_THRESH_R
            if det <= 0.0 or (tr * tr / det) >= (r + 1.0) ** 2 / r:
                return None

            return (float(s) + offset[0],
                    float(y) + offset[1],
                    float(x) + offset[2],
                    response)

        # Move to the neighbouring sample and retry
        s += int(round(offset[0]))
        y += int(round(offset[1]))
        x += int(round(offset[2]))

    return None   # failed to converge


# ─────────────────────────────────────────────────────────────────────────────
# SIFT step 3 – Dominant orientation assignment
# ─────────────────────────────────────────────────────────────────────────────

def _sift_assign_orientations(gaussians_oct: list,
                               candidates: list,
                               factor: float,
                               sigmas_oct: list) -> list[cv2.KeyPoint]:
    """
    Build a weighted gradient-orientation histogram for each candidate and
    return one cv2.KeyPoint per dominant orientation peak.

    All output coordinates and sizes are in *original-image* space.

    Parameters
    ----------
    gaussians_oct : list of float32 arrays (scale images for one octave)
    candidates    : list of (s_f, y_f, x_f, response) sub-pixel tuples
    factor        : 2^o – converts octave-frame coords to original-image coords
    sigmas_oct    : list of sigma values per scale (octave-frame)
    """
    S    = _SIFT_N_SCALES
    k    = 2.0 ** (1.0 / S)
    B    = _SIFT_ORI_BINS
    kps  = []
    img_h, img_w = gaussians_oct[0].shape

    for s_f, y_f, x_f, response in candidates:
        s_i = int(np.clip(round(s_f), 0, len(gaussians_oct) - 1))
        g   = gaussians_oct[s_i]

        # Sigma in the octave frame at the detected (possibly fractional) scale
        sig_oct = _SIFT_SIGMA0 * (k ** s_f)
        sig_wt  = _SIFT_ORI_SIGMA_FAC * sig_oct
        radius  = max(1, int(round(3.0 * sig_wt)))

        y_i = int(round(y_f))
        x_i = int(round(x_f))

        y0, y1 = y_i - radius, y_i + radius + 1
        x0, x1 = x_i - radius, x_i + radius + 1

        # Need a 1-pixel border for gradient computation
        if y0 < 1 or y1 > img_h - 1 or x0 < 1 or x1 > img_w - 1:
            continue

        # Build coordinate grids and compute gradients
        yg, xg = np.mgrid[y0:y1, x0:x1]
        gx  = (g[yg, xg + 1] - g[yg, xg - 1]) * 0.5
        gy  = (g[yg + 1, xg] - g[yg - 1, xg]) * 0.5
        mag = np.hypot(gx, gy)
        ori = np.arctan2(gy, gx)          # radians in (−π, π]

        # Gaussian-weighted magnitude
        w2   = 2.0 * sig_wt ** 2
        wt   = np.exp(-((yg - y_f) ** 2 + (xg - x_f) ** 2) / w2)
        wmag = (wt * mag).ravel()

        # Accumulate into circular histogram
        bin_f = (np.rad2deg(ori).ravel() % 360.0) * (B / 360.0)
        bin_i = bin_f.astype(np.int32) % B
        hist  = np.zeros(B, dtype=np.float64)
        np.add.at(hist, bin_i, wmag)

        # Smooth histogram 6× with a [¼, ½, ¼] kernel (wrap-around)
        for _ in range(6):
            hist = np.r_[hist[-1:], hist, hist[:1]]
            hist = np.convolve(hist, [0.25, 0.5, 0.25], mode='valid')

        max_val = hist.max()
        if max_val == 0.0:
            continue

        # Detect all peaks above the retention threshold
        for b in range(B):
            h_b = hist[b]
            if h_b < _SIFT_ORI_PEAK_RATIO * max_val:
                continue
            if h_b <= hist[(b - 1) % B] or h_b <= hist[(b + 1) % B]:
                continue   # not a local maximum

            # Parabolic sub-bin interpolation for smoother angle estimate
            hl    = hist[(b - 1) % B]
            hr    = hist[(b + 1) % B]
            b_hat = b + 0.5 * (hl - hr) / (hl - 2.0 * h_b + hr + 1e-12)
            angle = float(b_hat * 360.0 / B) % 360.0

            # Convert everything to original-image coordinate space
            x_orig   = float(x_f * factor)
            y_orig   = float(y_f * factor)
            sig_orig = sig_oct * factor        # σ in original-image pixels
            octave_i = int(round(np.log2(max(factor, 1.0))))

            kps.append(cv2.KeyPoint(
                x=x_orig,
                y=y_orig,
                size=float(2.0 * sig_orig),    # kp.size = 2 × σ (diameter)
                angle=angle,
                response=float(abs(response)),
                octave=octave_i,
                class_id=-1,
            ))

    return kps


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_keypoints(img: np.ndarray, n_keypoints: int = 20) -> list[cv2.KeyPoint]:
    """
    Detect SIFT keypoints from scratch and return the N most stable,
    non-overlapping ones.

    Pipeline
    --------
    1. Build Gaussian/DoG scale-space pyramid.
    2. Find discrete 3×3×3 extrema in (x, y, scale).
    3. Refine each extremum to sub-pixel accuracy (Taylor expansion);
       discard low-contrast and edge-like candidates.
    4. Assign one or more dominant orientations per keypoint.
    5. Sort by response, filter boundary / overlap conditions.

    Selection rules (unchanged from the cv2-based version):
      - Sort by response (strength) descending
      - Require the full patch region to fit inside the image
      - Require at least MIN_KP_DIST pixel gap between selected keypoints
    """
    # ── Normalise input ───────────────────────────────────────────────────────
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray01 = gray.astype(np.float32) / 255.0

    # ── Step 1: build scale-space ─────────────────────────────────────────────
    gaussians, dogs, factors, sigmas_oct = _sift_build_scale_space(gray01)

    # ── Steps 2–4: extrema → refine → orientations ───────────────────────────
    S           = _SIFT_N_SCALES
    raw_thresh  = 0.5 * _SIFT_CONTRAST_THRESH / S   # cheap pre-filter
    raw_kps: list[cv2.KeyPoint] = []

    for gaussians_oct, dogs_oct, factor, sigs in zip(
            gaussians, dogs, factors, sigmas_oct):

        h, w       = dogs_oct[0].shape
        candidates = []

        for s in range(1, S + 1):          # interior DoG scale layers
            d_prev = dogs_oct[s - 1]
            d_curr = dogs_oct[s]
            d_next = dogs_oct[s + 1]

            # Pre-filter: skip pixels that cannot pass the contrast test
            mask        = np.abs(d_curr[1:-1, 1:-1]) > raw_thresh
            ys, xs      = np.nonzero(mask)

            for yr, xr in zip(ys, xs):
                y, x = yr + 1, xr + 1
                val  = d_curr[y, x]

                # 3×3×3 extremum check (26 neighbours)
                cube = np.array([d_prev[y - 1:y + 2, x - 1:x + 2],
                                 d_curr[y - 1:y + 2, x - 1:x + 2],
                                 d_next[y - 1:y + 2, x - 1:x + 2]])
                rest = np.concatenate([cube.ravel()[:13], cube.ravel()[14:]])
                if not (val > rest.max() or val < rest.min()):
                    continue

                refined = _sift_refine_extremum(dogs_oct, s, y, x)
                if refined is not None:
                    candidates.append(refined)

        raw_kps.extend(
            _sift_assign_orientations(gaussians_oct, candidates, factor, sigs))

    # ── Step 5: select best N non-overlapping keypoints ───────────────────────
    raw_kps.sort(key=lambda kp: kp.response, reverse=True)

    h_img, w_img = gray01.shape
    selected: list[cv2.KeyPoint] = []

    for kp in raw_kps:
        x, y  = kp.pt
        sigma = kp.size / 2.0
        src_r = sigma * SCALE_FACTOR + 2   # +2 px safety margin

        if x - src_r < 0 or x + src_r >= w_img or y - src_r < 0 or y + src_r >= h_img:
            continue

        if any(np.hypot(x - s.pt[0], y - s.pt[1]) < MIN_KP_DIST
               for s in selected):
            continue

        selected.append(kp)
        if len(selected) >= n_keypoints:
            break

    return selected


def extract_canonical_patch(img: np.ndarray,
                             kp: cv2.KeyPoint) -> np.ndarray | None:
    """
    Extract a PATCH_SIZE × PATCH_SIZE canonical patch at the given keypoint.

    _build_M returns M_fwd which maps PATCH coords → IMAGE coords, i.e.
    M_fwd * (px, py, 1) = (ix, iy).

    warpAffine(src, M, dsize) convention:
        dst(x', y') = src( M_inverse * (x', y') )
    so to get dst(px, py) = img(M_fwd * (px, py)):
        we need M_inverse = M_fwd  →  pass M = M_fwd_inverse = invertAffineTransform(M_fwd)

    Uses INTER_NEAREST to preserve exact ±1 IPVO pixel values.
    """
    h, w = img.shape[:2]
    x, y = kp.pt
    sigma = kp.size / 2.0
    src_r = sigma * SCALE_FACTOR + 2

    if x - src_r < 0 or x + src_r >= w or y - src_r < 0 or y + src_r >= h:
        return None

    M_fwd = _build_M(kp, img.shape)
    M_extract = cv2.invertAffineTransform(M_fwd)  # pass to warpAffine

    # For dst pixel (px, py): sample img at M_fwd * (px, py, 1) = image coord ✓
    warped = cv2.warpAffine(img, M_extract, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REFLECT)

    patch = warped[:PATCH_SIZE, :PATCH_SIZE]
    if patch.shape != (PATCH_SIZE, PATCH_SIZE):
        return None
    return patch.copy()


def put_canonical_patch(img: np.ndarray,
                        kp: cv2.KeyPoint,
                        patch: np.ndarray) -> np.ndarray:
    """
    Write a modified canonical patch back into the image.

    For consistency with INTER_NEAREST extraction, each patch pixel (px, py)
    was read from image pixel  (ix, iy) = M_fwd * (px, py, 1)  rounded to
    the nearest integer.  We write the modified value back to the SAME pixel.

    This pixel-by-pixel approach writes to exactly PATCH_SIZE² image pixels
    (as opposed to warpAffine which overwrites scale²×PATCH_SIZE² pixels and
    damages pixels that were never extracted, hurting PSNR).
    """
    h, w    = img.shape[:2]
    M_fwd   = _build_M(kp, img.shape)
    img_out = img.copy()

    # Vectorised: build (PATCH_SIZE, PATCH_SIZE) grids of integer image coords
    py_g, px_g = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE]   # (PATCH_SIZE, PATCH_SIZE)
    ix = (M_fwd[0, 0] * px_g + M_fwd[0, 1] * py_g + M_fwd[0, 2])
    iy = (M_fwd[1, 0] * px_g + M_fwd[1, 1] * py_g + M_fwd[1, 2])

    ix_i = np.round(ix).astype(np.int32)
    iy_i = np.round(iy).astype(np.int32)

    valid = (ix_i >= 0) & (ix_i < w) & (iy_i >= 0) & (iy_i < h)
    img_out[iy_i[valid], ix_i[valid]] = patch[py_g[valid], px_g[valid]]
    return img_out


def save_keypoints(keypoints: list[cv2.KeyPoint], path: str) -> None:
    """Serialize keypoints to a pickle file."""
    data = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in keypoints]
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_keypoints(path: str) -> list[cv2.KeyPoint]:
    """Load keypoints from a pickle file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return [
        cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle,
                     response=response, octave=octave, class_id=class_id)
        for pt, size, angle, response, octave, class_id in data
    ]
