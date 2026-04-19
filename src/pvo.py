"""
IPVO (Improved Pixel Value Ordering) — Reversible data hiding.

Reference: "Reversible Data Hiding Based on Improved Pixel Value Ordering"

For a block of n pixels with sorted order x_{σ(1)} ≤ x_{σ(2)} ≤ ... ≤ x_{σ(n)},
σ(i) is the position of the i-th smallest value in the original (unsorted) array.

═══════════════════════════════════════════════════════
EMBEDDING
═══════════════════════════════════════════════════════

MAX end  — operates on the two largest values:
  u     = min(σ(n), σ(n-1))    ← smaller array-index among the top-2 positions
  v     = max(σ(n), σ(n-1))    ← larger  array-index among the top-2 positions
  d_max = x_u − x_v            ← CAN BE NEGATIVE when max is at the larger index

  boundary (x_{σ(n)} == 255):  skip              →  loc = 3
  d_max ∈ {0, 1}:              x'_{σ(n)} += b    →  loc = d_max   (embed bit b)
  otherwise:                   x'_{σ(n)} += 1    →  loc = 2       (shift)

  What d_max encodes:
    d_max = 0  →  gap = 0 (values equal); embeddable regardless of index order
    d_max = 1  →  gap = 1 AND σ(n) < σ(n-1)  (larger value at smaller index)
    d_max < 0  →  gap > 0 AND σ(n) > σ(n-1)  (larger value at larger index) → shift
    d_max ≥ 2  →  gap ≥ 2                                                    → shift

MIN end  — computed on the block AFTER the MAX step:
  s     = min(σ(1), σ(2))
  t     = max(σ(1), σ(2))
  d_min = x_s − x_t

  boundary (x_{σ(1)} == 0):    skip              →  loc = 3
  d_min ∈ {0, 1}:              x'_{σ(1)} -= b    →  loc = d_min   (embed bit b)
  otherwise:                   x'_{σ(1)} -= 1    →  loc = 2       (shift)

  Note: d_min = 1 when gap = 1 AND σ(1) > σ(2) (smaller value at larger index).

═══════════════════════════════════════════════════════
WHY THE LOCATION MAP IS UNAVOIDABLE
═══════════════════════════════════════════════════════

At extraction time we can observe d' = x'_{σ(n)} − x'_{σ(n-1)} ≥ 0, but d' = 1
is reached by two different histories:
  • original d=0, b=1 embedded  →  x' = x+1, d' = 1
  • original d=1, b=0 embedded  →  x' = x,   d' = 1
These require different restore actions, so the loc map is necessary.

═══════════════════════════════════════════════════════
EXTRACTION  (loc map drives the decoding)
═══════════════════════════════════════════════════════

After sorting the (possibly modified) block, let:
  d' = x'_{σ(n)} − x'_{σ(n-1)}    (always ≥ 0 — a plain gap, not the signed d)

MAX end:
  loc=3 → no change
  loc=2 → undo shift:  x_{σ(n)} = x'_{σ(n)} − 1
  loc=0 → b = (d' == 1);  restore: x_{σ(n)} = x'_{σ(n)} − b
  loc=1 → b = (d' == 2);  restore: x_{σ(n)} = x'_{σ(n)} − b

MIN end (re-sort after MAX restoration):
  d' = x'_{σ(2)} − x'_{σ(1)}    (always ≥ 0)

  loc=3 → no change
  loc=2 → undo shift:  x_{σ(1)} = x'_{σ(1)} + 1
  loc=0 → b = (d' == 1);  restore: x_{σ(1)} = x'_{σ(1)} + b
  loc=1 → b = (d' == 2);  restore: x_{σ(1)} = x'_{σ(1)} + b

Proof that extraction is correct:
  embed d=0, b=1  → x'=x+1 → d'=0+1=1 → b=(d'==1)=1 ✓   restore x'-1 = x ✓
  embed d=0, b=0  → x'=x   → d'=0     → b=(d'==1)=0 ✓   restore x'-0 = x ✓
  embed d=1, b=1  → x'=x+1 → d'=1+1=2 → b=(d'==2)=1 ✓   restore x'-1 = x ✓
  embed d=1, b=0  → x'=x   → d'=1+0=1 → b=(d'==2)=0 ✓   restore x'-0 = x ✓
  shift (d∉{0,1}) → x'=x+1 → undo x'-1 = x                              ✓

Capacity: up to 2 bits per 3×3 block → 200 bits per 32×32 patch.
"""

import numpy as np

BLOCK_SIZE = 3


def capacity_for_patch(patch_size: int) -> int:
    """Maximum embeddable bits in a square patch (2 bits per block)."""
    return ((patch_size // BLOCK_SIZE) ** 2) * 2


# ─────────────────────────────────────────────────────────────────────────────
# Block-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _embed_block(flat: np.ndarray, bit_max: int, bit_min: int
                 ) -> tuple[np.ndarray, int, int]:
    """
    Embed one bit at each end of a flattened block using IPVO.

    Args:
        flat:    1-D int16 array of BLOCK_SIZE² pixel values
        bit_max: watermark bit (0 or 1) for the MAX end
        bit_min: watermark bit (0 or 1) for the MIN end

    Returns:
        (modified_flat, loc_max, loc_min)
    """
    flat = flat.copy()

    # ── Sort ascending (stable keeps ties ordered by original position) ───────
    idx    = np.argsort(flat, kind='stable')
    i_max  = int(idx[-1])   # array-position of the maximum value
    i_max1 = int(idx[-2])   # array-position of the 2nd maximum

    # ── MAX end ───────────────────────────────────────────────────────────────
    u     = min(i_max, i_max1)
    v     = max(i_max, i_max1)
    d_max = int(flat[u]) - int(flat[v])   # signed; can be negative

    if flat[i_max] >= 255:
        loc_max = 3                          # boundary — cannot increase
    elif d_max == 0 or d_max == 1:
        flat[i_max] += bit_max               # embed
        loc_max = d_max                      # 0 or 1
    else:
        flat[i_max] += 1                     # shift
        loc_max = 2

    # ── Re-sort after MAX step (value at i_max may have changed) ─────────────
    idx2   = np.argsort(flat, kind='stable')
    i_min  = int(idx2[0])   # array-position of the minimum value
    i_min1 = int(idx2[1])   # array-position of the 2nd minimum

    # ── MIN end ───────────────────────────────────────────────────────────────
    s     = min(i_min, i_min1)
    t     = max(i_min, i_min1)
    d_min = int(flat[s]) - int(flat[t])   # signed; can be negative

    if flat[i_min] <= 0:
        loc_min = 3                          # boundary — cannot decrease
    elif d_min == 0 or d_min == 1:
        flat[i_min] -= bit_min               # embed
        loc_min = d_min                      # 0 or 1
    else:
        flat[i_min] -= 1                     # shift
        loc_min = 2

    return flat, loc_max, loc_min


def _extract_block(flat: np.ndarray, loc_max: int, loc_min: int
                   ) -> tuple[int | None, int | None, np.ndarray]:
    """
    Extract bits and restore original pixel values from a modified block.

    Args:
        flat:    1-D int16 array of the (possibly modified) block values
        loc_max: location code for the MAX end, from the embed step
        loc_min: location code for the MIN end, from the embed step

    Returns:
        (bit_max, bit_min, restored_flat)
        bit_* is None when that end carried no bit (loc ∈ {2, 3}).
    """
    flat = flat.copy()

    # ── Sort to find current max / 2nd-max ───────────────────────────────────
    idx    = np.argsort(flat, kind='stable')
    i_max  = int(idx[-1])
    i_max1 = int(idx[-2])

    # ── MAX end ───────────────────────────────────────────────────────────────
    # d' = x'_{σ(n)} − x'_{σ(n-1)}  (always ≥ 0 — plain unsigned gap)
    bit_max = None
    if loc_max == 3:
        pass                                    # nothing was done at embed
    elif loc_max == 2:
        flat[i_max] -= 1                        # undo shift
    else:
        d_prime = int(flat[i_max]) - int(flat[i_max1])
        if loc_max == 0:
            bit_max = int(d_prime == 1)         # d=0 + b=1  →  d'=1
        else:
            bit_max = int(d_prime == 2)         # d=1 + b=1  →  d'=2
        flat[i_max] -= bit_max                  # restore original max

    # ── Re-sort after MAX restoration ─────────────────────────────────────────
    idx2   = np.argsort(flat, kind='stable')
    i_min  = int(idx2[0])
    i_min1 = int(idx2[1])

    # ── MIN end ───────────────────────────────────────────────────────────────
    # d' = x'_{σ(2)} − x'_{σ(1)}  (always ≥ 0)
    bit_min = None
    if loc_min == 3:
        pass
    elif loc_min == 2:
        flat[i_min] += 1                        # undo shift
    else:
        d_prime = int(flat[i_min1]) - int(flat[i_min])
        if loc_min == 0:
            bit_min = int(d_prime == 1)         # d=0 + b=1  →  d'=1
        else:
            bit_min = int(d_prime == 2)         # d=1 + b=1  →  d'=2
        flat[i_min] += bit_min                  # restore original min

    return bit_max, bit_min, flat


# ─────────────────────────────────────────────────────────────────────────────
# Patch-level public API
# ─────────────────────────────────────────────────────────────────────────────

def embed(patch: np.ndarray, bits: np.ndarray
          ) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    """
    Embed watermark bits into a grayscale patch using IPVO.

    Bits are consumed SEQUENTIALLY — a bit is taken only when the block's
    d ∈ {0, 1} AND the pixel is not at a boundary (loc ≠ 2, 3).

    Args:
        patch: 2-D uint8 array  (patch_size × patch_size)
        bits:  1-D array of 0/1 values

    Returns:
        (modified_patch, location_map, num_embedded)
        location_map  – list of (loc_max, loc_min) for every block.
        num_embedded  – how many bits from `bits` were actually used.
    """
    patch    = patch.astype(np.int16).copy()
    h, w     = patch.shape
    loc_map: list[tuple[int, int]] = []
    bit_idx  = 0

    for row in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for col in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            flat = patch[row:row + BLOCK_SIZE,
                         col:col + BLOCK_SIZE].flatten().copy()

            # ── Sort ─────────────────────────────────────────────────────────
            idx    = np.argsort(flat, kind='stable')
            i_max  = int(idx[-1])
            i_max1 = int(idx[-2])

            # ── MAX end ───────────────────────────────────────────────────────
            u     = min(i_max, i_max1)
            v     = max(i_max, i_max1)
            d_max = int(flat[u]) - int(flat[v])

            if flat[i_max] >= 255:
                loc_max = 3
            elif d_max == 0 or d_max == 1:
                b = int(bits[bit_idx]) if bit_idx < len(bits) else 0
                flat[i_max] += b
                loc_max = d_max
                if bit_idx < len(bits):
                    bit_idx += 1
            else:
                flat[i_max] += 1
                loc_max = 2

            # ── Re-sort after MAX step ────────────────────────────────────────
            idx2   = np.argsort(flat, kind='stable')
            i_min  = int(idx2[0])
            i_min1 = int(idx2[1])

            # ── MIN end ───────────────────────────────────────────────────────
            s     = min(i_min, i_min1)
            t     = max(i_min, i_min1)
            d_min = int(flat[s]) - int(flat[t])

            if flat[i_min] <= 0:
                loc_min = 3
            elif d_min == 0 or d_min == 1:
                b = int(bits[bit_idx]) if bit_idx < len(bits) else 0
                flat[i_min] -= b
                loc_min = d_min
                if bit_idx < len(bits):
                    bit_idx += 1
            else:
                flat[i_min] -= 1
                loc_min = 2

            loc_map.append((loc_max, loc_min))
            patch[row:row + BLOCK_SIZE,
                  col:col + BLOCK_SIZE] = flat.reshape(BLOCK_SIZE, BLOCK_SIZE)

    return patch.astype(np.uint8), loc_map, bit_idx


def extract(patch: np.ndarray, location_map: list[tuple[int, int]],
            num_bits: int) -> np.ndarray:
    """
    Extract watermark bits from a patch using the location map.

    Args:
        patch:        2-D uint8 patch (may have been attacked)
        location_map: list of (loc_max, loc_min) produced during embedding
        num_bits:     number of bits to extract

    Returns:
        1-D uint8 array of extracted bits, length == num_bits.
    """
    h, w       = patch.shape
    extracted: list[int] = []
    map_idx    = 0

    for row in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for col in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            if len(extracted) >= num_bits or map_idx >= len(location_map):
                break

            flat             = patch[row:row + BLOCK_SIZE,
                                     col:col + BLOCK_SIZE].flatten().astype(np.int16)
            loc_max, loc_min = location_map[map_idx]
            map_idx         += 1

            bit_max, bit_min, _ = _extract_block(flat, loc_max, loc_min)

            if bit_max is not None and len(extracted) < num_bits:
                extracted.append(bit_max)
            if bit_min is not None and len(extracted) < num_bits:
                extracted.append(bit_min)

    while len(extracted) < num_bits:
        extracted.append(0)

    return np.array(extracted[:num_bits], dtype=np.uint8)
