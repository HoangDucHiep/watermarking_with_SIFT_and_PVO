# Watermark Debug Artifacts

This folder contains intermediate visualisations of the watermark embedding.

## wm1_diff_full.png
Full-image difference: |watermarked − original|, colormap JET.
Brighter = larger pixel change. Shows WHERE in the image the watermark
was written (i.e. the canonical patches around SIFT keypoints).

## wm2_patch_grid.png
Top 6 keypoint patches, 4 columns:
  - Column 1: original canonical patch
  - Column 2: watermarked canonical patch
  - Column 3: pixel difference |after − before| (hot colormap)
  - Column 4: bit heatmap: blue=bit0, red=bit1, gray=shift/skip

## wm3_loc_map.png
Per-patch location map. Each 3×3 block = 1 cell. Colour encodes what
IPVO did: skip / shift / embedded bit=0 / embedded bit=1.

## wm4_patch_comparison.png
Direct side-by-side original (top row) vs watermarked (bottom row) patches.

## Per-keypoint stats
- Total keypoints: 20
- Watermark length: 16 bits
- Watermark bits: [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

| # | Position (x,y) | σ | θ (°) | Bits embedded | Loc map len |
|---|-----------------|-------|--------|---------------|-------------|
| 1 | (313.3, 137.2) | 2.6 | 166.5 | 16/16 | 100 |
| 2 | (127.6, 331.6) | 9.6 | 83.3 | 16/16 | 100 |
| 3 | (115.8, 380.3) | 2.7 | 82.4 | 16/16 | 100 |
| 4 | (167.2, 325.7) | 8.3 | 71.2 | 16/16 | 100 |
| 5 | (475.8, 239.3) | 3.1 | 242.0 | 16/16 | 100 |
| 6 | (209.0, 322.2) | 3.0 | 180.9 | 16/16 | 100 |
| 7 | (221.6, 280.9) | 12.5 | 344.0 | 16/16 | 100 |
| 8 | (338.5, 253.8) | 6.9 | 83.9 | 16/16 | 100 |
| 9 | (271.8, 34.9) | 1.9 | 157.2 | 16/16 | 100 |
| 10 | (480.0, 300.4) | 3.3 | 333.6 | 16/16 | 100 |
| 11 | (427.5, 318.1) | 3.5 | 25.6 | 16/16 | 100 |
| 12 | (316.2, 76.2) | 2.0 | 191.4 | 16/16 | 100 |
| 13 | (79.3, 293.7) | 1.9 | 40.0 | 16/16 | 100 |
| 14 | (434.2, 266.3) | 7.9 | 92.7 | 16/16 | 100 |
| 15 | (206.0, 382.3) | 2.8 | 8.2 | 16/16 | 100 |
| 16 | (40.5, 309.4) | 4.8 | 165.7 | 16/16 | 100 |
| 17 | (275.1, 205.6) | 8.6 | 181.2 | 16/16 | 100 |
| 18 | (143.3, 230.0) | 6.7 | 316.5 | 16/16 | 100 |
| 19 | (246.8, 245.9) | 3.7 | 60.8 | 16/16 | 100 |
| 20 | (356.4, 143.5) | 2.0 | 105.0 | 16/16 | 100 |