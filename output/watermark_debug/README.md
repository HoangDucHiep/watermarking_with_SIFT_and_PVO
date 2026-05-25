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
| 1 | (192.6, 234.5) | 1.5 | 228.8 | 16/16 | 100 |
| 2 | (212.0, 293.0) | 1.6 | 63.6 | 16/16 | 100 |
| 3 | (129.7, 333.3) | 9.6 | 88.2 | 16/16 | 100 |
| 4 | (51.9, 304.3) | 3.6 | 163.0 | 16/16 | 100 |
| 5 | (168.8, 327.6) | 8.3 | 76.3 | 16/16 | 100 |
| 6 | (116.1, 380.5) | 2.7 | 89.6 | 16/16 | 100 |
| 7 | (302.0, 137.5) | 2.4 | 144.9 | 16/16 | 100 |
| 8 | (476.1, 239.5) | 3.1 | 249.0 | 16/16 | 100 |
| 9 | (339.3, 254.6) | 6.9 | 83.4 | 16/16 | 100 |
| 10 | (436.2, 310.2) | 1.5 | 20.4 | 16/16 | 100 |
| 11 | (229.6, 250.7) | 1.0 | 190.2 | 16/16 | 100 |
| 12 | (196.4, 371.6) | 1.4 | 72.4 | 16/16 | 100 |
| 13 | (272.0, 35.0) | 2.0 | 160.5 | 16/16 | 100 |
| 14 | (117.6, 294.6) | 1.5 | 61.1 | 16/16 | 100 |
| 15 | (407.1, 348.4) | 1.5 | 334.5 | 16/16 | 100 |
| 16 | (309.2, 61.1) | 0.9 | 18.8 | 16/16 | 100 |
| 17 | (480.3, 300.6) | 3.3 | 340.1 | 16/16 | 100 |
| 18 | (435.5, 268.1) | 8.0 | 99.5 | 16/16 | 100 |
| 19 | (408.6, 193.5) | 1.3 | 271.1 | 16/16 | 100 |
| 20 | (182.9, 64.1) | 1.2 | 169.1 | 16/16 | 100 |