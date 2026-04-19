# Pipeline: Watermarking với SIFT + IPVO

> **Tóm tắt**: Nhúng một chuỗi bit (watermark) vào ảnh xám theo cách **reversible** (có thể khôi phục ảnh gốc).
> Dùng **SIFT** để chọn vị trí nhúng ổn định, **IPVO** để nhúng/trích bit với thay đổi tối thiểu (±1 pixel).

---

## 0. Kiến trúc tổng thể

```
╔══════════════════════════════════════════════════════════════════════╗
║                          BÊN NHÚNG                                   ║
║                                                                      ║
║  Ảnh gốc (512×512)                                                   ║
║       │                                                              ║
║       ▼                                                              ║
║  [1] SIFT detect keypoints  ──► N keypoint (x,y,σ,θ)               ║
║       │                                                              ║
║       │   với mỗi keypoint:                                          ║
║       ▼                                                              ║
║  [2] Cắt canonical patch (32×32) ◄── warp theo (x,y,σ,θ)           ║
║       │                                                              ║
║       ▼                                                              ║
║  [3] IPVO embed watermark bits ──► modified patch (32×32)           ║
║       │                          └─► location_map (100 cặp)        ║
║       ▼                                                              ║
║  [4] Ghi patch trở lại ảnh ──► Ảnh watermarked                     ║
║       │                                                              ║
║  Lưu: location_maps, keypoints, descriptors (sidecar file)          ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                          BÊN TRÍCH                                   ║
║                                                                      ║
║  Ảnh (có thể bị tấn công)  +  sidecar file                         ║
║       │                                                              ║
║       ▼                                                              ║
║  [5] SIFT detect keypoints mới + match descriptor                   ║
║       │                                                              ║
║       │   với mỗi keypoint matched:                                  ║
║       ▼                                                              ║
║  [6] Cắt canonical patch (32×32) theo keypoint MỚI                 ║
║       │                                                              ║
║       ▼                                                              ║
║  [7] IPVO extract bits (dùng location_map gốc)                      ║
║       │                                                              ║
║       ▼                                                              ║
║  [8] Majority voting → watermark khôi phục                          ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 1. SIFT — Detect Keypoints

### 1.1 Keypoint là gì?

Một **keypoint** là một điểm đặc trưng trên ảnh, được biểu diễn bằng 4 thông số:

| Thông số | Ký hiệu | Ý nghĩa |
|---|---|---|
| Vị trí | `(x, y)` | Tọa độ pixel trong ảnh |
| Scale | `σ` | Kích thước vùng ảnh xung quanh |
| Hướng | `θ` | Góc chủ đạo của gradient (0°–360°) |
| Độ mạnh | `response` | Giá trị DoG tại điểm đó |

> **Tại sao cần θ và σ?**
> Khi ảnh bị xoay hoặc phóng to/thu nhỏ, `θ` và `σ` thay đổi tương ứng.
> Nhờ đó canonical patch luôn trích được vùng ảnh **giống nhau**, bất kể phép biến đổi.

---

### 1.2 Pipeline detect SIFT (tự cài đặt)

#### Bước 1 — Xây dựng Scale-Space (Gaussian Pyramid + DoG)

```
Ảnh gốc (I)
    │
    │  blur đến σ₀ = 1.6
    ▼
┌─────────────────────────────────────────────────────────┐
│ OCTAVE 0  (kích thước gốc)                              │
│                                                         │
│  G[0][0]──blur──G[0][1]──blur──G[0][2]──blur──G[0][3] │
│     │              │              │              │      │
│  DoG[0][0]     DoG[0][1]     DoG[0][2]     DoG[0][3]  │
│  = G[1]-G[0]   = G[2]-G[1]  = G[3]-G[2]  = G[4]-G[3] │
└─────────────────────────────────────────────────────────┘
    │
    │  downsample 2× (lấy G[0][S])
    ▼
┌─────────────────────────────────────────────────────────┐
│ OCTAVE 1  (kích thước / 2)                              │
│  ... tương tự ...                                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼  (tiếp tục đến OCTAVE 3)
```

- **Gaussian image**: ảnh làm mờ với sigma = σ₀ × k^s
- **DoG** (Difference of Gaussians) = xấp xỉ Laplacian of Gaussian — phát hiện blob/corner
- **Octave**: mỗi octave là 1 mức resolution; tổng `S+3 = 6` ảnh Gaussian, `S+2 = 5` DoG

#### Bước 2 — Tìm cực trị 3D (x, y, scale)

```
Lớp DoG s-1:   [ ][ ][ ][ ][ ]
                [ ][ ][ ][ ][ ]
                        
Lớp DoG s  :   [ ][ ][★][ ][ ]   ← pixel tại (y, x) đang xét
                [ ][×][×][×][ ]
                [ ][×][·][×][ ]   · = pixel trung tâm (y, x)
                [ ][×][×][×][ ]   × = 8 pixel lân cận cùng tầng
                [ ][ ][ ][ ][ ]   ★ = 9 pixel trên + 9 pixel dưới
                        
Lớp DoG s+1:   [ ][ ][ ][ ][ ]
```

Pixel tại `(y, x, s)` là cực trị **nếu nó là min hoặc max trong tất cả 26 láng giềng**.

#### Bước 3 — Tinh chỉnh sub-pixel (Taylor expansion)

Vị trí cực trị thực không nhất thiết nằm đúng trên pixel nguyên. Giải:

$$\delta = -H^{-1} \nabla D$$

với H là Hessian 3×3 của DoG tại (s, y, x). Lặp tối đa 5 lần.

Sau đó **lọc bỏ** keypoint:
- **Contrast filter**: `|D(x+δ)| < threshold/S` → loại (điểm quá mờ)
- **Edge filter**: `Tr(H_xy)²/Det(H_xy) ≥ (r+1)²/r` → loại (nằm trên cạnh, không ổn định)

#### Bước 4 — Gán hướng (Orientation Assignment)

```
Vùng bán kính 3×1.5σ xung quanh keypoint:

    ╔══════════════════╗
    ║  ↗ ↑ ↗ ↑ ↗ ↑    ║
    ║  → · ↗ · ↑ ·    ║    → tính gradient tại mỗi pixel
    ║  ↘ → ★ ← ↖ ↑    ║    ★ = tâm keypoint
    ║  ↓ · ↙ · ← ·    ║
    ║  ↓ ↙ ↓ ↙ ↓ ↙    ║
    ╚══════════════════╝
           ↓
    Histogram 36 bins (mỗi bin = 10°)
    Trọng số = magnitude × Gaussian(dist từ tâm)
           ↓
    θ = góc của bin cao nhất  →  kp.angle
    (có thể tạo thêm keypoint cho đỉnh phụ > 80% max)
```

#### Bước 5 — Lọc chọn N keypoint tốt nhất

```python
# Điều kiện chọn:
1. Sắp xếp theo response giảm dần
2. Patch phải nằm hoàn toàn trong ảnh: x ± src_r < biên  
   (src_r = σ × 6 + 2 pixel)
3. Cách nhau ít nhất MIN_KP_DIST = 38.4 px (tránh chồng patch)
```

---

## 2. Canonical Patch — Khái niệm và cách trích

### 2.1 Tại sao cần "canonical"?

**Vấn đề**: Nếu ảnh bị xoay 30°, patch cắt thẳng tại `(x, y)` sẽ chứa **nội dung khác hoàn toàn** so với trước khi xoay.

**Giải pháp**: Canonical patch là patch được cắt **theo hệ tọa độ của keypoint**:
- Luôn hướng theo `θ` (orientation) của keypoint
- Luôn scale theo `σ` (scale) của keypoint

→ Khi ảnh xoay, `θ` của keypoint thay đổi tương ứng → patch vẫn "nhìn thấy" cùng một vùng nội dung.

### 2.2 Ma trận affine M

Để cắt canonical patch, xây ma trận `M` (2×3) ánh xạ:

$$\text{patch pixel } (p_x, p_y) \xrightarrow{M} \text{image pixel } (i_x, i_y)$$

$$M = \begin{bmatrix} \cos(-\theta) \cdot s & -\sin(-\theta) \cdot s & t_x \\ \sin(-\theta) \cdot s & \cos(-\theta) \cdot s & t_y \end{bmatrix}$$

Trong đó:
- $s = \frac{PATCH\_SIZE/2}{\sigma \times 6}$ — tỉ lệ scale
- $\theta$ — hướng keypoint (rotate -θ để align)
- $t_x, t_y$ — dịch chuyển để keypoint nằm đúng tâm patch `(16, 16)`

```
Ảnh gốc (có thể đã xoay)          Canonical patch (32×32)

    ╔══════════════╗                ╔══════════════╗
    ║   ╲  ╱  ╲   ║                ║   ─────────  ║
    ║    ╲╱  ★ ╲  ║   ── M ──►    ║   ─── ★ ───  ║
    ║    ╱╲    ╱  ║                ║   ─────────  ║
    ╚══════════════╝                ╚══════════════╝
    keypoint xoay θ°               patch luôn thẳng đứng
```

### 2.3 Tại sao dùng INTER_NEAREST?

Mỗi pixel patch được lấy từ **đúng 1 pixel ảnh** (nearest neighbor, không nội suy).

→ Khi IPVO sửa pixel patch đi ±1, ta biết chính xác pixel nào trong ảnh gốc bị sửa.
→ Khi ghi ngược lại (`put_canonical_patch`), ta ghi đúng vào vị trí đó.

Nếu dùng bilinear/bicubic: pixel patch là tổng có trọng số của nhiều pixel ảnh → không thể ghi ngược chính xác.

---

## 3. Cấu trúc Patch và Block

```
PATCH (32×32 pixels)
┌────────────────────────────────┐
│ B B B│B B B│B B B│... (10 col)│
│ B B B│B B B│B B B│            │  B = pixel thuộc block
│ B B B│B B B│B B B│            │
│──────┼─────┼─────┤            │
│ B B B│B B B│B B B│            │
│ B B B│B B B│B B B│            │  10 hàng × 10 cột = 100 BLOCK
│ B B B│B B B│B B B│            │
│──────┼─────┼─────┤            │
│  ...                          │
└────────────────────────────────┘
  ↑ BLOCK (3×3 = 9 pixels)
  mỗi block nhúng tối đa 2 bits

Tổng capacity: 100 blocks × 2 bits = 200 bits/patch
```

### Một BLOCK 3×3 cụ thể:

```
Ví dụ pixel values:   Sau flatten:
┌──────────────┐
│ 120  85  200 │
│  45 110  170 │   →   [120, 85, 200, 45, 110, 170, 90, 155, 130]
│  90 155  130 │        pos: 0   1   2   3   4    5  6    7   8
└──────────────┘

Sau argsort ascending (stable):
sorted values: [45, 85, 90, 110, 120, 130, 155, 170, 200]
positions:      [3,   1,  6,   4,   0,   8,   7,   5,  2]
                                                         ↑         ↑
                                                   i_max1=5   i_max=2
                                                   (170)       (200)
                                              i_min=3    i_min1=1
                                              (45)        (85)
```

---

## 4. IPVO — Embed

### 4.1 Ý tưởng cốt lõi

Với mỗi block 3×3:
- **Đầu MAX**: tác động lên pixel **lớn nhất** (±1)
- **Đầu MIN**: tác động lên pixel **nhỏ nhất** (±1)
- Mỗi đầu nhúng **1 bit** → tối đa 2 bits/block

Thay đổi tối thiểu → PSNR cao → ảnh watermarked gần như giống hệt gốc.

### 4.2 Quy tắc nhúng IPVO (MAX end)

```
Sau sort ascending: x_{σ(1)} ≤ x_{σ(2)} ≤ ... ≤ x_{σ(9)}

Lấy 2 pixel lớn nhất:
  i_max  = σ(9)  ← vị trí của max trong flat array
  i_max1 = σ(8)  ← vị trí của 2nd max

Tính:
  u = min(i_max, i_max1)   ← vị trí nhỏ hơn trong array
  v = max(i_max, i_max1)   ← vị trí lớn hơn trong array
  d = flat[u] - flat[v]    ← CÓ THỂ ÂM!

┌──────────────────────────────────────────────────────────────┐
│  TRƯỜNG HỢP 1: flat[i_max] == 255                           │
│  → Biên trên, không thể tăng thêm                           │
│  → SKIP: flat không đổi,  loc_max = 3                       │
├──────────────────────────────────────────────────────────────┤
│  TRƯỜNG HỢP 2: d ∈ {0, 1}  → EMBED                         │
│                                                              │
│  d = 0: gap = 0 (max = 2nd_max)                             │
│         → flat[i_max] += b    (b = bit cần nhúng: 0 hoặc 1) │
│         → loc_max = 0                                        │
│                                                              │
│  d = 1: gap = 1 VÀ max ở vị trí CHỈ SỐ NHỎ HƠN             │
│         → flat[i_max] += b                                   │
│         → loc_max = 1                                        │
├──────────────────────────────────────────────────────────────┤
│  TRƯỜNG HỢP 3: d < 0 hoặc d ≥ 2  → SHIFT                   │
│                                                              │
│  Không nhúng bit, chỉ dịch để giữ tính reversible:          │
│  → flat[i_max] += 1                                          │
│  → loc_max = 2                                               │
│                                                              │
│  Lưu ý: d = -1 (gap=1 nhưng max ở chỉ số LỚN HƠN) → shift  │
│  Đây là điểm cải tiến IPVO so với PVO!                       │
└──────────────────────────────────────────────────────────────┘
```

**Tương tự cho MIN end** (nhưng ngược: `-=` thay vì `+=`):
- `d_min = flat[s] - flat[t]` với `s = min(i_min, i_min1)`, `t = max(i_min, i_min1)`
- Embed nếu `d_min ∈ {0, 1}`: `flat[i_min] -= b`, `loc_min = d_min`
- Shift nếu không: `flat[i_min] -= 1`, `loc_min = 2`

> **Quan trọng**: MIN end tính trên block SAU KHI đã xử lý MAX end (re-sort lại).

### 4.3 Ví dụ số

```
Block: [120, 85, 200, 45, 110, 170, 90, 155, 130]

Sau sort: [45, 85, 90, 110, 120, 130, 155, 170, 200]
positions:  3   1   6    4    0    8    7    5   2
                                               ↑    ↑
                                          i_max1=5  i_max=2

MAX end:
  u = min(2, 5) = 2,  v = max(2, 5) = 5
  d = flat[2] - flat[5] = 200 - 170 = 30  ←  d ≥ 2 → SHIFT
  flat[2] = 200 + 1 = 201
  loc_max = 2

Re-sort: [45, 85, 90, 110, 120, 130, 155, 170, 201]
          3   1   6    4    0    8    7    5    2
                                              ↑    ↑
                                         i_min=3  i_min1=1

MIN end:
  s = min(3, 1) = 1,  t = max(3, 1) = 3
  d = flat[1] - flat[3] = 85 - 45 = 40  ←  d ≥ 2 → SHIFT
  flat[3] = 45 - 1 = 44
  loc_min = 2

Kết quả: [120, 85, 201, 44, 110, 170, 90, 155, 130]
loc = (2, 2)  ← block này không nhúng được bit nào
```

---

## 5. Location Map — Tại sao không thể bỏ?

### 5.1 Vấn đề

Tại extraction time, bên trích **không biết watermark bits** (đó là thứ cần tìm).

Nhìn vào block sau khi nhúng, tính `d' = flat[i_max] - flat[i_max1]`, thấy `d' = 1`:

| Lịch sử thực tế | Điều cần làm |
|---|---|
| Embed với d=0, b=1 | Lấy bit **1**, restore max -= 1, đếm bit **+1** |
| Embed với d=1, b=0 | Lấy bit **0**, restore max -= 0, đếm bit **+1** |
| Shift với d=-1 | **Không có bit**, restore max -= 1, đếm bit **giữ nguyên** |

Ba trường hợp → cùng `d' = 1` → cần `loc` để phân biệt.

### 5.2 Location map là gì?

```
loc_map = [(loc_max, loc_min), (loc_max, loc_min), ...]
           block 0              block 1             ...

Tổng cộng: 100 phần tử cho 1 patch 32×32

Mỗi loc ∈ {0, 1, 2, 3}:
  0 → embed với d = 0
  1 → embed với d = 1
  2 → shift
  3 → biên (255 hoặc 0)
```

---

## 6. IPVO — Extract

### 6.1 Quy tắc trích (MAX end)

```
Tính: d' = flat[i_max] - flat[i_max1]   (luôn ≥ 0)

loc=3 → không làm gì
loc=2 → undo shift: flat[i_max] -= 1
loc=0 → b = (d' == 1)   ;  restore: flat[i_max] -= b
loc=1 → b = (d' == 2)   ;  restore: flat[i_max] -= b
```

### 6.2 Chứng minh đúng

| Lúc embed | d gốc | b | x' | d' quan sát | b trích | restore |
|---|---|---|---|---|---|---|
| loc=0 | 0 | 0 | x | 0 | `d'==1`=0 ✓ | x'-0=x ✓ |
| loc=0 | 0 | 1 | x+1 | 1 | `d'==1`=1 ✓ | x'-1=x ✓ |
| loc=1 | 1 | 0 | x | 1 | `d'==2`=0 ✓ | x'-0=x ✓ |
| loc=1 | 1 | 1 | x+1 | 2 | `d'==2`=1 ✓ | x'-1=x ✓ |
| loc=2 | ∉{0,1} | — | x+1 | — | None | x'-1=x ✓ |

MIN end: tương tự, `d' = flat[i_min1] - flat[i_min]`, restore với `+b` / `+1`.

---

## 7. Embed vào Ảnh và Ghi Ngược

### 7.1 Luồng hoàn chỉnh trong `embed_watermark`

```python
keypoints = detect_keypoints(img, N=20)
# ↑ N keypoint → watermark được nhúng N lần (redundancy)

for kp in keypoints:
    patch = extract_canonical_patch(img, kp)    # cắt patch 32×32
    modified_patch, loc_map, n = pvo.embed(patch, watermark)
    img_out = put_canonical_patch(img_out, kp, modified_patch)  # ghi ngược

embed_data = {
    'keypoints':     keypoints,
    'descriptors':   descriptors,   # 128-D SIFT descriptor mỗi kp
    'location_maps': [loc_map_kp0, loc_map_kp1, ...],
    'watermark':     watermark,
}
```

### 7.2 Ghi patch ngược (`put_canonical_patch`)

```
Với mỗi pixel patch (px, py):

  [ix]   [cos_a  -sin_a  tx] [px]
  [iy] = [sin_a   cos_a  ty] [py]
                              [1 ]

  → pixel ảnh tại (ix, iy) = pixel patch (px, py) đã modified
```

Cách này chỉ ghi đúng `32×32 = 1024` pixel ảnh, không ảnh hưởng vùng khác.

---

## 8. Trích Watermark sau Tấn công

### 8.1 Vấn đề khi ảnh bị tấn công

Sau khi xoay/crop/noise, keypoint gốc tại `(x, y, θ)` không còn ở đúng vị trí cũ.
Nếu dùng lại keypoint gốc để cắt patch → nội dung patch sai → extract sai.

### 8.2 SIFT Descriptor Matching (Lowe Ratio Test)

```
Ảnh bị tấn công
      │
      ▼
Detect keypoints mới + tính descriptor (128-D vector)
      │
      ▼
BFMatcher: so khớp descriptor_gốc vs descriptor_mới
      │
      │  Lowe ratio test:
      │  giữ match m nếu:
      │    dist(m) < 0.75 × dist(m_next_best)
      ▼
Danh sách: keypoint_mới ↔ keypoint_gốc_index
```

**Descriptor 128-D**: chia vùng 16×16 xung quanh keypoint thành 4×4 ô,
mỗi ô có histogram gradient 8 bins → 4×4×8 = 128 giá trị.
Descriptor **bất biến với xoay, scale, thay đổi độ sáng nhỏ**.

### 8.3 Majority Voting

```
Mỗi keypoint matched → extract ra 1 bản watermark (có thể nhiễu)

votes = [0, 0, 0, ..., 0]   ← tổng vote cho từng bit

for mỗi keypoint survived:
    bits = pvo.extract(patch, loc_map, wm_length)
    votes += bits                          # thêm vote

watermark_final = (votes > count / 2)     # đa số quyết
```

```
Keypoint 1: [1, 0, 1, 1, 0, 1, ...]   ← đúng
Keypoint 2: [1, 0, 1, 1, 0, 1, ...]   ← đúng  
Keypoint 3: [1, 1, 1, 0, 0, 1, ...]   ← bị nhiễu bit 1, 3
Keypoint 4: [0, 0, 1, 1, 0, 1, ...]   ← bị nhiễu bit 0
─────────────────────────────────────
votes:       [3, 1, 4, 3, 0, 4, ...]
count = 4    threshold = 2
result:      [1, 0, 1, 1, 0, 1, ...]  ✓
```

---

## 9. Tại sao SIFT + IPVO?

| Yêu cầu | Giải pháp |
|---|---|
| Nhúng vào vùng ổn định | SIFT chọn keypoint có response cao, không chồng nhau |
| Chống xoay/scale | Canonical patch theo (σ, θ) của keypoint |
| Khôi phục được keypoint sau tấn công | SIFT descriptor matching |
| Thay đổi ảnh tối thiểu | IPVO chỉ ±1 pixel → PSNR ≈ 50–55 dB |
| Có thể khôi phục ảnh gốc | IPVO reversible (với location_map) |
| Chịu được 1 số keypoint bị mất | Majority voting trên N keypoint |

---

## 10. Sơ đồ dữ liệu giữa các file

```
src/sift_utils.py
  ├── detect_keypoints()          → list[KeyPoint]
  ├── extract_canonical_patch()   → ndarray (32×32)
  └── put_canonical_patch()       → ndarray (H×W)

src/pvo.py
  ├── embed(patch, bits)          → (modified_patch, loc_map, n_emb)
  └── extract(patch, loc_map, n)  → bits array

src/watermark_embed.py
  └── embed_watermark(img, wm)
        → (watermarked_img, embed_data)
           embed_data: {keypoints, descriptors, location_maps, watermark}

src/watermark_extract.py
  └── extract_watermark(img_attacked, embed_data)
        → (watermark_final, surviving_kps, lost_indices)
```

---

## 11. Tóm tắt số liệu

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `PATCH_SIZE` | 32 | Patch 32×32 pixel |
| `BLOCK_SIZE` | 3 | Block 3×3 pixel |
| Số block/patch | 100 | `(32//3)² = 10²` |
| Capacity/patch | 200 bits | `100 × 2` |
| `SCALE_FACTOR` | 6.0 | Vùng ảnh = `6σ` pixel → tâm patch |
| `MIN_KP_DIST` | 38.4 px | Khoảng cách tối thiểu giữa 2 keypoint |
| `N keypoints` | 20 | Watermark nhúng 20 lần |
| `σ₀` | 1.6 | Sigma khởi đầu SIFT |
| `n_octaves` | 4 | Số tầng pyramid |
| `n_scales` | 3 | Intervals/octave |
