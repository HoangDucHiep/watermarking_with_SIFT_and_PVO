# Watermarking with SIFT and IPVO — Project Documentation

> **Mục tiêu:** Nhúng một chuỗi bit (watermark) vào ảnh xám theo cách **reversible** (có thể khôi phục ảnh gốc) và **robust** (chống chịu các phép tấn công hình học như xoay, phóng to, cắt...). Dùng **SIFT** để tìm vị trí nhúng ổn định, **IPVO** để nhúng bit với thay đổi pixel tối thiểu (±1).

---

## Mục lục

1. [Tổng quan Project](#1-tổng-quan-project)
2. [Luồng dữ liệu chính](#2-luồng-dữ-liệu-chính)
3. [SIFT Detector — Từ đầu đến cuối](#3-sift-detector--từ-đầu-đến-cuối)
4. [Canonical Patch — Trích patch bất biến](#4-canonical-patch--trích-patch-bất-biến)
5. [IPVO Watermarking — Nhúng bit vào block 3×3](#5-ipvo-watermarking--nhúng-bit-vào-block-3×3)
6. [Location Map — Tại sao cần thiết](#6-location-map--tại-sao-cần-thiết)
7. [Quy trình Embed đầy đủ](#7-quy-trình-embed-đầy-đủ)
8. [Quy trình Extract đầy đủ](#8-quy-trình-extract-đầy-đủ)
9. [Các phép tấn công hình học](#9-các-phép-tấn-công-hình-học)
10. [Các chỉ số đánh giá](#10-các-chỉ-số-đánh-giá)
11. [Cấu trúc file và dữ liệu](#11-cấu-trúc-file-và-dữ-liệu)
12. [So sánh SIFT+IPVO vs Baseline](#12-so-sánh-siftipvo-vs-baseline)
13. [Các tham số quan trọng](#13-các-tham-số-quan-trọng)
14. [Lệnh chạy](#14-lệnh-chạy)

---

## 1. Tổng quan Project

### 1.1 Bài toán

```
INPUT:  Ảnh xám gốc (512×512) + chuỗi bit ngẫu nhiên (16-64 bits)
OUTPUT: Ảnh đã nhúng watermark (không khác biệt nhiều so với gốc)
        + có thể trích watermark ra sau khi ảnh bị tấn công hình học
```

### 1.2 Hai thành phần cốt lõi

| Thành phần | Vai trò | File |
|---|---|---|
| **SIFT (cv2)** | Detect keypoints + compute descriptors 128-D cho matching | `cv2.SIFT_create()` trong `src/watermark_embed.py` |
| **IPVO** | Nhúng bit vào từng block 3×3 pixel, thay đổi tối thiểu | `src/pvo.py` |

> **Lưu ý:** SIFT dùng `cv2.SIFT` (OpenCV) cho cả detector và descriptor, đảm bảo **nhất quán hoàn toàn** giữa embed và extract. File `src/sift_utils.py` chứa code SIFT từ đầu cho mục đích học tập và debug (visualization).

### 1.3 Ý tưởng chính

```
Tưởng tượng ảnh là một tấm vải.

SIFT: Chọn N vị trí (keypoints) có "đặc điểm nổi bật" trên tấm vải.
      Đặc điểm = góc cạnh, điểm có độ tương phản cao.
      → Những điểm này vẫn tìm lại được dù ảnh bị xoay hay phóng to.

IPVO: Tại mỗi vị trí, lấy 1 patch 32×32 pixel.
      Chia thành 100 blocks 3×3.
      Mỗi block nhúng được 2 bits → 200 bits/patch.
      Thay đổi mỗi pixel tối đa ±1 → ảnh hầu như không thay đổi.
```

### 1.4 Tại sao cần cả hai?

| Vấn đề | Giải pháp |
|---|---|
| Ảnh bị xoay 30° | SIFT keypoint có `θ` thay đổi → canonical patch vẫn trích đúng vùng |
| Ảnh bị phóng to 0.8× | SIFT keypoint có `σ` thay đổi → patch scale tương ứng |
| Muốn khôi phục ảnh gốc | IPVO là reversible với location_map |
| Thay đổi pixel tối thiểu | IPVO chỉ thay đổi ±1 → PSNR ≈ 50 dB |
| Một số keypoint bị mất | Watermark nhúng N lần → majority voting |

---

## 2. Luồng dữ liệu chính

### 2.1 Luồng Embed (Nhúng)

```
┌─────────────────────────────────────────────────────────┐
│                     BÊN EMBED                           │
│                                                         │
│  ① Ảnh gốc (512×512)                                  │
│         │                                              │
│         ▼                                              │
│  ② SIFT detect → N keypoints (x,y,σ,θ,response)     │
│         │                                              │
│         ▼                                              │
│  ③ Với mỗi keypoint:                                  │
│         ├─ extract_canonical_patch → patch 32×32      │
│         ├─ IPVO.embed(patch, watermark)               │
│         │      → modified_patch 32×32                 │
│         │      → location_map (100 pairs)              │
│         └─ put_canonical_patch → ghi ngược vào ảnh    │
│         │                                              │
│         ▼                                              │
│  ④ Ảnh watermarked + embed_data (sidecar)            │
└─────────────────────────────────────────────────────────┘
```

**embed_data** chứa:
- `keypoints`: N keypoint đã chọn
- `descriptors`: SIFT descriptor 128-D cho mỗi keypoint
- `location_maps`: danh sách location_map cho mỗi keypoint
- `watermark`: chuỗi bit gốc
- `patch_size`: 32
- `n_embedded`: số bits thực sự nhúng cho mỗi keypoint

### 2.2 Luồng Extract (Trích)

```
┌─────────────────────────────────────────────────────────┐
│                    BÊN EXTRACT                          │
│                                                         │
│  ① Ảnh bị tấn công + embed_data (sidecar)            │
│         │                                              │
│         ▼                                              │
│  ② SIFT detect MỚI trên ảnh tấn công                  │
│         │                                              │
│         ▼                                              │
│  ③ Descriptor matching (Lowe ratio test)               │
│         → match keypoint gốc ↔ keypoint mới            │
│         │                                              │
│         ▼                                              │
│  ④ Với mỗi keypoint matched:                          │
│         ├─ extract_canonical_patch (dùng keypoint MỚI) │
│         └─ IPVO.extract(patch, loc_map, len(wm))       │
│            → bits đã trích                            │
│         │                                              │
│         ▼                                              │
│  ⑤ Majority voting từ N bản bits                      │
│         │                                              │
│         ▼                                              │
│  ⑥ Watermark cuối cùng                                │
└─────────────────────────────────────────────────────────┘
```

---

## 3. SIFT Detector — Từ đầu đến cuối

### 3.1 Keypoint là gì?

Một **keypoint** là điểm đặc trưng trong ảnh, biểu diễn bằng 4 thuộc tính:

| Thuộc tính | Ký hiệu | Ý nghĩa |
|---|---|---|
| Vị trí | `(x, y)` | Tọa độ pixel trong ảnh |
| Scale | `σ` | Bán kính vùng xung quanh (kp.size = 2σ, đường kính) |
| Hướng | `θ` | Góc chủ đạo của gradient (0°–360°) |
| Response | `response` | Giá trị DoG tại điểm đó (độ mạnh của keypoint) |

**Điểm quan trọng:** `(σ, θ)` làm cho keypoint **bất biến với scale và rotation**.

### 3.2 Tổng quan 5 bước SIFT

```
┌──────────────────────────────────────────────────────────────────┐
│                     SIFT DETECTION PIPELINE                       │
│                                                                   │
│  Bước 1: Scale-Space Construction                                 │
│    Ảnh gốc ──blur──► Gaussian pyramid ──diff──► DoG pyramid     │
│                                                                   │
│  Bước 2: Extrema Detection                                        │
│    Với mỗi pixel trong DoG: kiểm tra có phải max/min            │
│    trong 26 láng giềng 3D (x, y, scale) không?                  │
│                                                                   │
│  Bước 3: Keypoint Localization                                    │
│    Taylor expansion → sub-pixel refinement                        │
│    Lọc: contrast threshold + edge response                        │
│                                                                   │
│  Bước 4: Orientation Assignment                                   │
│    Histogram 36 bins của gradient orientation                     │
│    Đỉnh cao nhất → θ chính; đỉnh phụ → keypoint phụ            │
│                                                                   │
│  Bước 5: Keypoint Selection                                       │
│    Sort by response → lọc boundary → lọc overlap                │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 Bước 1: Scale-Space Construction

**Scale space** là tập hợp các ảnh được làm mờ với các mức sigma khác nhau, mô phỏng mắt người nhìn ở các khoảng cách khác nhau.

```
Octave 0 (kích thước gốc):
  G[0] ──blur──► G[1] ──blur──► G[2] ──blur──► G[3] ──blur──► G[4] ──blur──► G[5]
  G[0]=σ₀=1.6  G[1]=σ₀·k¹   G[2]=σ₀·k²   ...             G[5]=σ₀·k⁵
                    │              │                          │
                   DoG[0]        DoG[1]        ...          DoG[4]
                  (G[1]-G[0])  (G[2]-G[1])               (G[5]-G[4])

Octave 1 (1/2 kích thước):
  Lấy G[3] của octave 0, resize 1/2 → làm seed
  → lặp lại 6 blur levels + 5 DoG

Octave 2, 3: tương tự
```

**Tham số trong code:**
- `SIFT_SIGMA0 = 1.6` — sigma cơ sở
- `SIFT_N_SCALES = 3` — số intervals/octave
- `k = 2^(1/S) ≈ 1.26` — hệ số nhân sigma giữa các scale
- `S+3 = 6` Gaussian images/octave, `S+2 = 5` DoG images/octave
- `SIFT_N_OCTAVES = 4` — tối đa 4 octaves

**Tại sao cần DoG?**
```
DoG(x,y,σ) = G(x,y,kσ) − G(x,y,σ)
           ≈ kσ · ∇²G(x,y,σ)       (xấp xỉ Laplacian of Gaussian)
```

Laplacian of Gaussian (∇²G) có đặc tính: **đạt cực đại tại tâm của blob**. Nên DoG dùng để phát hiện blobs (vùng ảnh có độ tương phản cao).

### 3.4 Bước 2: Extrema Detection

Kiểm tra mỗi pixel trong DoG (trừ biên và 2 layer đầu/cuối) có phải là **cực trị** trong 3D neighborhood không.

```
       Layer s-1          Layer s           Layer s+1
        ┌───┬───┬───┐    ┌───┬───┬───┐    ┌───┬───┬───┐
        │   │ 9 │   │    │ 9 │ 9 │ 9 │    │   │ 9 │   │
        ├───┼───┼───┤    ├───┼───┼───┤    ├───┼───┼───┤
        │ 9 │ 9 │ 9 │    │ 9 │ ★ │ 9 │    │ 9 │ 9 │ 9 │
        ├───┼───┼───┤    ├───┼───┼───┤    ├───┼───┼───┤
        │   │ 9 │   │    │ 9 │ 9 │ 9 │    │   │ 9 │   │
        └───┴───┴───┘    └───┴───┴───┘    └───┴───┴───┘
                           ★ = pixel đang kiểm tra
                           9 = láng giềng (9 pixels mỗi layer)
                           Tổng: 26 láng giềng
```

**Quy tắc:** `★` là cực trị khi và chỉ khi:
```
★ > ALL(26 neighbors)    hoặc    ★ < ALL(26 neighbors)
```

**Pre-filter:** Trước khi kiểm tra 26 neighbors, lọc nhanh bằng contrast threshold để bỏ qua pixel yếu.

### 3.5 Bước 3: Sub-pixel Refinement (Taylor Expansion)

Vị trí cực trị thực không nằm đúng pixel nguyên. Dùng **Taylor expansion** để tinh chỉnh:

```
D(X) = D + ∂D/∂X · X + 0.5 · X^T · ∂²D/∂X² · X
```

Giải để tìm offset δ:

```
δ = −H⁻¹ · ∇D

trong đó:
  ∇D  = [∂D/∂s, ∂D/∂y, ∂D/∂x]^T   (gradient 3D)
  H   = Hessian 3×3 của D tại (s, y, x)
```

**Lặp tối đa 5 lần**, mỗi lần dịch điểm kiểm tra đến pixel lân cận rồi tính lại. Khi offset hội tụ (|δ| ≤ 0.5 ở mọi chiều), áp dụng 2 bộ lọc:

**Filter 1 — Contrast:**
```
|D(x̂)| < 0.04 / S   → loại (điểm quá mờ)
```

**Filter 2 — Edge Response:**
```
Tr(H_xy)² / Det(H_xy) ≥ (r+1)² / r   → loại (nằm trên cạnh dài)
```
Ý tưởng: keypoint tốt có cả hai curvature đều lớn (blob); keypoint trên cạnh có một curvature lớn, một nhỏ.

### 3.6 Bước 4: Orientation Assignment

Gán hướng chủ đạo cho mỗi keypoint để đạt **rotation invariance**.

```
1. Lấy vùng ảnh bán kính 3 × 1.5σ × scale_factor quanh keypoint
2. Tính gradient tại mỗi pixel:
   m(x,y) = √(L(x+1,y) − L(x-1,y))² + (L(x,y+1) − L(x,y-1))²
   θ(x,y) = arctan2(L(y+1,x) − L(y-1,x), L(x+1,y) − L(x-1,y))
3. Xây histogram 36 bins (0°–360°):
   - Mỗi bin = 10°
   - Trọng số = m(x,y) × Gaussian(dist từ keypoint)
4. Làm mượt histogram 6 lần bằng kernel [¼, ½, ¼]
5. Đỉnh cao nhất → θ chính
6. Đỉnh phụ trên 80% đỉnh cao nhất → tạo thêm keypoint phụ với góc đó
```

**Kết quả:** Một vị trí (x, y, σ) có thể tạo ra **nhiều keypoints** với các góc khác nhau.

### 3.7 Bước 5: Keypoint Selection

```
1. Sort tất cả candidate keypoints theo response giảm dần
2. Lọc bỏ keypoints có patch không nằm trong ảnh:
   x ± (σ × 6 + 2) phải trong biên
3. Lọc bỏ keypoints chồng nhau:
   khoảng cách < MIN_KP_DIST = 38.4 px → bỏ cái yếu hơn
4. Giữ lại N keypoints đầu tiên (mặc định N=20)
```

---

## 4. Canonical Patch — Trích patch bất biến

### 4.1 Vấn đề

```
Nếu ảnh bị xoay 30° và ta cắt patch tại (x,y) thẳng:
→ Patch chứa HOÀN TOÀN KHÁC nội dung so với trước khi xoay.
→ Không thể trích watermark đúng.
```

### 4.2 Giải pháp: Canonical Patch

Canonical patch được cắt theo **hệ tọa độ của keypoint**:
- **Scale:** bán kính vùng = σ × SCALE_FACTOR
- **Rotation:** xoay ngược −θ để align với hướng keypoint

```
Khi ảnh xoay 30°:
  → θ keypoint tăng 30°
  → canonical patch xoay ngược 30°
  → NỘI DUNG patch VẪN GIỐNG NHAU ✓
```

### 4.3 Ma trận Affine M

Dùng ma trận 2×3 để biến đổi tọa độ:

```
┌ ix ┐   ┌ cos_a  -sin_a   tx ┐ ┌ px ┐
│ iy │ = │ sin_a   cos_a   ty │ │ py │
└    ┘   └                   ┘ └ 1  ┘
```

```
  px, py = tọa độ trong patch (0..31)
  ix, iy = tọa độ trong ảnh gốc

  cos_a = cos(−θ) × scale
  sin_a = sin(−θ) × scale
  scale = (PATCH_SIZE / 2) / (σ × SCALE_FACTOR)
         = 16 / (σ × 6)

  tx, ty = translation để keypoint nằm đúng tâm (16, 16)
```

### 4.4 Tại sao INTER_NEAREST?

**Rất quan trọng.** Khi dùng `cv2.warpAffine(..., INTER_NEAREST)`:
- Mỗi pixel patch được lấy từ **đúng 1 pixel** ảnh
- Khi IPVO sửa pixel patch ±1, ta biết **chính xác** pixel nào trong ảnh bị sửa
- → Ghi ngược được chính xác

Nếu dùng bilinear/bicubic:
- Mỗi pixel patch = tổng có trọng số của nhiều pixel ảnh
- Khi ghi ngược, không biết sửa pixel nào → **PSNR giảm drastique**

---

## 5. IPVO Watermarking — Nhúng bit vào block 3×3

### 5.1 Cấu trúc Patch và Block

```
PATCH 32×32 pixels
├── 10 hàng × 10 cột = 100 BLOCKS
│
Mỗi BLOCK 3×3 = 9 pixels:
┌─────────────────────────────┐
│ 120  85  200 │ ← flat: [120, 85, 200, 45, 110, 170, 90, 155, 130]
│  45  110  170 │
│  90  155  130 │
└─────────────────────────────┘

argsort ascending (stable):
  sorted: [45, 85, 90, 110, 120, 130, 155, 170, 200]
  pos:    [3,  1,  6,   4,   0,   8,   7,   5,   2]
                                          ↑         ↑
                                     i_max1=5   i_max=2
                                     (170)       (200)
```

**Capacity:**
- 1 block → 2 bits (1 bit ở đầu MAX, 1 bit ở đầu MIN)
- 1 patch → 100 × 2 = 200 bits

### 5.2 Ý tưởng cốt lõi của IPVO

**Đầu MAX:** Tìm pixel lớn nhất trong block, tăng nó lên 1 nếu cần.
**Đầu MIN:** Tìm pixel nhỏ nhất, giảm nó xuống 1 nếu cần.

```
Block trước khi nhúng:
┌─────────────────┐
│ 120  85  200 │  ← max = 200, 2nd max = 170
│  45  110  170 │  ← min = 45, 2nd min = 85
│  90  155  130 │
└─────────────────┘

Sau khi nhúng bit (ví dụ: bit_max=1, bit_min=0):
┌─────────────────┐
│ 120  85  201 │  ← 200 + 1 = 201 (tăng 1)
│  44  110  170 │  ← 45 - 0 = 44 (giảm 0, không đổi!)
│  90  155  130 │
└─────────────────┘
```

**Tại sao thay đổi tối thiểu?**
- Chỉ thay đổi 1 hoặc 2 pixels trong block 9 pixels
- Mỗi pixel chỉ thay đổi ±1
- → Patch hầu như không thay đổi
- → PSNR ≈ 50 dB

### 5.3 Quy tắc nhúng chi tiết (MAX end)

**Bước 1:** Sort block ascending: `x_{σ(1)} ≤ x_{σ(2)} ≤ ... ≤ x_{σ(9)}`

**Bước 2:** Tìm 2 pixel lớn nhất:
```
i_max  = vị trí của max trong flat array (index của giá trị lớn nhất)
i_max1 = vị trí của 2nd max
u = min(i_max, i_max1)
v = max(i_max, i_max1)
d = flat[u] - flat[v]    ← GAP giữa 2 pixel lớn nhất
```

**Bước 3:** Quyết định dựa trên d:
```
TH1: flat[i_max] == 255 (đã ở max giá trị pixel)
     → SKIP: không nhúng được, loc_max = 3

TH2: d ∈ {0, 1} (gap nhỏ, cho phép nhúng)
     → flat[i_max] += b    (b = bit cần nhúng, 0 hoặc 1)
     → loc_max = d         (0 hoặc 1, ghi nhận gap gốc)

TH3: d < 0 hoặc d ≥ 2 (gap lớn, không nhúng được)
     → flat[i_max] += 1     (shift, tăng 1 để che dấu)
     → loc_max = 2
```

**Chú ý quan trọng:** `d < 0` có nghĩa là max nằm ở index lớn hơn so với 2nd max. Đây là trường hợp IPVO cải tiến so với PVO gốc — phải shift chứ không embed.

### 5.4 MIN end (tương tự nhưng sau khi re-sort)

Sau khi MAX end hoàn tất, **re-sort block** (vì giá trị đã thay đổi), rồi tìm 2 pixel nhỏ nhất:
```
i_min  = vị trí của min trong flat array
i_min1 = vị trí của 2nd min
s = min(i_min, i_min1)
t = max(i_min, i_min1)
d_min = flat[s] - flat[t]
```

Quyết định:
```
TH1: flat[i_min] == 0
     → SKIP: loc_min = 3

TH2: d_min ∈ {0, 1}
     → flat[i_min] -= b    (giảm 1 nếu b=1)
     → loc_min = d_min

TH3: d_min < 0 hoặc d_min ≥ 2
     → flat[i_min] -= 1    (shift)
     → loc_min = 2
```

### 5.5 Ví dụ số đầy đủ

```
Block gốc: [120, 85, 200, 45, 110, 170, 90, 155, 130]

Step 1: Sort
  sorted:   [45, 85, 90, 110, 120, 130, 155, 170, 200]
  positions: [3,  1,  6,   4,   0,   8,   7,   5,   2]
                                                    ↑    ↑
                                              i_max1=5  i_max=2

Step 2: MAX end
  u = min(2, 5) = 2, v = max(2, 5) = 5
  d = flat[2] - flat[5] = 200 - 170 = 30
  d ∉ {0, 1} → SHIFT
  flat[2] = 200 + 1 = 201
  loc_max = 2
  Block now: [120, 85, 201, 45, 110, 170, 90, 155, 130]

Step 3: Re-sort
  sorted:   [45, 85, 90, 110, 120, 130, 155, 170, 201]
  positions: [3,  1,  6,   4,   0,   8,   7,   5,   2]
                                     ↑    ↑
                               i_min=3  i_min1=1

Step 4: MIN end
  s = min(3, 1) = 1, t = max(3, 1) = 3
  d_min = flat[1] - flat[3] = 85 - 45 = 40
  d_min ∉ {0, 1} → SHIFT
  flat[3] = 45 - 1 = 44
  loc_min = 2

Kết quả: [120, 85, 201, 44, 110, 170, 90, 155, 130]
loc_map = (2, 2)    ← Block này không nhúng được bit nào (cả 2 đầu đều shift)
```

### 5.6 Ví dụ nhúng được bit

```
Block gốc: [120, 85, 150, 45, 110, 151, 90, 155, 130]
Watermark bits để nhúng: b_max=1, b_min=0

Step 1: Sort
  sorted:   [45, 85, 90, 110, 120, 130, 150, 151, 155]
  positions: [3,  1,  6,   4,   0,   8,   2,   5,   7]
                                              ↑    ↑
                                        i_max1=5  i_max=2
                                        (151)     (150) ← 150 là max!

Step 2: MAX end
  u = min(2, 5) = 2, v = max(2, 5) = 5
  d = flat[2] - flat[5] = 150 - 151 = −1   ← d < 0!
  → SHIFT: flat[2] = 150 + 1 = 151
  loc_max = 2

Block now: [120, 85, 151, 45, 110, 151, 90, 155, 130]

Step 3: Re-sort
  sorted:   [45, 85, 90, 110, 120, 130, 151, 151, 155]
  positions: [3,  1,  6,   4,   0,   8,   2,   5,   7]
                                     ↑    ↑
                               i_min=3  i_min1=1

Step 4: MIN end
  s = min(3, 1) = 1, t = max(3, 1) = 3
  d_min = flat[1] - flat[3] = 85 - 45 = 40
  d_min ∉ {0, 1} → SHIFT
  flat[3] = 45 - 1 = 44
  loc_min = 2

Kết quả: [120, 85, 151, 44, 110, 151, 90, 155, 130]
loc_map = (2, 2)    ← Vẫn không nhúng được!
```

**Ví dụ nhúng được:**
```
Block gốc: [120, 85, 130, 45, 110, 131, 90, 155, 130]
Watermark bits: b_max=1, b_min=0

Sorted:  [45, 85, 90, 110, 120, 130, 130, 131, 155]
pos:     [3,  1,  6,   4,   0,   2,   8,   5,   7]
                                          ↑    ↑
                                    i_max1=2  i_max=5
                                    (130)     (131)

MAX: u=min(5,2)=2, v=max(5,2)=5
     d = flat[2] - flat[5] = 130 - 131 = −1 → SHIFT
     flat[5] = 131 + 1 = 132
     loc_max = 2

Re-sort: [45, 85, 90, 110, 120, 130, 130, 132, 155]
pos:     [3,  1,  6,   4,   0,   2,   8,   5,   7]
                                     ↑    ↑
                               i_min=3  i_min1=1

MIN: s=min(3,1)=1, t=max(3,1)=3
     d_min = flat[1] - flat[3] = 85 - 45 = 40 → SHIFT
     flat[3] = 45 - 1 = 44
     loc_min = 2

Kết quả: [120, 85, 130, 44, 110, 132, 90, 155, 130]
loc_map = (2, 2)
```

**Ví dụ NHÚNG ĐƯỢC:**
```
Block gốc: [120, 85, 100, 45, 110, 100, 90, 155, 130]
Watermark bits: b_max=1, b_min=1

Sorted:  [45, 85, 90, 100, 100, 110, 120, 130, 155]
pos:     [3,  1,  6,   4,   2,   5,   0,   8,   7]
                               ↑    ↑
                         i_max1=4  i_max=2
                         (100)     (100)  ← cùng giá trị!

MAX: u=min(2,4)=2, v=max(2,4)=4
     d = flat[2] - flat[4] = 100 - 100 = 0  ← d ∈ {0, 1}!
     flat[2] = 100 + 1 = 101
     loc_max = 0       ← embed, gap gốc = 0
     (bit 1 đã nhúng)

Re-sort: [45, 85, 90, 100, 101, 110, 120, 130, 155]
pos:     [3,  1,  6,   4,   2,   5,   0,   8,   7]
                                     ↑    ↑
                               i_min=3  i_min1=1

MIN: s=min(3,1)=1, t=max(3,1)=3
     d_min = flat[1] - flat[3] = 85 - 45 = 40 → SHIFT
     flat[3] = 45 - 1 = 44
     loc_min = 2

Kết quả: [120, 85, 101, 44, 110, 100, 90, 155, 130]
loc_map = (0, 2)
Watermark: 1 bit ở MAX, 0 bit ở MIN
```

---

## 6. Location Map — Tại sao cần thiết

### 6.1 Vấn đề cốt lõi

Tại extraction time, bên trích **không biết** watermark bits (đó là thứ cần tìm). Chỉ biết giá trị pixel sau khi nhúng.

**Xét trường hợp:** `d' = flat[i_max] - flat[i_max1] = 1` sau khi nhúng:

| Lịch sử thực tế lúc embed | Giá trị gốc d | Bit nhúng | Giá trị sau nhúng | Hành động cần làm |
|---|---|---|---|---|
| Embed với d=0, b=1 | 0 | 1 | x+1 | Lấy bit **1**, restore x-1 |
| Embed với d=1, b=0 | 1 | 0 | x | Lấy bit **0**, restore x |
| Shift với d=−1 | −1 | — | x+1 | **Không bit**, restore x-1 |

Cả 3 trường hợp đều cho `d' = 1`. **Cần `loc` để phân biệt.**

### 6.2 Cấu trúc Location Map

```
location_map = [(loc_max, loc_min), (loc_max, loc_min), ...]
               block 0             block 1              ...
```

Mỗi `loc` ∈ {0, 1, 2, 3} có ý nghĩa:

| Giá trị | MAX end | MIN end |
|---|---|---|
| **0** | d gốc = 0, đã embed bit | d gốc = 0, đã embed bit |
| **1** | d gốc = 1, đã embed bit | d gốc = 1, đã embed bit |
| **2** | Shift (không embed bit) | Shift (không embed bit) |
| **3** | Skip biên (không embed) | Skip biên (không embed) |

### 6.3 Tại sao IPVO reversible?

Nhờ location_map, bên trích biết chính xác:
- Block nào đã embed bit, block nào chỉ shift
- Bit đã embed là 0 hay 1

→ Có thể khôi phục pixel gốc 100%.

---

## 7. Quy trình Embed đầy đủ

### 7.1 Mã giả

```python
def embed_watermark(img, watermark, n_keypoints=20):
    # Bước 1: Dùng cv2.SIFT detect keypoints + compute descriptors
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10)
    raw_kps, all_descs = sift.detectAndCompute(img, None)

    # Lưu original indices trước khi sort (descriptors[i] → raw_kps[i])
    for i, kp in enumerate(raw_kps):
        kp.class_id = i

    # Bước 2: Chọn top-N keypoints không chồng nhau
    raw_kps.sort(key=lambda kp: kp.response, reverse=True)
    selected = []
    for kp in raw_kps:
        if len(selected) >= n_keypoints:
            break
        if patch_fit_in_image(kp) and not_overlap(kp, selected):
            selected.append(kp)

    # Bước 3: Extract descriptors cho các keypoints đã chọn
    selected_indices = [kp.class_id for kp in selected]
    selected_descs = all_descs[selected_indices]

    # Bước 4: Với mỗi keypoint đã chọn
    for kp in selected:
        # 4a: Trích canonical patch 32×32
        patch = extract_canonical_patch(img, kp)
        if patch is None:
            location_maps.append(None)
            continue

        # 4b: Nhúng watermark bằng IPVO
        modified_patch, loc_map, n_emb = pvo.embed(patch, watermark)
        location_maps.append(loc_map)

        # 4c: Ghi patch đã sửa ngược vào ảnh
        img_out = put_canonical_patch(img_out, kp, modified_patch)

    # Bước 5: embed_data để lưu kèm ảnh
    embed_data = {
        'keypoints': selected,
        'descriptors': selected_descs,  # (N, 128), float32 từ cv2.SIFT
        'location_maps': location_maps,
        'watermark': watermark,
        'patch_size': 32,
        'n_embedded': n_embedded_list,
    }
    return img_out, embed_data
```

> **Nhất quán:** Cả `embed` và `extract` đều dùng `cv2.SIFT_create()` với cùng tham số, nên descriptors hoàn toàn tương thích.

### 7.2 put_canonical_patch — Ghi ngược chính xác

```python
# Mỗi pixel patch (px, py) tương ứng với pixel ảnh (ix, iy):
ix = M[0,0] * px + M[0,1] * py + M[0,2]
iy = M[1,0] * px + M[1,1] * py + M[1,2]

# Ghi modified_pixel vào đúng vị trí trong ảnh
img_out[iy_round][ix_round] = modified_patch[py][px]
```

Điểm mấu chốt: **chỉ ghi đúng 32×32 = 1024 pixels** vào ảnh. Không dùng warpAffine (sẽ ghi đè rất nhiều pixel trùng lắp, làm giảm PSNR).

---

## 8. Quy trình Extract đầy đủ

### 8.1 Descriptor Matching (Lowe Ratio Test)

Sau khi ảnh bị tấn công, các keypoints gốc không còn ở đúng vị trí. Cần **tìm lại** chúng.

```python
# Bước 1: Detect keypoints MỚI trên ảnh bị tấn công
sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01, edgeThreshold=20)
new_kps, new_descs = sift.detectAndCompute(img_attacked, None)

# Bước 2: So khớp descriptor bằng BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2)
raw_matches = bf.knnMatch(original_descriptors, new_descs, k=2)

# Bước 3: Lowe ratio test
for orig_idx, matches in enumerate(raw_matches):
    m, n = matches[0], matches[1]
    if m.distance < 0.75 * n.distance:  # LOWE_RATIO = 0.75
        # → Giữ match này
        matched_new_kps.append(new_kps[m.trainIdx])
```

**Lowe ratio test** đảm bảo match phải đủ tốt:
- `m.distance` = khoảng cách đến nearest neighbor
- `n.distance` = khoảng cách đến 2nd nearest neighbor
- Nếu nearest quá gần so với 2nd nearest → match đáng tin cậy

### 8.2 Không có Fallback

**Không có fallback.** Nếu một keypoint không match được thì nó đơn giản là **bị mất** — không dùng lại keypoint gốc.

Điều này đảm bảo tính nhất quán hoàn toàn: cả embed và extract đều dùng `cv2.SIFT` với cùng tham số, descriptors tương thích 100%.

### 8.3 Majority Voting

```
Mỗi keypoint matched → extract ra 1 bản watermark

votes = [0, 0, 0, ..., 0]

for mỗi keypoint survived:
    bits = pvo.extract(patch, loc_map, len(watermark))
    votes += bits              # thêm vote

watermark_final = (votes > count / 2)
```

```
Ví dụ: 4 keypoints matched

kp1: [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, ...]  ← đúng hoàn toàn
kp2: [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, ...]  ← đúng hoàn toàn
kp3: [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, ...]  ← lỗi bit 1, 3, 6, 7, 8
kp4: [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, ...]  ← lỗi bit 0

─────────────────────────────────────────────────
votes: [3, 1, 4, 3, 0, 4, 1, 2, 0, 4, ...]
count = 4,  threshold = 2

result: [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, ...]  ✓ Khôi phục đúng!
```

---

## 9. Các phép tấn công hình học

Project định nghĩa 8 phép tấn công trong `utils/attacks.py`:

### 9.1 Rotate (Xoay)

```python
def rotate(img, angle):
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
```

- Xoay ngược chiều kim đồng hồ
- Dùng `BORDER_REFLECT` để lấp đầy vùng đen
- Ảnh trả về cùng kích thước

### 9.2 Scale (Phóng to/thu nhỏ)

```python
def scale(img, factor):
    # factor > 1: zoom in → crop center
    # factor < 1: shrink → pad
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # resize về kích thước gốc
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
```

### 9.3 Translate (Dịch chuyển)

```python
def translate(img, tx, ty):
    # Circular shift: pixel ra biên sẽ quay lại từ biên kia
    result = np.roll(img, ty, axis=0)
    result = np.roll(result, tx, axis=1)
    return result
```

### 9.4 Crop (Cắt)

```python
def crop(img, ratio):
    # Cắt ratio% từ mỗi cạnh, resize về kích thước gốc
    cropped = img[dy:h-dy, dx:w-dx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
```

### 9.5 Affine (Biến dạng)

```python
def affine(img, shear=0.15, angle=12, scale_x=1.0, scale_y=1.0):
    # Kết hợp: shear + rotation + scale
    M = [[scale_x*cos_a + shear*sin_a, -scale_x*sin_a + shear*cos_a, tx],
         [scale_y*sin_a,                scale_y*cos_a,                ty]]
    return cv2.warpAffine(img, M, (w, h), ...)
```

### 9.6 Tổng hợp các phép tấn công

```python
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
```

---

## 10. Các chỉ số đánh giá

### 10.1 PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 20 × log₁₀(255 / √MSE)

MSE = mean((original - modified)²)

Ý nghĩa:
  PSNR > 50 dB → ảnh gần như không thay đổi (tốt cho watermarking)
  PSNR < 30 dB → có thể thấy sự khác biệt bằng mắt thường
```

### 10.2 SSIM (Structural Similarity Index)

```
SSIM ∈ [-1, 1], 1 = hoàn toàn giống nhau

Đo độ tương đồng về:
  - Độ sáng (luminance)
  - Độ tương phản (contrast)
  - Cấu trúc (structure)

SSIM > 0.95 → ảnh gần như không thay đổi
```

### 10.3 BER (Bit Error Rate)

```
BER = (số bits khác nhau) / (tổng số bits)

Ý nghĩa:
  BER = 0 → watermark trích ra hoàn toàn chính xác (tốt nhất)
  BER = 1 → tất cả bits đều sai
  BER càng thấp → watermark càng được bảo toàn tốt
```

### 10.4 NC (Normalized Correlation)

```
NC = (w₁·w₂) / √(||w₁||² × ||w₂||²)

w₁ = watermark gốc (0→-1, 1→+1)
w₂ = watermark trích ra (0→-1, 1→+1)

Ý nghĩa:
  NC = 1 → hoàn toàn giống nhau (tốt nhất)
  NC = 0 → không có tương quan
  NC = -1 → hoàn toàn ngược nhau
  NC > 0.7 → watermark được coi là khôi phục thành công
```

---

## 11. Cấu trúc file và dữ liệu

### 11.1 Cấu trúc project

```
project/
├── src/
│   ├── sift_utils.py         # SIFT detector tự cài (Gaussian, DoG, extrema, Taylor, orientation)
│   ├── pvo.py                # IPVO watermarking (embed, extract, location map)
│   ├── watermark_embed.py    # Embed pipeline (gọi sift + pvo)
│   ├── watermark_extract.py  # Extract pipeline (SIFT matching + majority voting)
│   └── watermark_baseline.py # Baseline: IPVO không dùng SIFT (grid keypoints)
├── utils/
│   ├── attacks.py            # 8 phép tấn công hình học
│   ├── metrics.py            # PSNR, SSIM, BER, NC
│   └── visualize.py          # Vẽ keypoints, patches, metrics table
├── data/
│   └── gray/                 # Ảnh test xám
│       └── *.tiff
├── output/                   # Kết quả chạy
│   ├── 1_keypoints.png      # Ảnh với keypoints được đánh dấu
│   ├── 2_watermarked.png    # Ảnh đã nhúng (không annotate)
│   ├── 3_embed_marked.png   # Ảnh với vùng patch được highlight
│   ├── 4_attacked_*.png     # Ảnh bị tấn công
│   ├── 5_extract_*.png      # Kết quả extract với green/red circles
│   ├── 6_metrics_table.png  # Bảng PSNR/SSIM/BER/NC
│   ├── 7_baseline_*.png     # Baseline không dùng SIFT
│   ├── 8_comparison_table.png # So sánh SIFT vs Baseline
│   ├── embed_data.pkl        # embed_data (sidecar file)
│   ├── sift_debug/          # Debug artifacts của SIFT pipeline
│   │   ├── step0_input_gray.png
│   │   ├── step1_oct0_gaussian_s0.png ...
│   │   ├── step2_oct0_raw_vs_refined.png
│   │   └── step5_final_keypoints.png
│   ├── watermark_debug/     # Debug artifacts của watermark
│   │   ├── wm1_diff_full.png
│   │   ├── wm2_patch_grid.png
│   │   ├── wm3_loc_map.png
│   │   └── wm4_patch_comparison.png
│   └── evaluation/          # Kết quả đánh giá trên toàn dataset
│       ├── results_sift.csv
│       ├── results_baseline.csv
│       ├── summary_ber_bar.png
│       ├── summary_heatmap_ber.png
│       ├── summary_heatmap_nc.png
│       └── stats.json
├── main.py                  # CLI chạy demo đầy đủ
├── evaluate.py               # Chạy đánh giá trên toàn dataset
├── requirements.txt
├── PIPELINE.md              # Pipeline diagram (file gốc)
└── sift_educational_website.html  # Website mô phỏng tương tác
```

### 11.2 embed_data structure

```python
embed_data = {
    'keypoints': [
        cv2.KeyPoint(x, y, size=2σ, angle=θ, response, octave, class_id),
        ...  # N keypoints
    ],
    'descriptors': np.ndarray,  # shape: (N, 128), float32
    'location_maps': [
        [(loc_max, loc_min), (loc_max, loc_min), ...],  # 100 pairs cho kp0
        [(loc_max, loc_min), ...],                       # 100 pairs cho kp1
        ...  # N patches
    ],
    'watermark': np.ndarray,  # shape: (wm_bits,), uint8, 0/1
    'patch_size': 32,
    'n_embedded': [n0, n1, ...],  # bits thực sự nhúng cho mỗi kp
}
```

---

## 12. So sánh SIFT+IPVO vs Baseline

### 12.1 Baseline là gì?

Baseline = IPVO không dùng SIFT. Thay vì dùng keypoints ổn định, nó dùng **grid keypoints** cố định:

```python
def grid_patch_keypoints(positions):
    kps = []
    for (r, c) in positions:
        kps.append(cv2.KeyPoint(
            x=c + PATCH_SIZE/2,
            y=r + PATCH_SIZE/2,
            size=float(PATCH_SIZE),
            angle=0,
            response=1.0,
            octave=0,
            class_id=-1,
        ))
    return kps
```

→ 20 vị trí cố định, không có `σ` (scale), không có `θ` (orientation).

### 12.2 Kết quả mong đợi

| Attack | SIFT+IPVO | Baseline IPVO | Giải thích |
|---|---|---|---|
| rotate_15 | BER thấp | BER cao | SIFT: `θ` tự động thay đổi, canonical patch vẫn đúng |
| rotate_45 | BER trung bình | BER cao | SIFT vẫn detect được nhưng có thể mất nhiều keypoints |
| scale_0.75 | BER thấp | BER cao | SIFT: `σ` tự động thay đổi |
| crop_20pct | BER trung bình | BER cao | Vùng crop có thể mất keypoints |
| affine | BER cao | BER rất cao | Affine transform khó khôi phục hoàn toàn |

### 12.3 Tại sao SIFT tốt hơn?

```
Baseline: Dùng vị trí cố định (x, y)
          → Ảnh xoay 30° → patch giờ chứa nội dung hoàn toàn khác
          → Extract sai

SIFT+IPVO: Dùng (x, y, σ, θ)
           → Ảnh xoay 30° → θ tăng 30° → canonical patch xoay ngược 30°
           → Nội dung patch VẪN GIỐNG NHAU
           → Extract đúng
```

---

## 13. Các tham số quan trọng

### 13.1 SIFT Parameters

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `SIFT_SIGMA0` | 1.6 | Sigma cơ sở của octave 0 |
| `SIFT_SIGMA_CAMERA` | 0.5 | Camera blur đã có sẵn |
| `SIFT_N_OCTAVES` | 4 | Số octaves tối đa |
| `SIFT_N_SCALES` | 3 | Intervals mỗi octave |
| `SIFT_CONTRAST_THRESH` | 0.04 | Ngưỡng lọc contrast |
| `SIFT_EDGE_THRESH_R` | 10.0 | Ngưỡng lọc edge response |
| `SIFT_ORI_BINS` | 36 | Bins trong orientation histogram |
| `SIFT_ORI_SIGMA_FAC` | 1.5 | Sigma factor cho orientation window |
| `SIFT_ORI_PEAK_RATIO` | 0.8 | Ngưỡng đỉnh phụ |

### 13.2 Patch/Block Parameters

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `PATCH_SIZE` | 32 | Kích thước patch (pixels) |
| `BLOCK_SIZE` | 3 | Kích thước block (pixels) |
| Blocks/patch | 100 | (32//3)² = 10² |
| Capacity/patch | 200 bits | 100 blocks × 2 bits |
| `SCALE_FACTOR` | 6.0 | σ × 6 = bán kính vùng trích |
| `MIN_KP_DIST` | 38.4 px | Khoảng cách tối thiểu giữa 2 keypoints |
| N keypoints mặc định | 20 | Watermark nhúng 20 lần |
| `LOWE_RATIO` | 0.75 | Ngưỡng Lowe ratio test |

---

## 14. Lệnh chạy

### 14.1 Chạy demo đầy đủ

```bash
# Demo mặc định (boat.512.tiff, 20 keypoints, 16 bits)
python main.py

# Chỉ định ảnh và tham số
python main.py --image data/gray/7.1.01.tiff --n_keypoints 15 --wm_bits 16

# Xuất debug SIFT
python main.py --dump_sift_steps

# Xuất debug watermark embedding
python main.py --dump_watermark_debug

# Cả hai cùng lúc
python main.py --dump_sift_steps --dump_watermark_debug
```

### 14.2 Chạy đánh giá trên dataset

```bash
# Tất cả ảnh, tất cả attacks
python evaluate.py

# Chỉ 5 ảnh đầu (smoke test nhanh)
python evaluate.py --subset 5

# Chỉ một số attacks
python evaluate.py --attacks rotate_15 scale_0.75

# Bỏ qua baseline (nhanh hơn)
python evaluate.py --no-baseline

# Kết hợp
python evaluate.py --subset 5 --attacks rotate_15 --no-baseline
```

### 14.3 Xem kết quả

Kết quả lưu trong `output/evaluation/`:
- `results_sift.csv` — chi tiết từng ảnh × từng attack
- `stats.json` — trung bình, std, min, max cho mỗi attack
- `summary_ber_bar.png` — biểu đồ cột BER so sánh SIFT vs Baseline
- `summary_heatmap_ber.png` — heatmap BER (ảnh × attack)
- `summary_heatmap_nc.png` — heatmap NC (ảnh × attack)
