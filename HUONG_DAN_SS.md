# Hướng Dẫn Tính Tay: SIFT + Spread Spectrum Watermarking

> Tài liệu này giúp bạn hiểu và tính tay thuật toán, đủ để giải thích trong báo cáo và thi vấn đáp.

---

## 1. Ý Tưởng Cốt Lõi

Watermarking dấu watermark (chuỗi bit `[b₀, b₁, ..., bₙ]`) vào ảnh sao cho:
- **Vô hình** với mắt người (PSNR > 40 dB)
- **Có thể trích xuất** sau khi ảnh bị tấn công hình học (rotate, scale, crop...)

**Vấn đề:** Geometric attack thay đổi pixel → cần chuẩn hoá hình học trước khi nhúng/trích.

**Giải pháp:** SIFT tìm "vùng đặc trưng" bất biến với hình học → nhúng SS vào đó.

---

## 2. Spread Spectrum (SS) — Nguyên Lý

### 2.1 PN Sequence (Pseudo-Noise)

Với mỗi bit `i`, tạo một chuỗi ngẫu nhiên `pᵢ` có cùng kích thước với patch (32×32 = 1024 số):

```
pᵢ = randn(1024, seed=i)   # Gaussian N(0,1)
pᵢ = pᵢ / ||pᵢ||           # chuẩn hoá về độ dài 1 (unit vector)
```

**Quan trọng:** Dùng cùng `seed` ở cả nhúng và trích xuất → PN giống hệt nhau.

Tính chất của các PN sequences: **gần như trực giao** (orthogonal):
```
<pᵢ, pⱼ> ≈ 0  với i ≠ j
<pᵢ, pᵢ> = 1  (chuẩn hoá)
```

---

### 2.2 Nhúng Watermark (Embedding)

Cho watermark `w = [b₀, b₁, ..., b_{n-1}]` với bᵢ ∈ {0, 1}, đổi sang dấu:

```
sᵢ = 2·bᵢ - 1  →  sᵢ ∈ {-1, +1}
     (bit=0 → s=-1,   bit=1 → s=+1)
```

Tính tổng nhiễu:
```
δ = α · Σᵢ (sᵢ · pᵢ)        (cộng tất cả contributions)
```

Nhúng vào patch:
```
patch_wm = clip(patch + δ, 0, 255)
```

Trong code: `α = 10.0` (tham số `ALPHA`)

---

### 2.3 Trích Xuất Watermark (Extraction) — Tính Tương Quan

Cho patch (có thể đã bị tấn công), tính **tương quan** với từng pᵢ:

```
corrᵢ = <patch_received, pᵢ>  =  Σⱼ patch_received[j] · pᵢ[j]
```

Giải mã bit:
```
bᵢ = 1  nếu corrᵢ > 0
bᵢ = 0  nếu corrᵢ ≤ 0
```

---

### 2.4 Tại Sao Hoạt Động? (Phân Tích Signal vs. Noise)

Khai triển `patch_received`:
```
patch_received = patch_wm + noise_attack
              = patch_orig + α·Σⱼ(sⱼ·pⱼ) + noise_attack
```

Khi tính tương quan với `pᵢ`:
```
corrᵢ = <patch_orig, pᵢ>  +  α·Σⱼ(sⱼ·<pⱼ, pᵢ>)  +  <noise_attack, pᵢ>

      ≈ noise_img              +  α·sᵢ·1                +  noise_atk/√N

      ≈ α·sᵢ  (nếu α đủ lớn và N đủ nhiều)
```

**Chứng minh robustness:**
- Signal: `α·sᵢ = ±10`
- Noise từ ảnh gốc: `<patch_orig, pᵢ>` ~ `N(0, σ_patch/√N)` ≈ `30/32 ≈ 0.94`
- Noise từ geometric attack (~5 units/pixel): `<noise, pᵢ>` ~ `5/√1024 ≈ 0.16`
- **SNR = 10 / (0.94 + 0.16) ≈ 9×** → rất an toàn, corrᵢ hầu như luôn đúng dấu

---

## 3. Ví Dụ Tính Tay (Patch 4×4, 2 Bits)

### Dữ liệu

- Patch gốc `P` (4×4 = 16 pixels):
```
P = [[120, 130, 125, 118],
     [115, 122, 128, 132],
     [119, 126, 121, 129],
     [124, 117, 133, 127]]
```

- Watermark: `w = [1, 0]`  →  signs: `s = [+1, -1]`
- `α = 2.0` (dùng nhỏ để tính tay dễ)

---

### Bước 1: Tạo PN Sequences (giả sử đã có)

```
p₀ = [+0.49, -0.14, +0.31, -0.07,  +0.22, -0.41, +0.18, +0.09,
      -0.33, +0.27, -0.19, +0.38,  -0.06, +0.25, -0.29, +0.15]

p₁ = [-0.12, +0.35, -0.28, +0.44,  -0.09, +0.21, -0.36, +0.17,
      +0.26, -0.40, +0.13, -0.22,  +0.38, -0.11, +0.29, -0.31]
```

*(||p₀|| = ||p₁|| = 1,  <p₀, p₁> ≈ 0)*

---

### Bước 2: Tính Delta

```
δ = α · (s₀·p₀ + s₁·p₁)
  = 2 · ((+1)·p₀ + (-1)·p₁)
  = 2 · (p₀ - p₁)
```

Tính từng pixel:
```
δ[0] = 2×(0.49 - (-0.12)) = 2×0.61  = +1.22
δ[1] = 2×(-0.14 - 0.35)   = 2×(-0.49)= -0.98
δ[2] = 2×(0.31 - (-0.28)) = 2×0.59  = +1.18
δ[3] = 2×(-0.07 - 0.44)   = 2×(-0.51)= -1.02
... (tiếp tục cho 16 phần tử)
```

---

### Bước 3: Nhúng vào Patch

```
patch_wm[i] = clip(P[i] + δ[i], 0, 255)

patch_wm[0,0] = clip(120 + 1.22,  0, 255) = 121
patch_wm[0,1] = clip(130 - 0.98,  0, 255) = 129
patch_wm[0,2] = clip(125 + 1.18,  0, 255) = 126
patch_wm[0,3] = clip(118 - 1.02,  0, 255) = 117
...
```

---

### Bước 4: Trích Xuất

**Tính corr₀ = <patch_wm, p₀>:**

```
corr₀ = patch_wm[0]·p₀[0] + patch_wm[1]·p₀[1] + ...
       = 121·(+0.49) + 129·(-0.14) + 126·(+0.31) + 117·(-0.07) + ...

Ước lượng:
  ≈ <P, p₀>  +  δ·p₀        (khai triển)
  ≈ ~0        +  α·s₀·||p₀||²   (vì p₀ gần trực giao với p₁)
  = 2 × (+1) × 1.0
  = +2.0  > 0  →  b₀ = 1  ✓
```

**Tính corr₁ = <patch_wm, p₁>:**

```
  ≈ ~0  +  α·s₁·||p₁||²
  = 2 × (-1) × 1.0
  = -2.0  < 0  →  b₁ = 0  ✓
```

**Kết quả: extracted = [1, 0] = watermark gốc ✅**

---

### Bước 5: Sau Khi Rotate 15° (SIFT locate đúng patch)

```
patch_attacked ≈ patch_wm + noise  (|noise| ≈ 3-5 mỗi pixel)

corr₀_attacked = <patch_attacked, p₀>
               ≈ <patch_wm, p₀>  +  <noise, p₀>
               ≈ +2.0            +  noise_total/√16
               ≈ +2.0            +  ±0.3          (rất nhỏ)
               ≈ +1.7 đến +2.3   → luôn > 0  →  b₀ = 1  ✓
```

---

## 4. Majority Voting Qua 20 Patches

```
Patch  1: decoded = [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  ✓
Patch  2: decoded = [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  ✓
Patch  3: decoded = [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  1 bit sai
...
Patch 20: decoded = [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  ✓

votes[0] = 20  → 20/20 = 100% vote 1  →  final[0] = 1  ✓
votes[1] = 1   → 1/20  = 5%  vote 1   →  final[1] = 0  ✓
...

→ BER = 0.0  (perfect extraction)
```

---

## 5. SIFT Geometric Normalisation

### Tại Sao Cần SIFT?

```
KHÔNG có SIFT (baseline):
  rotate 15° → patch lấy từ sai vị trí → nội dung patch khác hoàn toàn
  → corr ≈ random noise → BER ≈ 0.5

CÓ SIFT:
  rotate 15° → keypoint tại vị trí mới, góc mới (θ' = θ+15°)
  → canonical patch = vùng xoay -θ' = vùng xoay -(θ+15°)
  → nội dung ≈ canonical patch gốc (xoay -θ)
  → corr vẫn đúng dấu → BER ≈ 0
```

### Ma Trận Biến Đổi Canonical Patch

Với keypoint `(x, y, θ, σ)`:

```
P = 32  (patch size)
scale = (P/2) / (σ × SCALE_FACTOR)    # co/dãn về chuẩn
rad = -θ (rad)                          # xoay ngược chiều keypoint

M = [[scale·cos(rad), -scale·sin(rad), tx],
     [scale·sin(rad),  scale·cos(rad), ty]]

tx = x - (cos(rad) - sin(rad)) × (P/2)
ty = y - (sin(rad) + cos(rad)) × (P/2)
```

M biến `patch pixel (px, py)` → `image pixel (ix, iy)`:
- Tại `(px, py) = (P/2, P/2)` → `(ix, iy) = (x, y)`  ✓ (tâm patch = keypoint)
- Scale + Rotation làm cho patch cùng kích thước bất kể σ, θ

---

## 6. Bảng So Sánh Nhanh

| | Baseline SS | SIFT + SS |
|---|---|---|
| Vị trí patch | Lưới cố định | Keypoint SIFT |
| Sau translate | ❌ BER≈0.5 | ✅ BER=0 |
| Sau rotate/scale | ❌ BER≈0.5 | ✅ BER≈0 |
| Sau crop mạnh | ❌ BER≈0.5 | ✅ BER≈0.06 |
| PSNR | ~44 dB | ~44 dB |

---

## 7. Công Thức PSNR Lý Thuyết

```
MSE  = α² × n_bits / (H × W)

PSNR = 10 × log₁₀(255² / MSE)
```

Ví dụ (α=10, 16 bits, patch 32×32):
```
MSE  = 100 × 16 / 1024 = 1.5625
PSNR = 10 × log₁₀(65025 / 1.5625) ≈ 46.2 dB
```

---

## 8. Tóm Tắt Pipeline

```
EMBED:  I_orig
            → SIFT detect 20 keypoints
            → mỗi kp: cắt canonical patch (32×32)
            → SS embed: patch += α · Σ(sᵢ · pᵢ)  [pᵢ từ seed cố định]
            → ghi patch trở lại ảnh
            → lưu: keypoints, descriptors, seed
        = I_watermarked

ATTACK: I_watermarked → rotate/scale/crop → I_attacked

EXTRACT: I_attacked
            → SIFT re-detect keypoints
            → BFMatcher match với descriptors gốc
            → mỗi keypoint khớp: cắt canonical patch
            → SS extract: corrᵢ = <patch, pᵢ>  → bᵢ = sign(corrᵢ)
            → Majority voting qua 20 patches
         = watermark recovered
```
