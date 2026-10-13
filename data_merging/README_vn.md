# Gộp dữ liệu (data merging)

Pipeline này biến các file nguồn công khai thành **một bảng thuốc đã gán nhãn ADR hô hấp**:

`dird/data/cleaned_data/balanced_pulmonary_adr_dataset.csv`

Bản tiếng Anh: [README.md](README.md).

Có **hai script**, chạy theo thứ tự từ thư mục gốc `dird`:

```bash
python data_merging/01_curate_sources.py
python data_merging/02_build_dataset.py --skip-scoring
```

| Script | Việc làm |
|---|---|
| `01_curate_sources.py` | **Pha 1 — curate từng nguồn.** Lọc, lấy SMILES, chuẩn hóa cấu trúc (MEHC), ghi 4 bảng vào `data/curated_data`. |
| `02_build_dataset.py` | **Pha 2 — ghép dataset.** Gộp 4 bảng, gán nhãn, khử trùng, kiểm tra âm tính bằng PubChem, ghi bảng cuối. |

File sinh ra **luôn** nằm dưới `dird/data`, không ghi vào trong `data_merging/`.

---

## Cấu trúc thư mục

```
dird/
  data_merging/
    01_curate_sources.py
    02_build_dataset.py
    utils/                 helper dùng chung
    README.md
    README_vn.md
  data/
    raw_data/              nguồn đã tải (bạn đặt vào đây)
    intermediate_data/     checkpoint: cache API, MEHC, CSV trung gian pha 2
    curated_data/          4 bảng sau pha 1
    cleaned_data/          dataset cuối sau pha 2
```

| Thư mục | Vai trò |
|---|---|
| `data_merging/utils` | `paths.py` (đường dẫn), `mehc.py` (chuẩn hóa SMILES), `pubchem.py` (tra CID/SMILES và chấm điểm âm tính) |
| `data/raw_data` | File gốc. Thiếu file thì script báo lỗi; không tự tạo thư mục giả. |
| `data/intermediate_data/phase1_curation` | Cache CID/SMILES và thư mục MEHC theo từng nguồn |
| `data/intermediate_data/phase2_assembly` | CSV đánh số `01_` … `07_` — mỗi bước gộp một file |
| `data/curated_data` | Kết quả pha 1, đầu vào pha 2 |
| `data/cleaned_data` | Bảng cuối cho featurization / ML |

### File nguồn cần có

- `data/raw_data/adrecs/ADReCS_Respiratory_Drugs.csv` — thuốc + thuật ngữ ADR hô hấp từ ADReCS
- `data/raw_data/adrecs/pneumotox_full_database.csv` — pattern độc phổi từ Pneumotox
- `data/raw_data/sider/meddra_all_se.tsv` — tác dụng phụ SIDER (MedDRA)
- `data/raw_data/sider/drug_names.tsv` — tên thuốc SIDER (STITCH ID)
- `data/raw_data/drugcentral/drugcentral.structures.smiles.tsv` — cấu trúc DrugCentral (cột `INN`, `SMILES`)
- `data/raw_data/ctdbase/need2find.csv` — marker cơ chế từ CTD (`Drug_Name`, `Pattern`)

---

## Bốn nhãn mục tiêu

Mỗi hàng thuốc có bốn cột nhị phân:

| Cột | Ý nghĩa |
|---|---|
| `Label_ILD` | Bệnh phổi kẽ / xơ phổi / viêm phổi kẽ (ILD) |
| `Label_Edema` | Phù phổi |
| `Label_Pleural` | Tràn dịch / tổn thương màng phổi |
| `Label_Embolism` | Thuyên tắc phổi / huyết khối |

- **Dương tính:** ít nhất một nhãn = 1.
- **Ứng viên âm tính:** cả bốn nhãn = 0 — *chưa* được tin là âm tính thật.
- **Âm tính giữ lại:** ứng viên đó có điểm PubChem = 0 (không thấy bằng chứng ADR hô hấp trong mục độc tính / tác dụng phụ).

---

## Pha 1 — curate từng nguồn (`01_curate_sources.py`)

Pha 1 **không** ghép bốn nguồn thành một dataset. Nó xử lý từng nguồn riêng, để mọi thuốc đều có SMILES đã chuẩn hóa trước khi pha 2 gộp.

Chạy hết:

```bash
python data_merging/01_curate_sources.py
```

Chạy một phần (ví dụ chỉ ADReCS rồi SIDER):

```bash
python data_merging/01_curate_sources.py --source adrecs sider
```

Giá trị `--source`: `adrecs`, `sider`, `drugcentral`, `ctd`.

**DrugCentral bắt buộc ADReCS đã curate xong**, vì script loại các thuốc đã có trong ADReCS/Pneumotox.

### MEHC là gì?

MEHC (Molecular Entity Harmonization and Curation) chuẩn hóa SMILES (muối, tautomer, cách viết khác nhau) để cùng một hoạt chất từ ADReCS, SIDER, DrugCentral, CTD so sánh được với nhau, rồi khử trùng theo SMILES ở pha 2.

Kết quả MEHC và báo cáo nằm trong `data/intermediate_data/phase1_curation/<nguồn>/mehc/`.

---

### Bước 1.1 — ADReCS + Pneumotox

**Hàm:** `curate_adrecs()`  
**Ra:** `data/curated_data/adrecs_pneumotox_curated.csv`

1. Đọc ADReCS và Pneumotox. Chuẩn hóa tên thuốc (bỏ hydrate, hydrochloride, sodium, …) để khớp tên dù muối khác nhau.
2. Gom ADReCS theo tên đã chuẩn: một thuốc, nhiều `ADR_TERM` nối bằng `; `.
3. Gộp với Pneumotox:
   - có ở cả hai → `Source = Both`
   - chỉ Pneumotox → `Pneumotox_only` (CID đánh `TO_BE_FOUND`, sẽ tra PubChem)
   - chỉ ADReCS → `ADReCS_only`, `Pattern` dạng `ADReCS: <thuật ngữ>`
4. Tra CID còn thiếu, rồi lấy Canonical SMILES theo lô (có checkpoint, chạy lại không mất tiến độ).
5. Chạy MEHC, ghi bảng curated.

Checkpoint hữu ích: `cid_checkpoint.csv`, `smiles_checkpoint.csv` trong `intermediate_data/phase1_curation/adrecs/`.

---

### Bước 1.2 — SIDER

**Hàm:** `curate_sider()`  
**Ra:** `data/curated_data/sider_respiratory_adrs.csv`

1. Đọc `meddra_all_se.tsv`, giữ thuốc có MedDRA khớp một trong:
   - Pleural effusion
   - Pulmonary oedema
   - Pulmonary embolism
   - Interstitial pneumonia
2. Đổi STITCH ID (`CID1xxxxxxx`) thành CID PubChem bằng cách bỏ tiền tố `CID1`, rồi lấy SMILES. Cache: `sider_filtered_nomehc.csv`.
3. Chạy MEHC, gắn lại `STITCH_ID`, `Drug_Name`, `Respiratory_Patterns`.

---

### Bước 1.3 — DrugCentral

**Hàm:** `curate_drugcentral()`  
**Ra:** `data/curated_data/drugcentral_smiles.csv`  
**Điều kiện:** đã có `adrecs_pneumotox_curated.csv`.

1. Đọc DrugCentral (`INN`, `SMILES`).
2. Bỏ thuốc đã nằm trong tập ADReCS/Pneumotox (so tên chữ thường). Phần còn lại là **ứng viên chưa gán nhãn**.
3. Chạy MEHC (`get_report=False` vì bảng lớn).

Pha 2 sẽ **không** coi mọi hàng này là âm tính. Chúng vào nhóm “cả bốn nhãn = 0” rồi bị PubChem lọc.

---

### Bước 1.4 — CTD

**Hàm:** `curate_ctd()`  
**Ra:** `data/curated_data/ctd_mechanism_markers.csv`

1. Với mỗi `Drug_Name`, thử vài biến thể tên (tách dấu phẩy, bỏ ngoặc) rồi hỏi PubChem SMILES.
2. Chạy MEHC, join lại cột `Pattern` gốc.
3. Các hàng này là **dương tính bổ sung** ở pha 2 (`ild`, `pulmonary edema`, `pleural effusion`, `pulmonary embolism`).

---

## Pha 2 — ghép dataset (`02_build_dataset.py`)

Pha 2 đọc 4 bảng curated, ghi từng bước vào `data/intermediate_data/phase2_assembly/`, rồi ghi bảng cuối.

```bash
# Dùng lại điểm PubChem đã có (nhanh, khi cache còn đúng)
python data_merging/02_build_dataset.py --skip-scoring

# Gọi lại PubChem cho mọi ứng viên âm tính (chậm)
python data_merging/02_build_dataset.py
```

`--skip-scoring` chỉ có hiệu lực nếu đã có `07_pubchem_scored_negatives.csv`.

---

### Bước 2.1 — Gộp ADReCS/Pneumotox với SIDER

**File:** `01_merged_adrecs_sider.csv`

Nối hai bảng. SIDER đổi `Respiratory_Patterns` → `Pattern` cho cùng schema. Chưa gán nhãn.

---

### Bước 2.2 — Gán bốn nhãn từ `Pattern`

**File:** `02_labeled_by_pattern.csv`

Đọc mã kiểu Pneumotox (`I.b`, `II.a`, `V.a`, `VI.a`, …) và chữ trong `Pattern`:

| Nhãn | Mã (ví dụ) | Từ khóa chữ |
|---|---|---|
| ILD | `I.b`, `I.g`, `g` | interstitial, fibrosis, ILD |
| Edema | `II.a`, `II.d` | pulmonary oedema / edema |
| Pleural | `V.a` | pleural |
| Embolism | `VI.a`, `VI.b`, `VI.d` | embolism |

Một thuốc có thể dương tính nhiều nhãn.

---

### Bước 2.3 — Khử trùng theo `Drug_Name`

**File:** `03_deduplicated_by_drugname.csv`

Cùng tên thuốc có thể xuất hiện nhiều dòng (Pneumotox + dòng `ADReCS: …`). Giữ **một** dòng: ưu tiên bản Pneumotox hơn bản chỉ có text ADReCS.

---

### Bước 2.4 — Thêm ứng viên DrugCentral

**File:** `04_augmented_drugcentral.csv`

Thêm thuốc DrugCentral chưa có trong tập trên. Bốn nhãn = 0. Đây là **ứng viên âm tính**, chưa phải âm tính cuối.

---

### Bước 2.5 — Thêm dương tính CTD

**File:** `05_merged_ctd_positive_pool.csv`

Ánh xạ `Pattern` CTD:

| Pattern CTD | Nhãn bật |
|---|---|
| `ild` | `Label_ILD` |
| `pulmonary edema` | `Label_Edema` |
| `pleural effusion` | `Label_Pleural` |
| `pulmonary embolism` | `Label_Embolism` |

---

### Bước 2.6 — Tách ứng viên âm tính

**File:** `06_candidate_negatives_raw.csv`

Mọi hàng tổng bốn nhãn = 0. Phần còn lại của pool là dương tính, sẽ vào bảng cuối mà không cần PubChem.

---

### Bước 2.7 — Chấm điểm PubChem

**File:** `07_pubchem_scored_negatives.csv`  
**Cache:** `cache_pubchem_scores.csv`

Với mỗi ứng viên:

1. Lấy CID: ưu tiên `PubChem_ID` sẵn có, không thì tra theo tên (đã lọc dấu, rồi tên gốc).
2. Tải PUG-View (mô tả hợp chất) và tóm tắt bioassay.
3. Đếm từ khóa ADR hô hấp **chỉ** trong mục adverse / toxicity / side effect — không đếm indication (thuốc *trị* xơ phổi không bị thành dương tính giả).
4. `score = PUG_View + Assay_Summary`. `score == 0` mới được giữ là âm tính.

Gọi API có nghỉ (`SLEEP = 0.4` giây). Muốn tránh gọi lại: `--skip-scoring`.

---

### Bước 2.8 — Ghép bảng cuối

**File:** `data/cleaned_data/balanced_pulmonary_adr_dataset.csv`

1. Dương tính = hàng có bất kỳ nhãn = 1.
2. Âm tính = hàng `score == 0` từ bước 2.7.
3. Bỏ SMILES trùng (`keep="first"` — dương tính đứng trước nên được ưu tiên).
4. Cột giữ: `Drug_Name`, `Pattern`, `PubChem_ID`, `DrugBank_ID`, `KEGG_ID`, `SMILES`, bốn nhãn.

---

## Helper trong `utils/`

| File | Việc làm |
|---|---|
| `paths.py` | Mọi đường dẫn tuyệt đối dưới `dird/data`. `ROOT` = `data_merging` (thư mục cha của `utils/`). |
| `mehc.py` | `run_mehc(df, output_dir)`: đổi `SMILES` → `smiles` (MEHC yêu cầu), refine, đổi lại `SMILES`. |
| `pubchem.py` | Tra tên → CID/SMILES; chấm PUG-View + assay. Từ khóa chia theo 4 nhãn; chỉ đếm trong văn bản độc tính. |

Script pha 1/2 import kiểu: `from utils.paths import ...` (chạy `python data_merging/01_....py` từ gốc `dird`).

---

## Bước tiếp theo trong project (ngoài thư mục này)

1. **Featurization** (`featurization/feature_extraction.py`) — đọc bảng cleaned (hoặc bản refined), ghi RDKit 196 mô tả + ECFP4 2048 bit vào `data/featurized_data`.
2. **ML** (`ml_experiments/`) — chia train/test, tinh chỉnh siêu tham số, đánh giá; kết quả trong `dird/results/` (không ghi vào `data/`).
