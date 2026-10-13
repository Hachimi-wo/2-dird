# DIRD

Pipeline xây dataset ADR hô hấp, trích xuất đặc trưng phân tử, rồi huấn luyện bộ phân loại (KNN, SVM, random forest, XGBoost).

Bản tiếng Anh: [README.md](README.md).

Mọi lệnh chạy từ thư mục này (`dird`).

```
dird/
  data_merging/       Pha A — curate nguồn và ghép bảng đã gán nhãn
  featurization/      Pha B — mô tả RDKit + vân tay ECFP4
  ml_experiments/     Pha C — chia split, tinh chỉnh, đánh giá, thống kê
  data/               Mọi dataset sinh ra (raw → cleaned → features → splits)
  results/            Chỉ artifact ML (siêu tham số, xác suất, thống kê)
```

File sinh ra luôn nằm dưới `data/` (hoặc `results/` với ML). Không ghi vào trong `data_merging/`, `featurization/`, hay `ml_experiments/` ngoài mã nguồn.

---

## Lệnh chạy xuyên suốt

Đổi `Label_ILD` thành `Label_Edema`, `Label_Pleural`, hoặc `Label_Embolism` cho các endpoint khác. Bảng cleaned có **bốn nhãn nhị phân**, không có cột `class`.

```bash
# A. Gộp dữ liệu
python data_merging/01_curate_sources.py
python data_merging/02_build_dataset.py --skip-scoring

# B. Featurization (input mặc định: data/cleaned_data/balanced_pulmonary_adr_dataset.csv)
python featurization/feature_extraction.py --features both
python featurization/feature_extraction.py --features rdkit
python featurization/feature_extraction.py --features ecfp4

# C. Học máy (ví dụ: ILD, AUROC, đặc trưng kết hợp)
python ml_experiments/01_data_splitting.py --input both --label-col Label_ILD
python ml_experiments/02_hyperparameter_tuning.py --metric auroc --label-col Label_ILD
python ml_experiments/03_model_refitting_and_evaluation.py --metric auroc --label-col Label_ILD
python ml_experiments/04_statistical_analysis.py --label-col Label_ILD
```

`--skip-scoring` dùng lại cache PubChem trong `data/intermediate_data`. Bỏ cờ này thì chấm lại thuốc chưa nhãn (chậm).

Nếu `cleaned_data` có nhiều CSV, truyền `--input` cho featurization bằng tên file đó.

---

## Pha A — Gộp dữ liệu (`data_merging/`)

Biến nguồn công khai thành một bảng đã gán nhãn:

`data/cleaned_data/balanced_pulmonary_adr_dataset.csv`

Chi tiết từng bước: [data_merging/README_vn.md](data_merging/README_vn.md).

| Script | Việc làm |
|---|---|
| `01_curate_sources.py` | Lọc ADReCS/Pneumotox, SIDER, DrugCentral, CTD; lấy SMILES; chuẩn hóa MEHC. Ghi `data/curated_data/`. |
| `02_build_dataset.py` | Gộp, gán bốn nhãn, khử trùng, chấm hàng chưa nhãn trên PubChem, ghi bảng cleaned. |

File gốc đặt trong `data/raw_data/` (ADReCS, Pneumotox, SIDER, DrugCentral, CTD). Curate DrugCentral cần ADReCS đã chạy trước.

**Nhãn**

| Cột | Ý nghĩa |
|---|---|
| `Label_ILD` | Bệnh phổi kẽ / xơ phổi |
| `Label_Edema` | Phù phổi |
| `Label_Pleural` | Tổn thương / tràn dịch màng phổi |
| `Label_Embolism` | Thuyên tắc phổi / huyết khối |

**Dương tính:** ít nhất một nhãn = 1. Hàng cả bốn nhãn = 0 là **ứng viên âm tính**; chỉ giữ làm âm tính nếu PUG-View + assay PubChem không thấy bằng chứng ADR hô hấp trong văn bản độc tính.

Helper: `data_merging/utils/` (`paths.py`, `mehc.py`, `pubchem.py`).

---

## Pha B — Featurization (`featurization/`)

Đọc CSV trong `data/cleaned_data` (mặc định `balanced_pulmonary_adr_dataset.csv`; file khác qua `--input`).

Ghi vào `data/featurized_data/`:

| `--features` | File | Nội dung |
|---|---|---|
| `rdkit` | `rdkit_md_196.csv` | 196 mô tả RDKit |
| `ecfp4` | `ecfp4_2048.csv` | ECFP4 2048 bit (`ecfp_bit0` …) |
| `both` | `both.csv` | mô tả + vân tay |

Mỗi file giữ SMILES và cột nhãn (`class` nếu có, không thì bốn cột `Label_*`).

---

## Pha C — Học máy (`ml_experiments/`)

Phân loại nhị phân **từng endpoint một**. Phải truyền `--label-col` (với dataset này: `Label_ILD`, `Label_Edema`, `Label_Pleural`, hoặc `Label_Embolism`). Các cột `Label_*` còn lại bị bỏ, không dùng làm feature. Mỗi nhãn một thư mục, bốn endpoint không ghi đè nhau.

| Script | Input | Output |
|---|---|---|
| `01_data_splitting.py` | `data/featurized_data/` (`rdkit`, `ecfp4`, hoặc `both`) | `data/split_data/{nhãn}/{random,scaffold}/seed_{k}/train.csv` và `test.csv` |
| `02_hyperparameter_tuning.py` | chỉ tập train | `results/{nhãn}/tuned_hyperparameters/gridsearch_results_{auroc\|auprc}.csv` |
| `03_model_refitting_and_evaluation.py` | split + CSV tinh chỉnh | `results/{nhãn}/predicted_probs/{metric}/{split}_{model}.csv` |
| `04_statistical_analysis.py` | xác suất dự đoán | `results/{nhãn}/statistics/computed_statistics.csv` |

Cả random và scaffold đều stratified. GridSearchCV chỉ chạy trên **fold train** (pipeline imblearn + SMOTE; scaler trong `models/utils.py`: fingerprint không scale, count/fragment RobustScaler, `FractionCSP3` MinMaxScaler). Bắt buộc `--metric`: `auroc` hoặc `auprc`.

Mô hình: `ml_experiments/models/ml_models.py` (`get_models`).

---

## Cây dữ liệu

```
data/
  raw_data/              nguồn đã tải
  intermediate_data/     checkpoint PubChem/MEHC và CSV pha 2
  curated_data/          bốn bảng sau pha 1 gộp dữ liệu
  cleaned_data/          dataset đã gán nhãn
  featurized_data/       RDKit / ECFP / both
  split_data/{nhãn}/     train/test theo kiểu split và seed

results/{nhãn}/
  tuned_hyperparameters/
  predicted_probs/
  statistics/
```
