# DIRD

Pipeline for building a pulmonary adverse-drug-reaction (ADR) dataset, extracting molecular features, and training classifiers (KNN, SVM, random forest, XGBoost).

Vietnamese version: [README_vn.md](README_vn.md).

Run every command from this folder (`dird`).

```
dird/
  data_merging/       Phase A — curate sources and assemble the labeled table
  featurization/      Phase B — RDKit descriptors + ECFP4 fingerprints
  ml_experiments/     Phase C — splits, tuning, evaluation, statistics
  data/               All generated datasets (raw → cleaned → features → splits)
  results/            ML artifacts only (hyperparameters, probabilities, stats)
```

Generated tables always go under `data/` (or `results/` for ML). Nothing is written inside `data_merging/`, `featurization/`, or `ml_experiments/` except source code.

---

## End-to-end commands

Replace `Label_ILD` with `Label_Edema`, `Label_Pleural`, or `Label_Embolism` for the other endpoints. The cleaned table has **four binary labels**, not a column named `class`.

```bash
# A. Data merging
python data_merging/01_curate_sources.py
python data_merging/02_build_dataset.py --skip-scoring

# B. Featurization (default input: data/cleaned_data/balanced_pulmonary_adr_dataset.csv)
python featurization/feature_extraction.py --features both
python featurization/feature_extraction.py --features rdkit
python featurization/feature_extraction.py --features ecfp4

# C. Machine learning (example: ILD, AUROC, combined features)
python ml_experiments/01_data_splitting.py --input both --label-col Label_ILD
python ml_experiments/02_hyperparameter_tuning.py --metric auroc --label-col Label_ILD
python ml_experiments/03_model_refitting_and_evaluation.py --metric auroc --label-col Label_ILD
python ml_experiments/04_statistical_analysis.py --label-col Label_ILD
```

`--skip-scoring` reuses the PubChem cache already in `data/intermediate_data`. Omit it to rescore unlabeled drugs (slow).

If `cleaned_data` contains more than one CSV, pass `--input` to featurization with that filename.

---

## Phase A — Data merging (`data_merging/`)

Turn public sources into one labeled table:

`data/cleaned_data/balanced_pulmonary_adr_dataset.csv`

Step-by-step detail: [data_merging/README.md](data_merging/README.md).

| Script | Role |
|---|---|
| `01_curate_sources.py` | Filter ADReCS/Pneumotox, SIDER, DrugCentral, CTD; fetch SMILES; standardize with MEHC. Writes `data/curated_data/`. |
| `02_build_dataset.py` | Merge, assign four labels, deduplicate, score unlabeled rows on PubChem, write the cleaned table. |

Raw files belong in `data/raw_data/` (ADReCS, Pneumotox, SIDER, DrugCentral, CTD). DrugCentral curation requires ADReCS to have been curated first.

**Labels**

| Column | Meaning |
|---|---|
| `Label_ILD` | Interstitial lung disease / fibrosis |
| `Label_Edema` | Pulmonary edema |
| `Label_Pleural` | Pleural injury / effusion |
| `Label_Embolism` | Pulmonary embolism / thrombosis |

A row is a **positive** if any of these is 1. Rows with all zeros are **negative candidates**; they are kept as negatives only if PubChem PUG-View + assay scoring finds no pulmonary-ADR evidence in toxicity text.

Helpers: `data_merging/utils/` (`paths.py`, `mehc.py`, `pubchem.py`).

---

## Phase B — Featurization (`featurization/`)

Reads a CSV from `data/cleaned_data` (default name `balanced_pulmonary_adr_dataset.csv`; another file via `--input`).

Writes under `data/featurized_data/`:

| `--features` | File | Contents |
|---|---|---|
| `rdkit` | `rdkit_md_196.csv` | 196 RDKit descriptors |
| `ecfp4` | `ecfp4_2048.csv` | 2048-bit ECFP4 (`ecfp_bit0` …) |
| `both` | `both.csv` | descriptors + fingerprints |

Each file keeps SMILES and the label columns (`class` if present, otherwise the four `Label_*` columns).

---

## Phase C — Machine learning (`ml_experiments/`)

Binary classification **one endpoint at a time**. Pass `--label-col` (for this dataset: `Label_ILD`, `Label_Edema`, `Label_Pleural`, or `Label_Embolism`). Other `Label_*` columns are dropped so they are not used as features. Each label writes into its own folder, so four endpoints do not overwrite each other.

| Script | Input | Output |
|---|---|---|
| `01_data_splitting.py` | `data/featurized_data/` (`rdkit`, `ecfp4`, or `both`) | `data/split_data/{label}/{random,scaffold}/seed_{k}/train.csv` and `test.csv` |
| `02_hyperparameter_tuning.py` | training splits only | `results/{label}/tuned_hyperparameters/gridsearch_results_{auroc\|auprc}.csv` |
| `03_model_refitting_and_evaluation.py` | splits + tuning CSV | `results/{label}/predicted_probs/{metric}/{split}_{model}.csv` |
| `04_statistical_analysis.py` | predicted probabilities | `results/{label}/statistics/computed_statistics.csv` |

Both random and scaffold splits are stratified. GridSearchCV runs on **train folds only** (imblearn pipeline + SMOTE; scaler in `models/utils.py`: fingerprints unscaled, counts/fragments RobustScaler, `FractionCSP3` MinMaxScaler). `--metric` is required: `auroc` or `auprc`.

Models: `ml_experiments/models/ml_models.py` (`get_models`).

---

## Data layout

```
data/
  raw_data/              downloaded sources
  intermediate_data/     PubChem/MEHC checkpoints and phase-2 CSVs
  curated_data/          four tables after merging phase 1
  cleaned_data/          labeled dataset
  featurized_data/       RDKit / ECFP / both
  split_data/{label}/    train/test by split type and seed

results/{label}/
  tuned_hyperparameters/
  predicted_probs/
  statistics/
```
