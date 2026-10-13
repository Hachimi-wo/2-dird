# Data merging

This pipeline turns public sources into **one drug table labeled for pulmonary ADRs**:

`dird/data/cleaned_data/balanced_pulmonary_adr_dataset.csv`

Vietnamese version: [README_vn.md](README_vn.md).

There are **two scripts**. Run them in order from the `dird` repository root:

```bash
python data_merging/01_curate_sources.py
python data_merging/02_build_dataset.py --skip-scoring
```

| Script | What it does |
|---|---|
| `01_curate_sources.py` | **Phase 1 — curate each source.** Filter, fetch SMILES, standardize structures (MEHC), write four tables under `data/curated_data`. |
| `02_build_dataset.py` | **Phase 2 — assemble the dataset.** Merge the four tables, assign labels, deduplicate, validate negatives with PubChem, write the final table. |

Generated files **always** go under `dird/data`, never inside `data_merging/`.

---

## Folder layout

```
dird/
  data_merging/
    01_curate_sources.py
    02_build_dataset.py
    utils/                 shared helpers
    README.md
    README_vn.md
  data/
    raw_data/              downloaded sources (you place these here)
    intermediate_data/     checkpoints: API caches, MEHC, phase-2 CSVs
    curated_data/          four tables after phase 1
    cleaned_data/          final dataset after phase 2
```

| Folder | Role |
|---|---|
| `data_merging/utils` | `paths.py` (locations), `mehc.py` (SMILES standardization), `pubchem.py` (CID/SMILES lookup and negative scoring) |
| `data/raw_data` | Original downloads. A missing file raises an error; the scripts do not invent empty source folders. |
| `data/intermediate_data/phase1_curation` | Per-source CID/SMILES caches and MEHC output |
| `data/intermediate_data/phase2_assembly` | Numbered CSVs `01_` … `07_` — one file per merge step |
| `data/curated_data` | Phase-1 outputs; phase-2 inputs |
| `data/cleaned_data` | Final table for featurization / ML |

### Required source files

- `data/raw_data/adrecs/ADReCS_Respiratory_Drugs.csv` — drugs + respiratory ADR terms from ADReCS
- `data/raw_data/adrecs/pneumotox_full_database.csv` — pulmonary toxicity patterns from Pneumotox
- `data/raw_data/sider/meddra_all_se.tsv` — SIDER side effects (MedDRA)
- `data/raw_data/sider/drug_names.tsv` — SIDER drug names (STITCH IDs)
- `data/raw_data/drugcentral/drugcentral.structures.smiles.tsv` — DrugCentral structures (`INN`, `SMILES`)
- `data/raw_data/ctdbase/need2find.csv` — CTD mechanism markers (`Drug_Name`, `Pattern`)

---

## The four target labels

Each drug row has four binary columns:

| Column | Meaning |
|---|---|
| `Label_ILD` | Interstitial lung disease / pulmonary fibrosis / interstitial pneumonia |
| `Label_Edema` | Pulmonary edema |
| `Label_Pleural` | Pleural effusion / pleural injury |
| `Label_Embolism` | Pulmonary embolism / thrombosis |

- **Positive:** at least one label is 1.
- **Negative candidate:** all four labels are 0 — *not* yet trusted as a true negative.
- **Kept negative:** that candidate has PubChem score 0 (no pulmonary-ADR evidence in toxicity / adverse-effect text).

---

## Phase 1 — curate each source (`01_curate_sources.py`)

Phase 1 does **not** merge the four sources into one dataset. It processes each source on its own so every drug has a standardized SMILES before phase 2 concatenates them.

Run everything:

```bash
python data_merging/01_curate_sources.py
```

Run a subset (for example ADReCS then SIDER only):

```bash
python data_merging/01_curate_sources.py --source adrecs sider
```

Allowed `--source` values: `adrecs`, `sider`, `drugcentral`, `ctd`.

**DrugCentral requires ADReCS to be curated first**, because the script drops drugs already present in ADReCS/Pneumotox.

### What is MEHC?

MEHC (Molecular Entity Harmonization and Curation) standardizes SMILES (salts, tautomers, alternate writings) so the same active ingredient from ADReCS, SIDER, DrugCentral, and CTD can be compared, then deduplicated by SMILES in phase 2.

MEHC output and reports live under `data/intermediate_data/phase1_curation/<source>/mehc/`.

---

### Step 1.1 — ADReCS + Pneumotox

**Function:** `curate_adrecs()`  
**Output:** `data/curated_data/adrecs_pneumotox_curated.csv`

1. Read ADReCS and Pneumotox. Normalize drug names (strip hydrate, hydrochloride, sodium, …) so the same drug matches across salt forms.
2. Group ADReCS by the normalized name: one drug, multiple `ADR_TERM` values joined with `; `.
3. Merge with Pneumotox:
   - in both → `Source = Both`
   - Pneumotox only → `Pneumotox_only` (CID marked `TO_BE_FOUND`, looked up on PubChem later)
   - ADReCS only → `ADReCS_only`, `Pattern` written as `ADReCS: <term>`
4. Resolve missing CIDs, then batch-fetch Canonical SMILES (checkpointed, so a re-run keeps progress).
5. Run MEHC and write the curated table.

Useful checkpoints: `cid_checkpoint.csv`, `smiles_checkpoint.csv` in `intermediate_data/phase1_curation/adrecs/`.

---

### Step 1.2 — SIDER

**Function:** `curate_sider()`  
**Output:** `data/curated_data/sider_respiratory_adrs.csv`

1. Read `meddra_all_se.tsv` and keep drugs whose MedDRA term matches one of:
   - Pleural effusion
   - Pulmonary oedema
   - Pulmonary embolism
   - Interstitial pneumonia
2. Convert the STITCH ID (`CID1xxxxxxx`) to a PubChem CID by removing the `CID1` prefix, then fetch SMILES. Cache: `sider_filtered_nomehc.csv`.
3. Run MEHC, then reattach `STITCH_ID`, `Drug_Name`, and `Respiratory_Patterns`.

---

### Step 1.3 — DrugCentral

**Function:** `curate_drugcentral()`  
**Output:** `data/curated_data/drugcentral_smiles.csv`  
**Requires:** `adrecs_pneumotox_curated.csv` already exists.

1. Read DrugCentral (`INN`, `SMILES`).
2. Drop drugs already in the ADReCS/Pneumotox set (case-insensitive name match). What remains is an **unlabeled candidate pool**.
3. Run MEHC (`get_report=False` because the table is large).

Phase 2 will **not** treat every one of these rows as a negative. They enter the “all four labels = 0” group and are filtered by PubChem.

---

### Step 1.4 — CTD

**Function:** `curate_ctd()`  
**Output:** `data/curated_data/ctd_mechanism_markers.csv`

1. For each `Drug_Name`, try a few name variants (split on commas, strip parentheses) and ask PubChem for SMILES.
2. Run MEHC, then join back the original `Pattern` column.
3. These rows become **extra positives** in phase 2 (`ild`, `pulmonary edema`, `pleural effusion`, `pulmonary embolism`).

---

## Phase 2 — assemble the dataset (`02_build_dataset.py`)

Phase 2 reads the four curated tables, writes one intermediate per step under `data/intermediate_data/phase2_assembly/`, then writes the final CSV.

```bash
# Reuse existing PubChem scores (fast, when the cache is still valid)
python data_merging/02_build_dataset.py --skip-scoring

# Re-query PubChem for every negative candidate (slow)
python data_merging/02_build_dataset.py
```

`--skip-scoring` only works if `07_pubchem_scored_negatives.csv` already exists.

---

### Step 2.1 — Merge ADReCS/Pneumotox with SIDER

**File:** `01_merged_adrecs_sider.csv`

Concatenate the two tables. SIDER’s `Respiratory_Patterns` is renamed to `Pattern` so the schema matches. No labels yet.

---

### Step 2.2 — Assign the four labels from `Pattern`

**File:** `02_labeled_by_pattern.csv`

Read Pneumotox-style codes (`I.b`, `II.a`, `V.a`, `VI.a`, …) and text in `Pattern`:

| Label | Codes (examples) | Text keywords |
|---|---|---|
| ILD | `I.b`, `I.g`, `g` | interstitial, fibrosis, ILD |
| Edema | `II.a`, `II.d` | pulmonary oedema / edema |
| Pleural | `V.a` | pleural |
| Embolism | `VI.a`, `VI.b`, `VI.d` | embolism |

A drug can be positive for more than one label.

---

### Step 2.3 — Deduplicate by `Drug_Name`

**File:** `03_deduplicated_by_drugname.csv`

The same drug name can appear on several rows (Pneumotox plus an `ADReCS: …` line). Keep **one** row: prefer the Pneumotox record over ADReCS-only text.

---

### Step 2.4 — Add DrugCentral candidates

**File:** `04_augmented_drugcentral.csv`

Append DrugCentral drugs not already in the table. All four labels = 0. These are **negative candidates**, not final negatives.

---

### Step 2.5 — Add CTD positives

**File:** `05_merged_ctd_positive_pool.csv`

Map the CTD `Pattern`:

| CTD Pattern | Label set to 1 |
|---|---|
| `ild` | `Label_ILD` |
| `pulmonary edema` | `Label_Edema` |
| `pleural effusion` | `Label_Pleural` |
| `pulmonary embolism` | `Label_Embolism` |

---

### Step 2.6 — Split off negative candidates

**File:** `06_candidate_negatives_raw.csv`

Every row whose four labels still sum to 0. The rest of the pool is positive and goes to the final table without a PubChem check.

---

### Step 2.7 — Score with PubChem

**File:** `07_pubchem_scored_negatives.csv`  
**Cache:** `cache_pubchem_scores.csv`

For each candidate:

1. Resolve a CID: prefer an existing `PubChem_ID`, otherwise look up the name (punctuation stripped first, then the raw string).
2. Download PUG-View (compound record) and the bioassay summary.
3. Count pulmonary-ADR keywords **only** under adverse / toxicity / side-effect headings — not indications (a drug *used to treat* fibrosis must not become a false positive).
4. `score = PUG_View + Assay_Summary`. Only `score == 0` is kept as a negative.

API calls pause between rows (`SLEEP = 0.4` seconds). To skip a re-query: `--skip-scoring`.

---

### Step 2.8 — Write the final table

**File:** `data/cleaned_data/balanced_pulmonary_adr_dataset.csv`

1. Positives = rows with any label = 1.
2. Negatives = rows with `score == 0` from step 2.7.
3. Drop duplicate SMILES (`keep="first"` — positives are concatenated first, so they win).
4. Columns kept: `Drug_Name`, `Pattern`, `PubChem_ID`, `DrugBank_ID`, `KEGG_ID`, `SMILES`, and the four labels.

---

## Helpers in `utils/`

| File | What it does |
|---|---|
| `paths.py` | All absolute paths under `dird/data`. `ROOT` is `data_merging` (parent of `utils/`). |
| `mehc.py` | `run_mehc(df, output_dir)`: rename `SMILES` → `smiles` (MEHC’s expected name), refine, rename back to `SMILES`. |
| `pubchem.py` | Name → CID/SMILES; PUG-View + assay scoring. Keywords are grouped by the four labels; hits count only in toxicity text. |

Phase 1/2 scripts import as `from utils.paths import ...` (run `python data_merging/01_....py` from the `dird` root).

---

## Next steps in the project (outside this folder)

1. **Featurization** (`featurization/feature_extraction.py`) — read the cleaned table (or a refined copy) and write 196 RDKit descriptors + ECFP4 2048-bit fingerprints under `data/featurized_data`.
2. **ML** (`ml_experiments/`) — train/test splits, hyperparameter tuning, evaluation; artifacts go under `dird/results/` (not under `data/`).
