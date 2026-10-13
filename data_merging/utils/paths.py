"""Project paths for the data-merging pipeline.

Layout
------
This file lives at ``dird/data_merging/utils/paths.py``, so:

* ``ROOT`` is ``dird/data_merging`` (parent of ``utils/``)
* ``PROJECT_ROOT`` is ``dird`` (the repo root)
* generated tables always go under ``dird/data``, never under ``data_merging/``

Stages
------
raw_data
    Downloaded sources (ADReCS, Pneumotox, SIDER, DrugCentral, CTD).
intermediate_data
    Checkpoints (CID/SMILES caches, MEHC output, phase-2 assembly CSVs).
curated_data
    One curated table per source after filtering + SMILES + MEHC.
cleaned_data
    The final labeled dataset consumed by featurization.
"""

from pathlib import Path

# data_merging/utils/paths.py -> data_merging -> dird
ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
DATA = PROJECT_ROOT / "data"

RAW = DATA / "raw_data"
INTERMEDIATE = DATA / "intermediate_data"
CURATED = DATA / "curated_data"
CLEANED = DATA / "cleaned_data"

# Per-source working dirs (API caches, MEHC folders, filtered intermediates).
ADRECS_DIR = INTERMEDIATE / "phase1_curation" / "adrecs"
SIDER_DIR = INTERMEDIATE / "phase1_curation" / "sider"
DRUGCENTRAL_DIR = INTERMEDIATE / "phase1_curation" / "drugcentral"
CTD_DIR = INTERMEDIATE / "phase1_curation" / "ctdbase"
ASSEMBLY_DIR = INTERMEDIATE / "phase2_assembly"

# Expected raw filenames. Scripts fail loudly if a file is missing rather than
# searching for alternatives.
RAW_ADRECS = RAW / "adrecs" / "ADReCS_Respiratory_Drugs.csv"
RAW_PNEUMOTOX = RAW / "adrecs" / "pneumotox_full_database.csv"
RAW_SIDER_SE = RAW / "sider" / "meddra_all_se.tsv"
RAW_SIDER_NAMES = RAW / "sider" / "drug_names.tsv"
RAW_DRUGCENTRAL = RAW / "drugcentral" / "drugcentral.structures.smiles.tsv"
RAW_CTD = RAW / "ctdbase" / "need2find.csv"

# Phase-1 outputs consumed by 02_build_dataset.py.
CURATED_ADRECS = CURATED / "adrecs_pneumotox_curated.csv"
CURATED_SIDER = CURATED / "sider_respiratory_adrs.csv"
CURATED_DRUGCENTRAL = CURATED / "drugcentral_smiles.csv"
CURATED_CTD = CURATED / "ctd_mechanism_markers.csv"

# Final table written by phase 2 (then featurized from cleaned_data).
FINAL_DATASET = CLEANED / "balanced_pulmonary_adr_dataset.csv"

# Four binary pulmonary ADR labels. A row is a positive if any of these is 1;
# unlabeled rows (all zeros) are PubChem-scored and kept only if score == 0.
LABEL_COLS = ["Label_ILD", "Label_Edema", "Label_Pleural", "Label_Embolism"]


def ensure_dirs():
    """Create the data folders the pipeline writes into.

    Raw download dirs are not created here on purpose: missing raw files
    should raise a clear FileNotFoundError instead of an empty directory.
    """
    for path in (
        RAW,
        CURATED,
        CLEANED,
        ADRECS_DIR,
        SIDER_DIR,
        DRUGCENTRAL_DIR,
        CTD_DIR,
        ASSEMBLY_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
