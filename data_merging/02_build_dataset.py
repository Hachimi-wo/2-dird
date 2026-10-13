"""Phase 2: merge curated sources, label, validate negatives, write cleaned dataset."""

import csv
import re
import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from utils.paths import (
    ASSEMBLY_DIR,
    CURATED_ADRECS,
    CURATED_CTD,
    CURATED_DRUGCENTRAL,
    CURATED_SIDER,
    FINAL_DATASET,
    LABEL_COLS,
    ensure_dirs,
)
from utils.pubchem import SLEEP, assay_summary, pug_view, resolve_cid, score_assay, score_pug

CODE_RE = re.compile(r"[A-Z]+[a-z]*\.[a-z]+")
ADR_RE = re.compile(r"ADReCS:\s*(.*?)(?=\n|$)", re.IGNORECASE)


def _cache_value(value):
    """Normalize values used to identify one scoring input."""
    return "" if pd.isna(value) else str(value).strip()


def _cache_key(row):
    """Key a PubChem score by the exact molecule-identifying input."""
    return tuple(
        _cache_value(row.get(column))
        for column in ("Drug_Name", "PubChem_ID", "SMILES")
    )


def extract_codes(pattern):
    return [] if pd.isna(pattern) else CODE_RE.findall(pattern)


def extract_clauses(pattern):
    if pd.isna(pattern):
        return []
    return [m.strip() for m in ADR_RE.findall(pattern) if m.strip()]


def _hit(codes, clauses_low, pat_low, code_prefixes, text_keys):
    for code in codes:
        if code.startswith(tuple(code_prefixes)) or code in code_prefixes:
            return 1
    for clause in clauses_low:
        if any(key in clause for key in text_keys):
            return 1
    if any(key in pat_low for key in text_keys):
        return 1
    return 0


def assign_labels(pattern):
    pat_low = str(pattern).lower() if not pd.isna(pattern) else ""
    codes = extract_codes(pattern)
    clauses = [c.lower() for c in extract_clauses(pattern)]
    ild = _hit(codes, clauses, pat_low, ["I.b", "I.g", "g"], ["interstitial", "fibrosis", "ILD"])
    edema = _hit(
        codes,
        clauses,
        pat_low,
        ["II.a", "II.d"],
        ["pulmonary oedema", "pulmonary edema", "noncardiogenic pulmonary edema", "noncardiogenic pulmonary oedema"],
    )
    pleural = _hit(codes, clauses, pat_low, ["V.a"], ["pleural"])
    embolism = _hit(codes, clauses, pat_low, ["VI.a", "VI.b", "VI.d"], ["embolism"])
    return ild, edema, pleural, embolism


def ctd_labels(pattern):
    mapping = {
        "ild": {"Label_ILD": 1, "Label_Edema": 0, "Label_Pleural": 0, "Label_Embolism": 0},
        "pulmonary edema": {"Label_ILD": 0, "Label_Edema": 1, "Label_Pleural": 0, "Label_Embolism": 0},
        "pleural effusion": {"Label_ILD": 0, "Label_Edema": 0, "Label_Pleural": 1, "Label_Embolism": 0},
        "pulmonary embolism": {"Label_ILD": 0, "Label_Edema": 0, "Label_Pleural": 0, "Label_Embolism": 1},
    }
    return mapping.get(str(pattern).strip().casefold(), {c: 0 for c in LABEL_COLS})


def merge_adrecs_sider():
    adrecs = pd.read_csv(CURATED_ADRECS)
    sider = (
        pd.read_csv(CURATED_SIDER)[["Drug_Name", "Respiratory_Patterns", "SMILES"]]
        .rename(columns={"Respiratory_Patterns": "Pattern"})
    )
    merged = pd.concat([adrecs, sider], ignore_index=True)[adrecs.columns.tolist()]
    path = ASSEMBLY_DIR / "01_merged_adrecs_sider.csv"
    merged.to_csv(path, index=False)
    return merged


def annotate_labels(merged):
    labels = merged["Pattern"].apply(assign_labels)
    merged = merged.copy()
    merged["Label_ILD"] = [x[0] for x in labels]
    merged["Label_Edema"] = [x[1] for x in labels]
    merged["Label_Pleural"] = [x[2] for x in labels]
    merged["Label_Embolism"] = [x[3] for x in labels]
    path = ASSEMBLY_DIR / "02_labeled_by_pattern.csv"
    merged.to_csv(path, index=False)
    return path


def deduplicate_by_drugname(labeled_path):
    groups = defaultdict(list)
    with open(labeled_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        for i, row in enumerate(reader):
            groups[row["Drug_Name"]].append((i, row))
    picked = []
    for rows in groups.values():
        first = min(rows, key=lambda x: x[0])
        last = max(rows, key=lambda x: x[0])
        picked.append(first if "ADReCS" in last[1].get("Pattern", "") else last)
    picked.sort(key=lambda x: x[0])
    path = ASSEMBLY_DIR / "03_deduplicated_by_drugname.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(row for _, row in picked)
    return pd.read_csv(path)


def augment_drugcentral(positives):
    dc = pd.read_csv(CURATED_DRUGCENTRAL)
    if "Unnamed: 0" in dc.columns:
        dc = dc.drop(columns=["Unnamed: 0"])
    known = set(positives["Drug_Name"].str.lower())
    dc = dc[~dc["INN"].str.lower().isin(known)]
    extra = pd.DataFrame({
        "Drug_Name": dc["INN"],
        "Pattern": "",
        "PubChem_ID": None,
        "DrugBank_ID": None,
        "KEGG_ID": None,
        "SMILES": dc["SMILES"],
        "Label_ILD": 0,
        "Label_Edema": 0,
        "Label_Embolism": 0,
        "Label_Pleural": 0,
    })
    out = pd.concat([positives, extra], ignore_index=True)
    path = ASSEMBLY_DIR / "04_augmented_drugcentral.csv"
    out.to_csv(path, index=False)
    return out


def integrate_ctd(pool):
    ctd = pd.read_csv(CURATED_CTD)
    labeled = pd.concat(
        [ctd.reset_index(drop=True), pd.DataFrame(ctd["Pattern"].apply(ctd_labels).tolist())],
        axis=1,
    )
    for col in LABEL_COLS:
        if col not in pool.columns:
            pool[col] = 0
    out = pd.concat([pool, labeled], ignore_index=True, sort=False)
    path = ASSEMBLY_DIR / "05_merged_ctd_positive_pool.csv"
    out.to_csv(path, index=False)
    return out


def candidate_negatives(pool):
    neg = pool[pool[LABEL_COLS].sum(axis=1) == 0].copy()
    path = ASSEMBLY_DIR / "06_candidate_negatives_raw.csv"
    neg.to_csv(path, index=False)
    return neg


def score_negatives(candidates, use_cache=True):
    cache_path = ASSEMBLY_DIR / "cache_pubchem_scores.csv"
    cache = {}
    if use_cache and cache_path.is_file():
        cache = {
            _cache_key(row): row
            for row in pd.read_csv(cache_path).to_dict("records")
        }
    results = []
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="PubChem score"):
        name = row.get("Drug_Name", "")
        key = _cache_key(row)
        if key in cache:
            results.append(cache[key])
            continue
        cid, src = resolve_cid(row.get("PubChem_ID", ""), name)
        if not cid:
            scored = {
                "Drug_Name": name,
                "SMILES": row.get("SMILES", ""),
                "PubChem_ID": row.get("PubChem_ID", ""),
                "CID": None,
                "CID_Source": "none",
                "score": 0,
                "status": "no_cid",
                "PUG_View": 0,
                "Assay_Summary": 0,
            }
        else:
            pug = pug_view(cid)
            assay = assay_summary(cid)
            pug_score = score_pug(pug) if pug else 0
            assay_score = score_assay(assay) if assay else 0
            scored = {
                "Drug_Name": name,
                "SMILES": row.get("SMILES", ""),
                "PubChem_ID": row.get("PubChem_ID", ""),
                "CID": cid,
                "CID_Source": src,
                "score": pug_score + assay_score,
                "status": "ok" if (pug or assay) else "no_data",
                "PUG_View": pug_score,
                "Assay_Summary": assay_score,
            }
        results.append(scored)
        cache[key] = scored
        if len(results) % 50 == 0:
            pd.DataFrame(cache.values()).to_csv(cache_path, index=False)
        time.sleep(SLEEP)
    pd.DataFrame(cache.values()).to_csv(cache_path, index=False)
    scored_df = pd.DataFrame(results)
    path = ASSEMBLY_DIR / "07_pubchem_scored_negatives.csv"
    scored_df.to_csv(path, index=False)
    return scored_df


def assemble_final(pool, scored):
    positive = pool[pool[LABEL_COLS].any(axis=1)].copy()
    neg = scored[(scored["status"] == "ok") & (scored["score"] == 0)]
    negative = pd.DataFrame({
        "Drug_Name": neg["Drug_Name"],
        "Pattern": "",
        "PubChem_ID": neg["PubChem_ID"],
        "DrugBank_ID": "",
        "KEGG_ID": "",
        "SMILES": neg["SMILES"],
        "Label_ILD": 0,
        "Label_Edema": 0,
        "Label_Pleural": 0,
        "Label_Embolism": 0,
    })
    df = pd.concat([positive, negative], ignore_index=True)
    df = df[["Drug_Name", "Pattern", "PubChem_ID", "DrugBank_ID", "KEGG_ID", "SMILES"] + LABEL_COLS]
    df = df.drop_duplicates(subset=["SMILES"], keep="first")
    FINAL_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_DATASET, index=False)
    print(f"positive={len(positive)} negative={len(negative)} total={len(df)} -> {FINAL_DATASET}")
    return df


def main(skip_scoring=False):
    ensure_dirs()
    merged = merge_adrecs_sider()
    labeled_path = annotate_labels(merged)
    positives = deduplicate_by_drugname(labeled_path)
    pool = augment_drugcentral(positives)
    pool = integrate_ctd(pool)
    candidates = candidate_negatives(pool)
    scored_path = ASSEMBLY_DIR / "07_pubchem_scored_negatives.csv"
    if skip_scoring and scored_path.is_file():
        scored = pd.read_csv(scored_path)
    else:
        scored = score_negatives(candidates)
    assemble_final(pool, scored)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the cleaned pulmonary ADR dataset from curated sources."
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Reuse 07_pubchem_scored_negatives.csv instead of calling PubChem.",
    )
    args = parser.parse_args()
    main(skip_scoring=args.skip_scoring)
