"""Phase 1: curate ADReCS/Pneumotox, SIDER, DrugCentral, and CTD into curated_data."""

import csv
import gzip
import os
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

from utils.mehc import run_mehc
from utils.paths import (
    ADRECS_DIR,
    CTD_DIR,
    CURATED_ADRECS,
    CURATED_CTD,
    CURATED_DRUGCENTRAL,
    CURATED_SIDER,
    DRUGCENTRAL_DIR,
    RAW_ADRECS,
    RAW_CTD,
    RAW_DRUGCENTRAL,
    RAW_PNEUMOTOX,
    RAW_SIDER_NAMES,
    RAW_SIDER_SE,
    SIDER_DIR,
    ensure_dirs,
)
from utils.pubchem import (
    cid_from_name,
    is_valid_smiles,
    smiles_batch,
    smiles_from_name,
)

REMOVE_TOKENS = [
    r"\bmonohydrate\b", r"\bdihydrate\b", r"\btrihydrate\b",
    r"\bhydrochloride\b", r"\bhydrobromide\b", r"\bmesylate\b",
    r"\bmaleate\b", r"\bfumarate\b", r"\bacetate\b", r"\bpropionate\b",
    r"\bsulfate\b", r"\bphosphate\b", r"\bcilexetil\b", r"\bexetil\b",
    r"\bproxetil\b", r"\bsodium\b", r"\bpotassium\b", r"\bcalcium\b",
    r"\bmagnesium\b", r"\bzinc\b", r"\b(?:i\.v\.|iv)\b",
]
SIDER_PATTERNS = {
    "Pleural effusion", "Pulmonary oedema",
    "Pulmonary embolism", "Interstitial pneumonia",
}


def read_csv_any(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def normalize_drug_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    for pat in REMOVE_TOKENS:
        name = re.sub(pat, "", name)
    return re.sub(r"\s+", " ", name).strip()


def sider_pubchem_cid(stitch_id):
    """Convert SIDER's ``CID1...`` identifier to the PubChem CID."""
    value = str(stitch_id).strip()
    if value.startswith("CID1"):
        value = value[4:]
    elif value.startswith("CID"):
        value = value[3:]
    return str(int(value)) if value.isdigit() else value


def smiles_from_chembl(db_id):
    if not db_id or not str(db_id).startswith("DB"):
        return ""
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/molecule"
        f"?molecule_dictionary__drugbank_id={db_id}&format=json"
    )
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            mols = r.json().get("molecules") or []
            if mols:
                return (mols[0].get("molecule_structures") or {}).get("canonical_smiles", "") or ""
    except Exception:
        return ""
    return ""


def curate_adrecs():
    ensure_dirs()
    adrecs = read_csv_any(RAW_ADRECS)
    pneumo = read_csv_any(RAW_PNEUMOTOX)
    adrecs["norm"] = adrecs["DRUG_NAME"].apply(normalize_drug_name)
    adrecs = adrecs[adrecs["norm"] != ""]
    grouped = adrecs.groupby("norm", as_index=False).agg({
        "DRUG_NAME": lambda x: x.iloc[0],
        "PubChem_ID": lambda x: x.iloc[0] if len(x) > 0 else None,
        "ADR_TERM": lambda x: "; ".join(sorted(set(x))),
    }).rename(columns={"DRUG_NAME": "drug_name_rep", "PubChem_ID": "pubchem_id_rep"})

    pneumo["norm"] = pneumo["Drug_Name"].apply(normalize_drug_name)
    pneumo = pneumo[pneumo["norm"] != ""].drop_duplicates(subset=["Drug_Name", "Pattern"])
    adrecs_norms = set(grouped["norm"])
    pneumo_norms = set(pneumo["norm"])
    adrecs_only = adrecs_norms - pneumo_norms
    pneumo_only = pneumo_norms - adrecs_norms

    only_df = grouped[grouped["norm"].isin(adrecs_only)].copy()
    only_df["Pattern"] = "ADReCS: " + only_df["ADR_TERM"]
    only_df = only_df.rename(columns={"drug_name_rep": "Drug_Name", "pubchem_id_rep": "PubChem_ID"})
    only_df[["Drug_Name", "norm", "PubChem_ID"]].to_csv(
        ADRECS_DIR / "adrecs_only_drugs.csv", index=False, encoding="utf-8-sig"
    )

    merged = pneumo[["Drug_Name", "Pattern", "norm"]].copy()
    pubchem_map = dict(zip(grouped["norm"], grouped["pubchem_id_rep"]))
    merged["PubChem_ID"] = merged["norm"].map(pubchem_map)
    merged.loc[merged["norm"].isin(pneumo_only), "PubChem_ID"] = "TO_BE_FOUND"
    merged = pd.concat(
        [merged, only_df[["Drug_Name", "Pattern", "norm", "PubChem_ID"]]],
        ignore_index=True,
    )
    merged["Source"] = merged["norm"].map(
        lambda n: "Pneumotox_only" if n in pneumo_only else "ADReCS_only" if n in adrecs_only else "Both"
    )
    merged = merged[["Drug_Name", "Pattern", "PubChem_ID", "Source", "norm"]]
    merged_path = ADRECS_DIR / "merged_database_v2.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    cid_cp_path = ADRECS_DIR / "cid_checkpoint.csv"
    cid_cp = {}
    if cid_cp_path.is_file():
        cp = pd.read_csv(cid_cp_path)
        cid_cp = dict(zip(cp["Drug_Name"], cp["PubChem_ID"]))
    todo = merged[merged["PubChem_ID"] == "TO_BE_FOUND"]["Drug_Name"].unique()
    resolved = {}
    for name in tqdm(todo, desc="ADReCS CID"):
        if name in cid_cp:
            resolved[name] = cid_cp[name]
            continue
        resolved[name] = cid_from_name(name) or "NOT_FOUND"
        cid_cp.update(resolved)
        pd.DataFrame(cid_cp.items(), columns=["Drug_Name", "PubChem_ID"]).to_csv(
            cid_cp_path, index=False
        )
        time.sleep(0.4)
    merged["PubChem_ID"] = merged.apply(
        lambda r: resolved.get(r["Drug_Name"], "NOT_FOUND") if r["PubChem_ID"] == "TO_BE_FOUND" else r["PubChem_ID"],
        axis=1,
    )

    cids = []
    for value in merged["PubChem_ID"].unique():
        try:
            cids.append(int(float(value)))
        except Exception:
            pass
    mapping = smiles_batch(cids, checkpoint=ADRECS_DIR / "smiles_checkpoint.csv")

    def lookup(value):
        if pd.isna(value):
            return ""
        key = str(value).replace(".0", "")
        return mapping.get(key, "") if key.isdigit() else ""

    merged["SMILES"] = merged["PubChem_ID"].apply(lookup)
    if "DrugBank_ID" in merged.columns:
        missing = merged["SMILES"].isna() | (merged["SMILES"] == "")
        for idx, row in merged[missing].iterrows():
            extra = smiles_from_chembl(str(row.get("DrugBank_ID", "")).strip())
            if extra:
                merged.at[idx, "SMILES"] = extra
            time.sleep(0.3)
    chembl_path = ADRECS_DIR / "adrecs_chembl_smiles.csv"
    merged.to_csv(chembl_path, index=False, encoding="utf-8-sig")

    refined = run_mehc(merged, ADRECS_DIR / "mehc")
    raw_u = merged.drop_duplicates(subset=["Drug_Name"], keep="first")
    cur_u = refined.drop_duplicates(subset=["Drug_Name"], keep="first")
    out = raw_u.merge(
        cur_u[["Drug_Name", "SMILES"]].rename(columns={"SMILES": "SMILES_cur"}),
        on="Drug_Name",
        how="inner",
        suffixes=("_raw", "_cur"),
    )
    if "SMILES_raw" in out.columns:
        out["SMILES"] = out["SMILES_cur"]
        out = out.drop(columns=["SMILES_raw", "SMILES_cur"])
    elif "SMILES_cur" in out.columns:
        out["SMILES"] = out["SMILES_cur"]
        out = out.drop(columns=["SMILES_cur"])
    cols = [c for c in merged.columns if c in out.columns]
    if "SMILES" not in cols:
        cols.append("SMILES")
    CURATED_ADRECS.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(CURATED_ADRECS, index=False, encoding="utf-8-sig")
    print(f"ADReCS curated: {len(out)} -> {CURATED_ADRECS}")


def _open_any(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def curate_sider():
    ensure_dirs()
    pattern_map = {p.lower(): p for p in SIDER_PATTERNS}
    names = {}
    if RAW_SIDER_NAMES.is_file():
        with _open_any(RAW_SIDER_NAMES) as handle:
            names = {row[0].strip(): row[1].strip() for row in csv.reader(handle, delimiter="\t") if len(row) >= 2}

    drug_patterns = {}
    with _open_any(RAW_SIDER_SE) as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 6:
                continue
            low = row[5].strip().lower()
            matched = {orig for key, orig in pattern_map.items() if key == low or key in low}
            if matched:
                drug_patterns.setdefault(row[0].strip(), set()).update(matched)

    out_raw = SIDER_DIR / "sider_filtered_nomehc.csv"
    cached = {}
    if out_raw.is_file():
        with open(out_raw, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if is_valid_smiles(row.get("SMILES")):
                    cached[row["STITCH_ID"]] = row

    missing_cids = [
        sider_pubchem_cid(stitch_id)
        for stitch_id in drug_patterns
        if stitch_id not in cached
    ]
    fetched = smiles_batch(
        missing_cids,
        checkpoint=SIDER_DIR / "smiles_checkpoint.csv",
        batch=5,
        delay=1.5,
    )

    results = []
    for stitch_id, patterns in drug_patterns.items():
        if stitch_id in cached:
            results.append(cached[stitch_id])
            continue
        cid = sider_pubchem_cid(stitch_id)
        results.append({
            "STITCH_ID": stitch_id,
            "Drug_Name": names.get(stitch_id, f"STITCH_{stitch_id}"),
            "SMILES": fetched.get(str(cid), "NA"),
            "Respiratory_Patterns": "; ".join(sorted(patterns)),
        })
    partial_path = out_raw.with_suffix(".partial.csv")
    with open(partial_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["STITCH_ID", "Drug_Name", "SMILES", "Respiratory_Patterns"])
        writer.writeheader()
        writer.writerows(results)
        handle.flush()
        os.fsync(handle.fileno())
    partial_path.replace(out_raw)

    raw = pd.read_csv(out_raw)
    raw["index"] = range(1, len(raw) + 1)
    mehc_in = raw[["SMILES", "index"]].copy()
    mehc_in.to_csv(SIDER_DIR / "respiratory_labeled_for_curation.csv", index=False, encoding="utf-8-sig")
    refined = run_mehc(mehc_in, SIDER_DIR / "mehc")
    if "index" not in refined.columns:
        refined["index"] = range(1, len(refined) + 1)
    merged = refined.merge(raw[["index", "STITCH_ID", "Drug_Name", "Respiratory_Patterns"]], on="index", how="left")
    CURATED_SIDER.parent.mkdir(parents=True, exist_ok=True)
    merged[["STITCH_ID", "Drug_Name", "SMILES", "Respiratory_Patterns"]].to_csv(
        CURATED_SIDER, index=False, encoding="utf-8-sig"
    )
    print(f"SIDER curated: {len(merged)} -> {CURATED_SIDER}")


def curate_drugcentral():
    ensure_dirs()
    toxic = pd.read_csv(CURATED_ADRECS).dropna(subset=["Drug_Name"])
    dc = pd.read_csv(RAW_DRUGCENTRAL, sep="\t").dropna(subset=["INN"])
    known = set(toxic["Drug_Name"].astype(str).str.lower().str.strip())
    dc_only = dc[~dc["INN"].astype(str).str.lower().str.strip().isin(known)].copy()
    dc_only.to_csv(DRUGCENTRAL_DIR / "drugcentralonly.csv", index=False, encoding="utf-8")
    refined = run_mehc(dc_only, DRUGCENTRAL_DIR / "mehc_output", get_report=False)
    CURATED_DRUGCENTRAL.parent.mkdir(parents=True, exist_ok=True)
    refined.to_csv(CURATED_DRUGCENTRAL, index=False, encoding="utf-8")
    print(f"DrugCentral curated: {len(refined)} -> {CURATED_DRUGCENTRAL}")


def _ctd_name_variants(name):
    variants = [name]
    if "," in name:
        variants.extend(part.strip() for part in name.split(","))
    cleaned = re.sub(r"[,\s]+", " ", re.sub(r"\([^)]*\)", "", name)).strip()
    if cleaned and cleaned != name:
        variants.append(cleaned)
    return list(dict.fromkeys(v for v in variants if v))


def curate_ctd():
    ensure_dirs()
    raw = pd.read_csv(RAW_CTD, encoding="utf-8")
    names = raw["Drug_Name"].dropna().unique().tolist()
    checkpoint_path = CTD_DIR / "ctdbase_smiles.csv"
    cached = {}
    if checkpoint_path.is_file():
        checkpoint = pd.read_csv(checkpoint_path)
        cached = {
            row["Drug_Name"]: row["SMILES"]
            for row in checkpoint.to_dict("records")
            if is_valid_smiles(row.get("SMILES"))
        }

    rows = []
    for name in tqdm(names, desc="CTD SMILES"):
        smiles = cached.get(name)
        if not smiles:
            for variant in _ctd_name_variants(name):
                smiles = smiles_from_name(variant)
                if smiles:
                    break
        cached[name] = smiles
        rows.append({"Drug_Name": name, "SMILES": smiles})
        checkpoint = pd.DataFrame(
            [{"Drug_Name": key, "SMILES": cached.get(key)} for key in names]
        )
        checkpoint.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
        time.sleep(1.0)
    smiles_df = pd.DataFrame(rows)
    smiles_df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
    refined = run_mehc(smiles_df, CTD_DIR / "mehc")
    if "SMILES" not in refined.columns and "smiles" in refined.columns:
        refined = refined.rename(columns={"smiles": "SMILES"})
    merged = refined.merge(raw[["Drug_Name", "Pattern"]], on="Drug_Name", how="inner")
    merged = merged[["SMILES", "Drug_Name", "Pattern"]].drop_duplicates()
    CURATED_CTD.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CURATED_CTD, index=False, encoding="utf-8-sig")
    print(f"CTD curated: {len(merged)} -> {CURATED_CTD}")


SOURCES = {
    "adrecs": curate_adrecs,
    "sider": curate_sider,
    "drugcentral": curate_drugcentral,
    "ctd": curate_ctd,
}


def main(sources=None):
    ensure_dirs()
    order = sources or ["adrecs", "sider", "drugcentral", "ctd"]
    for name in order:
        SOURCES[name]()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Curate raw sources into data/curated_data.")
    parser.add_argument(
        "--source",
        nargs="+",
        choices=list(SOURCES),
        default=list(SOURCES),
        help="Which sources to curate (default: all).",
    )
    args = parser.parse_args()
    main(args.source)
