"""Run MEHC SMILES standardization on a table.

MEHC (Molecular Entity Harmonization and Curation) canonicalizes structures
(salts, tautomers, representation quirks) so the same drug from ADReCS,
SIDER, DrugCentral, and CTD can be compared and later deduplicated by SMILES.

The ``mehc_curation`` import is deferred: phase-2 scoring does not need MEHC,
and importing RDKit/MEHC at module load would slow every ``utils`` import.
"""

from pathlib import Path

import pandas as pd


def run_mehc(df, output_dir, smiles_col="SMILES", get_report=True):
    """Standardize SMILES with MEHC and return the refined table.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a SMILES column (``SMILES`` or ``smiles``). Extra columns
        are kept and written alongside the refined structures.
    output_dir : path-like
        MEHC writes a ``refinement/`` subdirectory here (reports + CSV). Re-runs
        of the same source reuse this folder as a checkpoint.
    smiles_col : str
        Column to rename to ``smiles`` if MEHC's expected lowercase name is
        missing. Pipeline tables use ``SMILES``; MEHC uses ``smiles``.
    get_report : bool
        If True, MEHC also writes a human-readable refinement report. DrugCentral
        is large, so phase 1 turns this off there.

    Returns
    -------
    pandas.DataFrame
        Refined rows with the SMILES column named ``SMILES`` again so later
        scripts do not have to special-case MEHC's lowercase name.

    Notes
    -----
    ``n_cpu=1`` avoids multiprocessing issues on Windows when the scripts are
    launched as ``python data_merging/01_curate_sources.py``.
    """
    from mehc_curation.refinement import RefinementStage

    frame = df.copy()
    # MEHC looks for "smiles". Upstream tables use "SMILES".
    if smiles_col in frame.columns and "smiles" not in frame.columns:
        frame = frame.rename(columns={smiles_col: "smiles"})
    if "smiles" not in frame.columns:
        raise KeyError("No SMILES column for MEHC")

    # Empty SMILES cannot be refined and would crash the RDKit parser.
    frame = frame.dropna(subset=["smiles"]).reset_index(drop=True)
    # RefinementStage expects the SMILES column first.
    cols = ["smiles"] + [c for c in frame.columns if c != "smiles"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refined = RefinementStage(frame[cols]).complete_refinement(
        output_dir=str(output_dir),
        get_report=get_report,
        n_cpu=1,
    )

    # Some MEHC versions return None and only write the CSV. Fall back to that
    # file rather than treating an otherwise successful run as a failure.
    if refined is None:
        path = output_dir / "refinement" / "post_refined_smiles.csv"
        if not path.is_file():
            raise FileNotFoundError(f"MEHC did not write {path}")
        refined = pd.read_csv(path)

    if "smiles" in refined.columns and "SMILES" not in refined.columns:
        refined = refined.rename(columns={"smiles": "SMILES"})
    return refined
