"""Step 1/4: Create n random and scaffold train/test splits from a SMILES dataset.

Run after featurization. Next: 02_hyperparameter_tuning.py

Input (under dird/data/featurized_data):

    rdkit_md_196.csv
    ecfp4_2048.csv
    both.csv

Output layout (under dird/data/split_data/{label}):

    data/split_data/{label}/random/seed_{k}/train.csv
    data/split_data/{label}/random/seed_{k}/test.csv
    data/split_data/{label}/scaffold/seed_{k}/train.csv
    data/split_data/{label}/scaffold/seed_{k}/test.csv

Both random and scaffold splits are stratified by the label.
Saved CSVs contain only feature columns and the label column (SMILES dropped).
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FEATURIZED_DIR = PROJECT_ROOT / "data" / "featurized_data"
SPLIT_DATA_DIR = PROJECT_ROOT / "data" / "split_data"

FEATURIZED_INPUTS = {
    "rdkit": "rdkit_md_196.csv",
    "ecfp4": "ecfp4_2048.csv",
    "both": "both.csv",
}

# Label columns that must not be used as features when training one endpoint.
KNOWN_LABEL_COLS = [
    "class",
    "Class",
    "CLASS",
    "label",
    "Label",
    "target",
    "Target",
    "Label_ILD",
    "Label_Edema",
    "Label_Pleural",
    "Label_Embolism",
]


def resolve_featurized_input(input_path):
    """Resolve a featurized CSV under dird/data/featurized_data."""
    raw = Path(input_path)
    if raw.is_file():
        return raw

    key = str(input_path).strip().lower()
    name = FEATURIZED_INPUTS.get(key) or FEATURIZED_INPUTS.get(raw.name.lower())
    if name is None:
        name = raw.name
    path = FEATURIZED_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"No featurized file at {path}. Expected --input rdkit, ecfp4, or both."
        )
    return path


def resolve_smiles_column(df, smiles_col=None):
    """Use 'SMILES' or 'smiles' (or an explicit column name)."""
    if smiles_col:
        if smiles_col not in df.columns:
            raise KeyError(
                f"No SMILES column '{smiles_col}'. Columns: {list(df.columns)}"
            )
        return smiles_col
    for name in ("SMILES", "smiles"):
        if name in df.columns:
            return name
    matches = [c for c in df.columns if str(c).lower() == "smiles"]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"No 'SMILES' or 'smiles' column. Columns: {list(df.columns)}"
    )


def murcko_scaffold(smiles, index):
    """Bemis-Murcko scaffold; unique fallback for invalid or acyclic molecules."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles("" if smiles is None else str(smiles))
    if mol is None:
        return f"invalid_{index}"

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(scaffold)


def random_split(df, label_col, test_size, seed):
    """Stratified random train/test split."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_col],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def scaffold_split(df, smiles_col, label_col, test_size, seed):
    """Scaffold split, stratified by label.

    Molecules that share a Murcko scaffold stay in the same set. Each
    scaffold is assigned to train or test to match the per-class test size.
    """
    y = df[label_col].to_numpy()
    groups = defaultdict(list)
    for i, smiles in enumerate(df[smiles_col]):
        groups[murcko_scaffold(smiles, i)].append(i)

    classes, class_counts = np.unique(y, return_counts=True)
    target_test = {c: cnt * test_size for c, cnt in zip(classes, class_counts)}

    rng = np.random.RandomState(seed)
    order = list(groups)
    rng.shuffle(order)

    train_idx, test_idx = [], []
    test_counts = {c: 0.0 for c in classes}

    for scaffold in order:
        members = groups[scaffold]
        add = {c: np.sum(y[members] == c) for c in classes}
        err_test = sum((test_counts[c] + add[c] - target_test[c]) ** 2 for c in classes)
        err_train = sum((test_counts[c] - target_test[c]) ** 2 for c in classes)
        if err_test <= err_train:
            test_idx.extend(members)
            for c in classes:
                test_counts[c] += add[c]
        else:
            train_idx.extend(members)

    if not test_idx:
        moved = groups[order[-1]]
        test_idx = list(moved)
        train_idx = [i for i in train_idx if i not in set(moved)]
    elif not train_idx:
        moved = groups[order[-1]]
        train_idx = list(moved)
        test_idx = [i for i in test_idx if i not in set(moved)]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def _save_xy(df, path, smiles_col, label_col):
    """Write features + the chosen label only (drop SMILES and other Label_* columns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    drop = [smiles_col]
    for col in KNOWN_LABEL_COLS:
        if col in df.columns and col != label_col:
            drop.append(col)
    df.drop(columns=drop).to_csv(path, index=False)


def split_dataset(
    input_path,
    output_dir=None,
    smiles_col=None,
    label_col="class",
    n_seeds=5,
    test_size=0.2,
    splits=("random", "scaffold"),
    clean_negatives=True,
):
    """Write n seeds of random and scaffold splits for 02_hyperparameter_tuning.py."""
    input_path = resolve_featurized_input(input_path)
    output_dir = (
        Path(output_dir) if output_dir is not None else SPLIT_DATA_DIR / label_col
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    smiles_col = resolve_smiles_column(df, smiles_col)
    if label_col not in df.columns:
        available = [c for c in KNOWN_LABEL_COLS if c in df.columns]
        hint = f" Available labels: {available}." if available else ""
        raise KeyError(
            f"No label column '{label_col}'.{hint} Columns: {list(df.columns)}"
        )
    if clean_negatives:
        other_labels = [c for c in KNOWN_LABEL_COLS if c.startswith("Label_") and c in df.columns and c != label_col]
        all_zero = df[other_labels + [label_col]].sum(axis=1).eq(0)
        df = df[(df[label_col] == 1) | all_zero].reset_index(drop=True)
        print(f"clean-negative filter {label_col}: {len(df)} rows")

    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(n_seeds):
            if split == "random":
                train, test = random_split(df, label_col, test_size, seed)
            elif split == "scaffold":
                train, test = scaffold_split(df, smiles_col, label_col, test_size, seed)
            else:
                raise ValueError("split must be 'random' or 'scaffold'")

            seed_dir = output_dir / split / f"seed_{seed}"
            _save_xy(train, seed_dir / "train.csv", smiles_col, label_col)
            _save_xy(test, seed_dir / "test.csv", smiles_col, label_col)
            print(f"{split} seed {seed}: train={len(train)} test={len(test)} -> {seed_dir}")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 1/4: Create random and scaffold splits for 02_hyperparameter_tuning.py"
    )
    parser.add_argument(
        "--input",
        required=True,
        choices=["rdkit", "ecfp4", "both"],
        help="Featurized file in dird/data/featurized_data: rdkit, ecfp4, or both.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Split folder. Default: data/split_data/{label-col}.",
    )
    parser.add_argument(
        "--smiles-col",
        default=None,
        help="SMILES column name. Default: use 'SMILES' or 'smiles'.",
    )
    parser.add_argument(
        "--label-col",
        default="class",
        help="Binary target. Use Label_ILD, Label_Edema, Label_Pleural, or Label_Embolism for the cleaned dataset (default: class).",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--target-zero-negatives",
        action="store_true",
        help="Use every target-label zero as negative (legacy policy).",
    )
    args = parser.parse_args()

    split_dataset(
        input_path=args.input,
        output_dir=args.output,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        n_seeds=args.n_seeds,
        test_size=args.test_size,
        clean_negatives=not args.target_zero_negatives,
    )
