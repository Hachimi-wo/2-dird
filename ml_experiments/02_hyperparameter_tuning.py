"""Step 2/4: GridSearchCV on pre-split training sets.

Run after 01_data_splitting.py. Next: 03_model_refitting_and_evaluation.py

Loads training CSVs written by 01_data_splitting.py:

    data/split_data/{label}/random/seed_{k}/train.csv
    data/split_data/{label}/scaffold/seed_{k}/train.csv

For each seed, only the training set is used. Results are saved to
dird/results/{label}/tuned_hyperparameters as one CSV:
split, seed, model, best_parameters, metric.

The scoring metric is chosen by the user: 'auroc' or 'auprc'.
"""

from pathlib import Path
import json
import re
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.ml_models import get_models

RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_DATA_DIR = PROJECT_ROOT / "data" / "split_data"

RESULT_COLUMNS = ["split", "seed", "model", "best_parameters", "metric"]

METRIC_ALIASES = {
    "auroc": "roc_auc",
    "roc_auc": "roc_auc",
    "roc-auc": "roc_auc",
    "auc": "roc_auc",
    "auprc": "average_precision",
    "pr_auc": "average_precision",
    "prauc": "average_precision",
    "average_precision": "average_precision",
}

SPLIT_DIR_ALIASES = {
    "random": ("random", "random_split", "random-split"),
    "scaffold": ("scaffold", "scaffold_split", "scaffold-split", "scafold"),
}

def resolve_metric(metric):
    """Map auroc/auprc (and aliases) to a sklearn scoring name."""
    if metric is None:
        key = "auroc"
    else:
        key = str(metric).strip().lower().replace(" ", "_")
    if key not in METRIC_ALIASES:
        raise ValueError("metric must be 'auroc' or 'auprc'")
    return METRIC_ALIASES[key]


def metric_filename_tag(metric):
    """Canonical metric name for output files: auroc or auprc."""
    scoring = resolve_metric(metric)
    return "auroc" if scoring == "roc_auc" else "auprc"


def _parse_seed(path):
    match = re.search(r"seed[_-]?(\d+)", path.as_posix(), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def find_split_dir(data_dir, split):
    """Locate the folder that holds one outer-split type."""
    data_dir = Path(data_dir)
    for name in SPLIT_DIR_ALIASES[split]:
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No '{split}' split folder under {data_dir}. "
        f"Expected one of: {', '.join(SPLIT_DIR_ALIASES[split])}"
    )


def find_training_files(data_dir, split):
    """Find every seed's training file written by 01_data_splitting.py.

    Expected layout:
        {data_dir}/{split}/seed_{k}/train.csv
    """
    split_dir = find_split_dir(data_dir, split)
    files = {}
    for path in sorted(split_dir.glob("seed_*/train.csv")):
        if not path.is_file():
            continue
        seed = _parse_seed(path.parent)
        if seed is None:
            continue
        files[seed] = path

    if not files:
        raise FileNotFoundError(
            f"No training files found in {split_dir}. "
            "Expected seed_*/train.csv from 01_data_splitting.py"
        )
    return dict(sorted(files.items()))


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


def load_xy(path, label_col="class"):
    """Load a CSV that contains only feature columns and a target column."""
    df = pd.read_csv(path)
    if label_col not in df.columns:
        available = [c for c in KNOWN_LABEL_COLS if c in df.columns]
        hint = f" Available labels: {available}." if available else ""
        raise KeyError(
            f"{path} has no label column '{label_col}'.{hint} Columns: {list(df.columns)}"
        )
    extra = [c for c in KNOWN_LABEL_COLS if c in df.columns and c != label_col]
    X = df.drop(columns=[label_col] + extra)
    y = df[label_col]
    return X, y


def load_all_training_sets(data_dir, split, label_col="class"):
    """Load every seed's training set for one split type."""
    train_sets = {}
    for seed, path in find_training_files(data_dir, split).items():
        train_sets[seed] = load_xy(path, label_col=label_col)
    return train_sets


def grid_search_one_model(
    name,
    pipeline,
    params,
    X,
    y,
    cv,
    scoring,
    n_jobs=4,
    verbose=0,
):
    """Run GridSearchCV on one training set and one metric."""
    search = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=params,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        error_score=float("nan"),
    )
    search.fit(X, y)
    return {
        "model": name,
        "best_parameters": json.dumps(search.best_params_),
        "metric": float(search.best_score_),
    }


def run_seed_grid_search(
    X_train,
    y_train,
    seed,
    scoring,
    split=None,
    n_splits=5,
    n_jobs=4,
    verbose=0,
):
    """Tune every model on a single seed's training data."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = get_models(X_train, random_state=seed)

    rows = []
    for name, spec in models.items():
        row = grid_search_one_model(
            name=name,
            pipeline=spec["pipeline"],
            params=spec["params"],
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        row["split"] = split
        row["seed"] = seed
        rows.append(row)

    return pd.DataFrame(rows)[RESULT_COLUMNS]


def run_split_experiments(
    train_sets,
    scoring,
    split=None,
    n_splits=5,
    n_jobs=4,
    verbose=0,
):
    """Run GridSearchCV on all seeds of one split type."""
    frames = [
        run_seed_grid_search(
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            scoring=scoring,
            split=split,
            n_splits=n_splits,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        for seed, (X_train, y_train) in sorted(train_sets.items())
    ]
    return pd.concat(frames, ignore_index=True)


def run_all_experiments(
    data_dir=None,
    splits=("random", "scaffold"),
    metric="auroc",
    label_col="class",
    n_splits=5,
    n_jobs=4,
    verbose=0,
    output_dir=None,
    output_path=None,
):
    """Load training seeds from data/split_data/{label}, run GridSearchCV, save one CSV.

    metric: 'auroc' or 'auprc'
    Results are saved under dird/results/{label}/tuned_hyperparameters.
    """
    data_dir = Path(data_dir) if data_dir is not None else SPLIT_DATA_DIR / label_col
    scoring = resolve_metric(metric)
    frames = []

    for split in splits:
        train_sets = load_all_training_sets(
            data_dir=data_dir,
            split=split,
            label_col=label_col,
        )
        frames.append(
            run_split_experiments(
                train_sets=train_sets,
                scoring=scoring,
                split=split,
                n_splits=n_splits,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )

    results = pd.concat(frames, ignore_index=True)[RESULT_COLUMNS]

    metric_tag = metric_filename_tag(metric)
    if output_path is None:
        output_dir = (
            Path(output_dir)
            if output_dir is not None
            else RESULTS_DIR / label_col / "tuned_hyperparameters"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"gridsearch_results_{metric_tag}.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 2/4: GridSearchCV hyperparameter tuning on pre-split training sets."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder written by 01_data_splitting.py. Default: data/split_data/{label-col}.",
    )
    parser.add_argument(
        "--label-col",
        default="class",
        help="Binary target used when splitting (default: class). For the cleaned dataset: Label_ILD, Label_Edema, Label_Pleural, or Label_Embolism.",
    )
    parser.add_argument(
        "--metric",
        choices=["auroc", "auprc"],
        required=True,
        help="Metric used for GridSearchCV: auroc (ROC-AUC) or auprc (average precision).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    run_all_experiments(
        data_dir=args.data_dir,
        metric=args.metric,
        label_col=args.label_col,
        output_dir=args.output_dir,
        output_path=args.output_path,
    )
