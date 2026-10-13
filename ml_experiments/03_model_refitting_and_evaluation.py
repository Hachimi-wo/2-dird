"""Step 3/4: Refit tuned models on the full training set and score the test set.

Run after 02_hyperparameter_tuning.py. Next: 04_statistical_analysis.py

Reads best_parameters from:
    results/{label}/tuned_hyperparameters/gridsearch_results_{metric}.csv

Reads split files from:
    data/split_data/{label}/{split}/seed_{k}/train.csv
    data/split_data/{label}/{split}/seed_{k}/test.csv

Writes one probability CSV per split and model type (all seeds in that file):
    results/{label}/predicted_probs/{metric}/{split}_{model}.csv
"""

from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.base import clone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.ml_models import get_models

SPLIT_DATA_DIR = PROJECT_ROOT / "data" / "split_data"
RESULTS_DIR = PROJECT_ROOT / "results"


def metric_filename_tag(metric):
    key = str(metric).strip().lower().replace(" ", "_")
    if key in {"auroc", "roc_auc", "roc-auc", "auc"}:
        return "auroc"
    if key in {"auprc", "pr_auc", "prauc", "average_precision"}:
        return "auprc"
    raise ValueError("metric must be 'auroc' or 'auprc'")


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
    df = pd.read_csv(path)
    if label_col not in df.columns:
        available = [c for c in KNOWN_LABEL_COLS if c in df.columns]
        hint = f" Available labels: {available}." if available else ""
        raise KeyError(
            f"{path} has no label column '{label_col}'.{hint} Columns: {list(df.columns)}"
        )
    extra = [c for c in KNOWN_LABEL_COLS if c in df.columns and c != label_col]
    return df.drop(columns=[label_col] + extra), df[label_col]


def load_split_xy(split, seed, label_col="class", data_dir=None):
    root = Path(data_dir) if data_dir is not None else SPLIT_DATA_DIR / label_col
    seed_dir = root / split / f"seed_{seed}"
    train_path = seed_dir / "train.csv"
    test_path = seed_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Missing train/test in {seed_dir}")
    X_train, y_train = load_xy(train_path, label_col=label_col)
    X_test, y_test = load_xy(test_path, label_col=label_col)
    return X_train, y_train, X_test, y_test


def load_tuning_results(metric, label_col="class"):
    path = (
        RESULTS_DIR
        / label_col
        / "tuned_hyperparameters"
        / f"gridsearch_results_{metric_filename_tag(metric)}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"No tuning results at {path}")
    return pd.read_csv(path)


def build_pipeline(model_name, best_parameters, seed, X):
    models = get_models(X, random_state=int(seed))
    if model_name not in models:
        raise KeyError(f"Unknown model '{model_name}'. Expected one of: {list(models)}")
    pipeline = clone(models[model_name]["pipeline"])
    pipeline.set_params(**json.loads(best_parameters))
    return pipeline


def predict_positive_proba(pipeline, X):
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    positive = 1 if 1 in classes else classes[-1]
    return proba[:, classes.index(positive)]


def refit_and_evaluate(
    metric,
    label_col="class",
    data_dir=None,
    tuning_path=None,
    output_dir=None,
):
    """Refit each tuned model on train, predict test probabilities, save CSVs."""
    data_dir = Path(data_dir) if data_dir is not None else SPLIT_DATA_DIR / label_col
    metric_tag = metric_filename_tag(metric)
    if tuning_path is None:
        tuning = load_tuning_results(metric, label_col=label_col)
    else:
        tuning = pd.read_csv(tuning_path)

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else RESULTS_DIR / label_col / "predicted_probs" / metric_tag
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    for row in tuning.itertuples(index=False):
        split = row.split
        seed = int(row.seed)
        model_name = row.model
        X_train, y_train, X_test, y_test = load_split_xy(
            split, seed, label_col=label_col, data_dir=data_dir
        )
        pipeline = build_pipeline(
            model_name,
            row.best_parameters,
            seed,
            X_train,
        )
        pipeline.fit(X_train, y_train)
        probability = predict_positive_proba(pipeline, X_test)

        pred = pd.DataFrame(
            {
                "seed": seed,
                "sample_index": range(len(y_test)),
                "y_true": y_test.to_numpy(),
                "probability": probability,
            }
        )
        grouped.setdefault((split, model_name), []).append(pred)

    saved = {}
    for (split, model_name), frames in grouped.items():
        out = pd.concat(frames, ignore_index=True)
        path = output_dir / f"{split}_{model_name}.csv"
        out.to_csv(path, index=False)
        saved[(split, model_name)] = path
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 3/4: Refit tuned models on train and save test probabilities."
    )
    parser.add_argument(
        "--metric",
        choices=["auroc", "auprc"],
        required=True,
        help="Which hyperparameter-tuning CSV to use: auroc or auprc.",
    )
    parser.add_argument(
        "--label-col",
        default="class",
        help="Binary target used when splitting (default: class). For the cleaned dataset: Label_ILD, Label_Edema, Label_Pleural, or Label_Embolism.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Split folder. Default: data/split_data/{label-col}.",
    )
    parser.add_argument("--tuning-path", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Probability folder. Default: results/{label-col}/predicted_probs/{metric}.",
    )
    args = parser.parse_args()

    refit_and_evaluate(
        metric=args.metric,
        label_col=args.label_col,
        data_dir=args.data_dir,
        tuning_path=args.tuning_path,
        output_dir=args.output_dir,
    )
