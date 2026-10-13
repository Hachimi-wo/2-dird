"""Step 4/4: Compute metrics, 95% CIs, and paired t-tests from predicted probabilities.

Run after 03_model_refitting_and_evaluation.py

Reads:
    results/{label}/predicted_probs/{tuning_metric}/{split}_{model}.csv

Writes one file:
    results/{label}/statistics/computed_statistics.csv
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_COLUMNS = [
    "analysis",
    "tuning_metric",
    "split",
    "metric",
    "model",
    "paired_model",
    "n",
    "mean",
    "sd",
    "ci_95_lower",
    "ci_95_upper",
    "t_statistic",
    "p_value",
]


def compute_metrics(y_true, y_proba):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= 0.5).astype(int)

    def _safe(fn, default=np.nan):
        try:
            return float(fn())
        except ValueError:
            return default

    return {
        "auroc": _safe(lambda: roc_auc_score(y_true, y_proba)),
        "auprc": _safe(lambda: average_precision_score(y_true, y_proba)),
        "balanced_accuracy": _safe(lambda: balanced_accuracy_score(y_true, y_pred)),
        "f1": _safe(lambda: f1_score(y_true, y_pred, zero_division=0)),
        "precision": _safe(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "mcc": _safe(lambda: matthews_corrcoef(y_true, y_pred)),
        "kappa": _safe(lambda: cohen_kappa_score(y_true, y_pred)),
    }


def mean_sd_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return n, np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if n == 1:
        return n, mean, np.nan, np.nan, np.nan
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return n, mean, sd, mean - tcrit * se, mean + tcrit * se


def parse_pred_path(path, pred_dir):
    split, model = path.stem.rsplit("_", 1)
    parent = path.parent
    tuning_metric = parent.name if parent.resolve() != pred_dir.resolve() else ""
    return tuning_metric, split, model


def collect_seed_metrics(pred_dir):
    rows = []
    paths = list(pred_dir.glob("*.csv")) + list(pred_dir.glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No predicted probability CSVs in {pred_dir}")

    for path in sorted(set(paths)):
        tuning_metric, split, model = parse_pred_path(path, pred_dir)
        df = pd.read_csv(path)
        for seed, part in df.groupby("seed"):
            scores = compute_metrics(part["y_true"], part["probability"])
            for metric_name, value in scores.items():
                rows.append(
                    {
                        "tuning_metric": tuning_metric,
                        "split": split,
                        "model": model,
                        "seed": int(seed),
                        "metric": metric_name,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def summary_rows(seed_metrics):
    rows = []
    group_cols = ["tuning_metric", "split", "model", "metric"]
    for keys, part in seed_metrics.groupby(group_cols, dropna=False):
        n, mean, sd, lo, hi = mean_sd_ci(part["value"])
        tuning_metric, split, model, metric_name = keys
        rows.append(
            {
                "analysis": "summary",
                "tuning_metric": tuning_metric,
                "split": split,
                "metric": metric_name,
                "model": model,
                "paired_model": "",
                "n": n,
                "mean": mean,
                "sd": sd,
                "ci_95_lower": lo,
                "ci_95_upper": hi,
                "t_statistic": np.nan,
                "p_value": np.nan,
            }
        )
    return rows


def paired_ttest_rows(seed_metrics):
    rows = []
    group_cols = ["tuning_metric", "split", "metric"]
    for keys, part in seed_metrics.groupby(group_cols, dropna=False):
        tuning_metric, split, metric_name = keys
        models = sorted(part["model"].unique())
        for model_a, model_b in combinations(models, 2):
            a = part.loc[part["model"] == model_a, ["seed", "value"]]
            b = part.loc[part["model"] == model_b, ["seed", "value"]]
            paired = a.merge(b, on="seed", suffixes=("_a", "_b"))
            paired = paired.dropna()
            n = len(paired)
            if n < 2:
                t_stat, p_value = np.nan, np.nan
            else:
                t_stat, p_value = stats.ttest_rel(paired["value_a"], paired["value_b"])
            rows.append(
                {
                    "analysis": "paired_ttest",
                    "tuning_metric": tuning_metric,
                    "split": split,
                    "metric": metric_name,
                    "model": model_a,
                    "paired_model": model_b,
                    "n": n,
                    "mean": np.nan,
                    "sd": np.nan,
                    "ci_95_lower": np.nan,
                    "ci_95_upper": np.nan,
                    "t_statistic": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                }
            )
    return rows


def run_statistical_analysis(pred_dir=None, output_path=None, decimals=4, label_col=None):
    if pred_dir is None:
        if not label_col:
            raise ValueError("Pass --label-col or --pred-dir")
        pred_dir = RESULTS_DIR / label_col / "predicted_probs"
    if output_path is None:
        if not label_col:
            raise ValueError("Pass --label-col or --output-path")
        output_path = RESULTS_DIR / label_col / "statistics" / "computed_statistics.csv"
    pred_dir = Path(pred_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decimals = int(decimals)
    if decimals < 0:
        raise ValueError("decimals must be >= 0")

    seed_metrics = collect_seed_metrics(pred_dir)
    out = pd.DataFrame(
        summary_rows(seed_metrics) + paired_ttest_rows(seed_metrics),
        columns=OUTPUT_COLUMNS,
    )
    out.to_csv(output_path, index=False, float_format=f"%.{decimals}f")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 4/4: Metrics, 95% CIs, and paired t-tests from predicted probabilities."
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Endpoint folder under results/{label-col}. Required unless --pred-dir and --output-path are set.",
    )
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimal places in computed_statistics.csv (default: 4).",
    )
    args = parser.parse_args()
    run_statistical_analysis(
        pred_dir=args.pred_dir,
        output_path=args.output_path,
        decimals=args.decimals,
        label_col=args.label_col,
    )
