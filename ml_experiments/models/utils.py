from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import numpy as np
import pandas as pd


# RDKit descriptors that are naturally bounded.
BOUNDED_DESCRIPTORS = {
    "FractionCSP3",
}

FINGERPRINT_NAME_PREFIXES = (
    "bit_",
    "fp_",
    "ecfp_bit",
    "ecfp",
    "fcfp",
    "morgan_",
    "maccs",
    "avalon_",
    "fingerprint",
    "rdkfp",
    "pubchem",
)


def _is_fingerprint_name(name):
    text = str(name).strip()
    if text.isdigit():
        return True
    lower = text.lower().replace("-", "_")
    return any(lower.startswith(prefix) for prefix in FINGERPRINT_NAME_PREFIXES)


def _is_binary_01(column):
    values = pd.Series(column).dropna()
    if values.empty:
        return False
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        return False
    unique = np.unique(numeric.to_numpy())
    return np.isin(unique, [0, 1]).all()


def _column_values(X, name, index):
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        if name in X.columns:
            return X[name]
        return X.iloc[:, index]
    return np.asarray(X)[:, index]


def get_rdkit_column_transformer(descriptor_names, X=None):
    """Build a ColumnTransformer for RDKit descriptors and fingerprints."""

    fingerprint_cols = []
    count_cols = []
    bounded_cols = []
    continuous_cols = []

    for i, name in enumerate(descriptor_names):
        values = _column_values(X, name, i)
        is_binary = values is not None and _is_binary_01(values)

        # Binary fingerprints (0/1) are left unscaled.
        if _is_fingerprint_name(name) and (values is None or is_binary):
            fingerprint_cols.append(i)
            continue

        # Count-like fingerprints still use the count scaler.
        if _is_fingerprint_name(name):
            count_cols.append(i)
            continue

        # Fragment and count-like descriptors.
        if name.startswith("fr_") or "Count" in name or name.startswith("Num"):
            count_cols.append(i)

        # Naturally bounded descriptors.
        elif name in BOUNDED_DESCRIPTORS:
            bounded_cols.append(i)

        # Remaining continuous descriptors.
        else:
            continuous_cols.append(i)

    fingerprint_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("variance", VarianceThreshold(threshold=0.01)),
    ])

    count_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.01)),
        ("scaler", RobustScaler()),
    ])

    bounded_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.01)),
        ("scaler", MinMaxScaler()),
    ])

    continuous_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.01)),
        ("scaler", RobustScaler()),
    ])

    transformers = []
    if fingerprint_cols:
        transformers.append(("fingerprint", fingerprint_pipeline, fingerprint_cols))
    if count_cols:
        transformers.append(("count", count_pipeline, count_cols))
    if bounded_cols:
        transformers.append(("bounded", bounded_pipeline, bounded_cols))
    if continuous_cols:
        transformers.append(("continuous", continuous_pipeline, continuous_cols))

    col_transformer = ColumnTransformer(transformers)
    return col_transformer
