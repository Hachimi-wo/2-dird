from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .utils import get_rdkit_column_transformer


def _build_pipeline(classifier, preprocessor, random_state=42):
    """Build a leakage-safe classification pipeline."""
    return Pipeline([
        ("preprocess", clone(preprocessor)),
        ("smote", SMOTE(random_state=random_state)),
        ("clf", classifier),
    ])


def get_models(X, random_state=42):
    """Create model pipelines and their hyperparameter search spaces."""
    descriptor_names = list(X.columns) if hasattr(X, "columns") else list(range(X.shape[1]))
    preprocessor = get_rdkit_column_transformer(descriptor_names, X)

    models = {
        "KNN": {
            "pipeline": _build_pipeline(
                CalibratedClassifierCV(
                    KNeighborsClassifier(weights="distance", n_jobs=1),
                    method="sigmoid",
                    ensemble=False,
                ),
                preprocessor,
                random_state=random_state,
            ),
            "params": {
                "clf__estimator__n_neighbors": list(range(3, 15)),
                "clf__estimator__p": [1, 2],
            },
        },

        "SVM": {
            "pipeline": _build_pipeline(
                CalibratedClassifierCV(
                    SVC(kernel="rbf", random_state=random_state),
                    method="sigmoid",
                    ensemble=False,
                ),
                preprocessor,
                random_state=random_state,
            ),
            "params": {
                "clf__estimator__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "clf__estimator__gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "clf__estimator__class_weight": [None, "balanced"],
            },
        },

        "RF": {
            "pipeline": _build_pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=1,
                ),
                preprocessor,
                random_state=random_state,
            ),
            "params": {
                "clf__max_depth": [2, 3, 4, 5, 6],
                "clf__min_samples_split": [2, 3, 4, 5, 6],
                "clf__max_features": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },

        "XGB": {
            "pipeline": _build_pipeline(
                XGBClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=1,
                    eval_metric="logloss",
                ),
                preprocessor,
                random_state=random_state,
            ),
            "params": {
                "clf__learning_rate": [0.001, 0.01, 0.1],
                "clf__max_depth": [2, 3, 4, 5, 6],
                "clf__colsample_bytree": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
    }

    return models
