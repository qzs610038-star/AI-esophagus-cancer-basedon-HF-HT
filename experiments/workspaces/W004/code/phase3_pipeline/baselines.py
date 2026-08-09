"""Leakage-safe low-capacity Phase 3 baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def mean_pool(patches: np.ndarray) -> np.ndarray:
    array = np.asarray(patches, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"patches must be a non-empty finite matrix, got {array.shape}")
    return array.mean(axis=0)


def pool_modalities(
    pathology_bags: list[np.ndarray],
    pathway_bags: list[np.ndarray],
    mode: str,
) -> np.ndarray:
    if len(pathology_bags) != len(pathway_bags):
        raise ValueError("pathology and pathway bag counts differ")
    pathology = np.vstack([mean_pool(bag) for bag in pathology_bags])
    pathway = np.vstack([mean_pool(bag) for bag in pathway_bags])
    if mode == "pathology":
        return pathology
    if mode == "pathway":
        return pathway
    if mode == "concat":
        return np.concatenate([pathology, pathway], axis=1)
    raise ValueError(f"unknown modality mode: {mode}")


def build_baseline(
    kind: str,
    *,
    random_state: int,
    pca_components: int = 30,
) -> Pipeline:
    """Build an unfitted estimator; callers must fit it on training rows only."""

    if kind == "logistic":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    if kind == "pca_logistic":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=random_state)),
                ("model", estimator),
            ]
        )
    if kind == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state)),
            ]
        )
    if kind == "random_forest":
        return Pipeline(
            [("model", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=1))]
        )
    raise ValueError(f"unknown baseline kind: {kind}")


@dataclass
class BaselineResult:
    validation_probability: np.ndarray
    test_probability: np.ndarray
    fitted_model: Pipeline


def fit_fold_baseline(
    kind: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    *,
    random_state: int,
    pca_components: int = 30,
) -> BaselineResult:
    """Fit exactly once on training rows and predict frozen validation/test rows."""

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    if x_train.ndim != 2 or y_train.ndim != 1 or len(x_train) != len(y_train):
        raise ValueError("training features/labels have incompatible shapes")
    if len(np.unique(y_train)) != 2:
        raise ValueError("training fold must contain both binary classes")
    components = min(pca_components, x_train.shape[0] - 1, x_train.shape[1])
    if kind == "pca_logistic" and components < 1:
        raise ValueError("not enough training rows for PCA")
    model = build_baseline(kind, random_state=random_state, pca_components=components)
    model.fit(x_train, y_train)
    return BaselineResult(
        validation_probability=model.predict_proba(np.asarray(x_validation))[:, 1],
        test_probability=model.predict_proba(np.asarray(x_test))[:, 1],
        fitted_model=model,
    )
