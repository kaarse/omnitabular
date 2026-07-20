"""Utility helpers for OmniTabular."""

from __future__ import annotations

from typing import List, Tuple, Callable, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

def detect_task(y: np.ndarray, threshold: int = 20) -> str:
    """
    Infer whether the target ``y`` represents a classification or regression task.

    Heuristic
    ---------
    * If ``y`` dtype is float AND the number of unique values exceeds ``threshold``,
      treat as regression.
    * Otherwise treat as classification.

    Parameters
    ----------
    y : array-like
    threshold : int
        Max number of unique values (for float targets) to still be considered
        classification.

    Returns
    -------
    "classification" or "regression"
    """
    y = np.asarray(y)
    n_unique = len(np.unique(y))
    if np.issubdtype(y.dtype, np.floating) and n_unique > threshold:
        return "regression"
    return "classification"


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

def _log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Numerically stable log-loss."""
    from sklearn.metrics import log_loss
    return log_loss(y_true, proba)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_true, y_pred))


def _auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1]
    return float(roc_auc_score(y_true, proba, multi_class="ovr"))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import mean_absolute_error
    return float(mean_absolute_error(y_true, y_pred))


METRIC_REGISTRY = {
    # name -> (func, higher_is_better, needs_proba)
    "log_loss":  (_log_loss,  False, True),
    "accuracy":  (_accuracy,  True,  False),
    "auc":       (_auc,       True,  True),
    "rmse":      (_rmse,      False, False),
    "mae":       (_mae,       False, False),
}


def get_default_metric(task: str) -> str:
    return "log_loss" if task == "classification" else "rmse"


def evaluate(
    metric_name: Union[str, Callable],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    higher_is_better: bool = False,
    needs_proba: bool = False,
) -> float:
    """Evaluate a metric by name or custom callable."""
    if callable(metric_name):
        if needs_proba:
            if proba is None:
                raise ValueError("Custom metric requires probability predictions.")
            return float(metric_name(y_true, proba))
        return float(metric_name(y_true, y_pred))
        
    func, higher_is_better, needs_proba = METRIC_REGISTRY[metric_name]
    if needs_proba:
        if proba is None:
            raise ValueError(f"Metric '{metric_name}' requires probability predictions.")
        return func(y_true, proba)
    return func(y_true, y_pred)


# ---------------------------------------------------------------------------
# Cross-validated OOF predictions
# ---------------------------------------------------------------------------

def oof_predictions(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-fold (OOF) predictions for a model class.

    Returns
    -------
    oof_pred : np.ndarray
        Hard OOF predictions (n_samples,).
    oof_proba : np.ndarray or None
        Probability OOF predictions (n_samples, n_classes) for classification,
        else None.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    oof_pred = np.zeros(n, dtype=float)

    if task == "classification":
        n_classes = len(np.unique(y))
        oof_proba = np.zeros((n, n_classes), dtype=float)
    else:
        oof_proba = None

    kfold = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if task == "classification"
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )

    for train_idx, val_idx in kfold.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]

        m = model_cls(**model_kwargs)
        m.fit(X_tr, y_tr)
        oof_pred[val_idx] = m.predict(X_val)

        if task == "classification":
            oof_proba[val_idx] = m.predict_proba(X_val)

    return oof_pred, oof_proba
