"""XGBoost wrapper for OmniTabular."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "XGBoost is required for XGBModel. Install it with: pip install xgboost"
    ) from exc

from .base import BaseModel
from .utils import detect_task


class XGBModel(BaseModel):
    """
    XGBoost model with automatic task detection.

    Wraps :class:`xgboost.XGBClassifier` or :class:`xgboost.XGBRegressor`.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
    max_depth : int
        Maximum tree depth.
    subsample : float
        Row subsampling fraction.
    colsample_bytree : float
        Feature subsampling fraction.
    n_jobs : int
    random_state : int
    **kwargs
        Extra parameters forwarded to the underlying XGB estimator.
    """

    def __init__(
        self,
        task: str = "auto",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(task=task)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._extra_kwargs = kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "XGBModel":
        X = self._to_numpy(X)
        y = np.asarray(y)

        if self.task == "auto":
            self.task = detect_task(y)

        if self.task == "classification":
            # For multi-class classification, XGBoost requires consecutive integer labels starting from 0.
            # In our use cases we mostly pass already processed data, but we instantiate Classifier directly.
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self._extra_kwargs,
            )
        else:
            self._model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self._extra_kwargs,
            )

        self._model.fit(X, y, **fit_params)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(self._to_numpy(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is not available for regression.")
        return self._model.predict_proba(self._to_numpy(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances."""
        self._check_fitted()
        return self._model.feature_importances_

    def __repr__(self) -> str:
        return (
            f"XGBModel(task='{self.task}', n_estimators={self.n_estimators}, "
            f"lr={self.learning_rate}, max_depth={self.max_depth})"
        )
