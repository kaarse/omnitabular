"""Extremely Randomized Trees (Extra Trees) wrapper for OmniTabular."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from .base import BaseModel
from .utils import detect_task


class ETModel(BaseModel):
    """
    Extremely Randomized Trees model with automatic task detection.

    Wraps :class:`sklearn.ensemble.ExtraTreesClassifier` or
    :class:`sklearn.ensemble.ExtraTreesRegressor`.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}
    n_estimators : int
        Number of trees in the forest (default 100).
    max_depth : int or None
        Maximum depth of the tree (default None).
    min_samples_split : int or float
        The minimum number of samples required to split an internal node (default 2).
    n_jobs : int or None
        Parallel jobs for training. -1 = all CPUs (default -1).
    random_state : int or None
    **kwargs
        Additional kwargs forwarded to the underlying sklearn estimator.
    """

    def __init__(
        self,
        task: str = "auto",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        n_jobs: Optional[int] = -1,
        random_state: Optional[int] = 42,
        **kwargs,
    ) -> None:
        super().__init__(task=task)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._extra_kwargs = kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "ETModel":
        X = self._to_numpy(X)
        y = np.asarray(y)

        if self.task == "auto":
            self.task = detect_task(y)

        if self.task == "classification":
            self._model = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **self._extra_kwargs,
            )
        else:
            self._model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **self._extra_kwargs,
            )

        self._model.fit(X, y)
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
        """Return feature importances (impurity-based)."""
        self._check_fitted()
        return self._model.feature_importances_

    def __repr__(self) -> str:
        return (
            f"ETModel(task='{self.task}', n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth})"
        )
