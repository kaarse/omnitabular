"""CatBoost wrapper for OmniTabular."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import catboost as cb
except ImportError as exc:
    raise ImportError(
        "CatBoost is required for CatBoostModel. Install it with: pip install catboost"
    ) from exc

from .base import BaseModel
from .utils import detect_task


class CatBoostModel(BaseModel):
    """
    CatBoost model with automatic task detection.

    Wraps :class:`catboost.CatBoostClassifier` or :class:`catboost.CatBoostRegressor`.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}
    iterations : int
        Number of boosting rounds.
    learning_rate : float
    depth : int
        Maximum tree depth.
    random_seed : int
    verbose : int or bool
        Verbosity level (default False to suppress output).
    **kwargs
        Extra parameters forwarded to the underlying CatBoost estimator.
    """

    def __init__(
        self,
        task: str = "auto",
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(task=task)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.verbose = verbose
        self._extra_kwargs = kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "CatBoostModel":
        X = self._to_numpy(X)
        y = np.asarray(y)

        if self.task == "auto":
            self.task = detect_task(y)

        if self.task == "classification":
            self._model = cb.CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                random_seed=self.random_seed,
                verbose=self.verbose,
                **self._extra_kwargs,
            )
        else:
            self._model = cb.CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                random_seed=self.random_seed,
                verbose=self.verbose,
                **self._extra_kwargs,
            )

        self._model.fit(X, y, **fit_params)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        preds = self._model.predict(self._to_numpy(X))
        return np.ravel(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is not available for regression.")
        return self._model.predict_proba(self._to_numpy(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances."""
        self._check_fitted()
        return self._model.get_feature_importance()

    def __repr__(self) -> str:
        return (
            f"CatBoostModel(task='{self.task}', iterations={self.iterations}, "
            f"lr={self.learning_rate}, depth={self.depth})"
        )
