"""KNN wrapper for OmniTabular."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .base import BaseModel
from .utils import detect_task


class KNNModel(BaseModel):
    """
    K-Nearest Neighbours model with automatic task detection.

    Wraps :class:`sklearn.neighbors.KNeighborsClassifier` or
    :class:`sklearn.neighbors.KNeighborsRegressor` depending on the task.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}
    n_neighbors : int
        Number of neighbours (default 5).
    weights : {"uniform", "distance"}
        Weight function for neighbours.
    metric : str
        Distance metric (default "minkowski").
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}
    n_jobs : int or None
        Parallel jobs. -1 = all CPUs.
    **kwargs
        Additional kwargs forwarded to the underlying sklearn estimator.

    Examples
    --------
    >>> from omnitabular import KNNModel
    >>> import numpy as np
    >>> X = np.random.rand(100, 4)
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> m = KNNModel(task="classification", n_neighbors=7)
    >>> m.fit(X, y)
    >>> m.predict(X[:5])
    """

    def __init__(
        self,
        task: str = "auto",
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "minkowski",
        algorithm: str = "auto",
        n_jobs: Optional[int] = -1,
        **kwargs,
    ) -> None:
        super().__init__(task=task)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self._extra_kwargs = kwargs
        self._model = None

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "KNNModel":
        X = self._to_numpy(X)
        y = np.asarray(y)

        if self.task == "auto":
            self.task = detect_task(y)

        knn_cls = (
            KNeighborsClassifier if self.task == "classification" else KNeighborsRegressor
        )

        self._model = knn_cls(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
            **self._extra_kwargs,
        )
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is not available for regression.")
        X = self._to_numpy(X)
        return self._model.predict_proba(X)

    def __repr__(self) -> str:
        return (
            f"KNNModel(task='{self.task}', n_neighbors={self.n_neighbors}, "
            f"weights='{self.weights}', metric='{self.metric}')"
        )
