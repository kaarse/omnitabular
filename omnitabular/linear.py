"""Linear / Logistic Regression wrapper for OmniTabular."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .utils import detect_task


class LinearModel(BaseModel):
    """
    Linear / Logistic Regression model with automatic task detection.

    * **Classification** → ``sklearn.linear_model.LogisticRegression``
      (multinomial, lbfgs solver by default).
    * **Regression** → ``sklearn.linear_model.LinearRegression``.

    Features are automatically standard-scaled before fitting.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}
    C : float
        Inverse regularisation strength (classification only, default 1.0).
    penalty : {"l1", "l2", "elasticnet", None}
        Penalty type for logistic regression.
    solver : str
        Optimisation solver for logistic regression (default "lbfgs").
    max_iter : int
        Max iterations for logistic regression convergence.
    fit_intercept : bool
    random_state : int or None
    n_jobs : int or None
    scale : bool
        Whether to standard-scale features before fitting (recommended).
    **kwargs
        Additional kwargs forwarded to the underlying sklearn estimator.

    Examples
    --------
    >>> from omnitabular import LinearModel
    >>> import numpy as np
    >>> X = np.random.rand(200, 4)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
    >>> m = LinearModel(task="classification")
    >>> m.fit(X, y)
    >>> m.predict_proba(X[:5])
    """

    def __init__(
        self,
        task: str = "auto",
        C: float = 1.0,
        penalty: Optional[str] = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        fit_intercept: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = -1,
        scale: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(task=task)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scale = scale
        self._extra_kwargs = kwargs
        self._model = None
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "LinearModel":
        X = self._to_numpy(X)
        y = np.asarray(y)

        if self.task == "auto":
            self.task = detect_task(y)

        # Optional scaling
        if self.scale:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        if self.task == "classification":
            self._model = LogisticRegression(
                C=self.C,
                l1_ratio=0,  # equivalent to L2 penalty
                solver=self.solver,
                max_iter=self.max_iter,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
                **self._extra_kwargs,
            )
        else:
            self._model = LinearRegression(
                fit_intercept=self.fit_intercept,
                n_jobs=self.n_jobs,
                **self._extra_kwargs,
            )

        self._model.fit(X, y)
        self._fitted = True
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = self._to_numpy(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(self._transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is not available for regression.")
        return self._model.predict_proba(self._transform(X))

    @property
    def coef_(self) -> np.ndarray:
        """Return model coefficients."""
        self._check_fitted()
        return self._model.coef_

    def __repr__(self) -> str:
        if self.task == "classification":
            return f"LinearModel(task='classification', C={self.C}, penalty='{self.penalty}')"
        return f"LinearModel(task='regression', fit_intercept={self.fit_intercept})"
