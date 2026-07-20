"""SVM wrapper for OmniTabular."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .utils import detect_task


class SVMModel(BaseModel):
    """
    Support Vector Machine model with automatic task detection.

    * **Classification** → ``sklearn.svm.SVC`` (with ``probability=True``).
    * **Regression** → ``sklearn.svm.SVR``.

    Features are automatically standard-scaled before fitting (SVM is
    sensitive to feature scale).

    Parameters
    ----------
    task : {"auto", "classification", "regression"}
    C : float
        Regularisation parameter (default 1.0).
    kernel : {"linear", "poly", "rbf", "sigmoid"}
        Kernel function (default "rbf").
    gamma : {"scale", "auto"} or float
        Kernel coefficient.
    degree : int
        Degree for poly kernel.
    max_iter : int
        Hard limit on iterations (-1 = no limit).
    random_state : int or None
    scale : bool
        Whether to standard-scale features.
    **kwargs
        Extra kwargs forwarded to the underlying sklearn estimator.

    Examples
    --------
    >>> from omnitabular import SVMModel
    >>> import numpy as np
    >>> X = np.random.rand(200, 4)
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> m = SVMModel(task="classification", kernel="rbf")
    >>> m.fit(X, y)
    >>> m.predict_proba(X[:5])
    """

    def __init__(
        self,
        task: str = "auto",
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        degree: int = 3,
        max_iter: int = -1,
        random_state: Optional[int] = 42,
        scale: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(task=task)
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.random_state = random_state
        self.scale = scale
        self._extra_kwargs = kwargs
        self._model = None
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "SVMModel":
        X = self._to_numpy(X)
        y = np.asarray(y)

        if self.task == "auto":
            self.task = detect_task(y)

        if self.scale:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        if self.task == "classification":
            base_svc = SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                max_iter=self.max_iter,
                random_state=self.random_state,
                **self._extra_kwargs,
            )
            self._model = CalibratedClassifierCV(
                base_svc, ensemble=False,
            )
        else:
            self._model = SVR(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                max_iter=self.max_iter,
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

    def __repr__(self) -> str:
        return (
            f"SVMModel(task='{self.task}', C={self.C}, "
            f"kernel='{self.kernel}', gamma='{self.gamma}')"
        )
