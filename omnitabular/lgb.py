"""LightGBM wrapper for OmniTabular."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError as exc:
    raise ImportError(
        "LightGBM is required for LGBModel. Install it with: pip install lightgbm"
    ) from exc

from .base import BaseModel
from .utils import detect_task


class LGBModel(BaseModel):
    """
    LightGBM model with automatic task detection.

    Wraps :class:`lightgbm.LGBMClassifier` or :class:`lightgbm.LGBMRegressor`.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
    num_leaves : int
        Max number of leaves per tree.
    max_depth : int
        -1 for unlimited.
    subsample : float
        Row subsampling fraction.
    colsample_bytree : float
        Feature subsampling fraction.
    min_child_samples : int
    reg_alpha : float
        L1 regularisation.
    reg_lambda : float
        L2 regularisation.
    random_state : int
    n_jobs : int
    categorical_features : list of int or str, optional
        Column indices / names of categorical features.
    early_stopping_rounds : int or None
        Enable early stopping when validation data is provided via fit_params.
    verbose : int
        LightGBM verbosity (-1 = silent).
    **kwargs
        Extra parameters forwarded to the underlying LGBM estimator.

    Examples
    --------
    >>> from omnitabular import LGBModel
    >>> import numpy as np
    >>> X = np.random.rand(200, 8)
    >>> y = np.random.rand(200)
    >>> m = LGBModel(task="regression", n_estimators=100)
    >>> m.fit(X, y)
    >>> m.predict(X[:5])
    """

    def __init__(
        self,
        task: str = "auto",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = -1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        categorical_features: Optional[List] = None,
        early_stopping_rounds: Optional[int] = 50,
        verbose: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(task=task)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.categorical_features = categorical_features or "auto"
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self._extra_kwargs = kwargs
        self._model = None

    # ------------------------------------------------------------------

    def _build_model(self, task: str):
        shared = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **self._extra_kwargs,
        )
        if task == "classification":
            return LGBMClassifier(**shared)
        return LGBMRegressor(**shared)

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "LGBModel":
        X = self._to_numpy(X)
        y = np.asarray(y)

        if self.task == "auto":
            self.task = detect_task(y)

        self._model = self._build_model(self.task)

        # Build kwargs for fit
        lgb_fit_params = dict(fit_params)

        # Handle early stopping
        if self.early_stopping_rounds and "eval_set" in lgb_fit_params:
            lgb_fit_params.setdefault(
                "callbacks",
                [lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                 lgb.log_evaluation(period=-1)],
            )
        else:
            lgb_fit_params.setdefault("callbacks", [lgb.log_evaluation(period=-1)])

        # Categorical feature handling
        if isinstance(self.categorical_features, list):
            lgb_fit_params["categorical_feature"] = self.categorical_features

        self._model.fit(X, y, **lgb_fit_params)
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

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances (gain-based) from the underlying LGB model."""
        self._check_fitted()
        return self._model.feature_importances_

    def __repr__(self) -> str:
        return (
            f"LGBModel(task='{self.task}', n_estimators={self.n_estimators}, "
            f"lr={self.learning_rate}, num_leaves={self.num_leaves})"
        )
