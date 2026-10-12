"""Abstract base model interface for OmniTabular."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Uniform interface for all OmniTabular base learners.

    Every concrete model must honour:
        fit(X, y, **fit_params)
        predict(X) -> np.ndarray
        predict_proba(X) -> np.ndarray  (classification only)
    """

    def __init__(self, task: str = "auto") -> None:
        """
        Parameters
        ----------
        task : {"auto", "classification", "regression"}
            If "auto", task type is inferred on first ``fit``.
        """
        if task not in ("auto", "classification", "regression"):
            raise ValueError(f"Unknown task '{task}'. Choose from 'auto', 'classification', 'regression'.")
        self.task = task
        self._fitted = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "BaseModel":
        """Fit the model on (X, y)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard predictions."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates (classification only)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_proba.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet. Call fit() first.")

    def _prepare_target(self, y):
        from .utils import detect_task
        y = np.asarray(y)
        if self.task == "auto":
            self.task = detect_task(y)
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
        return y

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task='{self.task}')"
