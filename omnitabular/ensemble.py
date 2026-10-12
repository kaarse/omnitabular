"""Weighted ensemble that combines model predictions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseModel


class WeightedEnsemble:
    """
    Combines predictions from a list of fitted base models using fixed weights.

    For **classification**, predictions are blended as a weighted sum of
    ``predict_proba`` outputs.  For **regression**, a weighted sum of
    ``predict`` outputs is used.

    Parameters
    ----------
    models : list of BaseModel
        Fitted base learners.
    weights : array-like of shape (n_models,)
        Blending weights (need not sum to 1 — they will be normalised).
    task : {"classification", "regression"}

    Attributes
    ----------
    weights_ : np.ndarray
        Normalised weight vector.

    Examples
    --------
    >>> from omnitabular import KNNModel, LGBModel, WeightedEnsemble
    >>> import numpy as np
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 3, 100)
    >>> knn = KNNModel(task="classification").fit(X, y)
    >>> lgb = LGBModel(task="classification").fit(X, y)
    >>> ens = WeightedEnsemble([knn, lgb], weights=[0.4, 0.6], task="classification")
    >>> ens.predict_proba(X[:5])
    """

    def __init__(
        self,
        models: List[BaseModel],
        weights: np.ndarray,
        task: str,
    ) -> None:
        if len(models) != len(weights):
            raise ValueError(
                f"Number of models ({len(models)}) must match number of weights ({len(weights)})."
            )
        self.models = models
        self.task = task

        w = np.asarray(weights, dtype=float)
        w = np.clip(w, 0, None)
        total = w.sum()
        if total == 0:
            raise ValueError("All weights are zero — cannot normalise.")
        self.weights_ = w / total

    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of predict_proba outputs (classification only)."""
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification.")
        probas = [m.predict_proba(X) for m in self.models]
        blended = sum(self.weights_[i] * probas[i] for i in range(len(self.models)))
        return blended

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        For classification: argmax over blended probabilities.
        For regression: weighted average of predictions.
        """
        if self.task == "classification":
            proba = self.predict_proba(X)
            if hasattr(self.models[0], "classes_"):
                return self.models[0].classes_[np.argmax(proba, axis=1)]
            # Recover class labels from first model (which must be fitted)
            if hasattr(getattr(self.models[0], "_model", None), "classes_"):
                classes = self.models[0]._model.classes_
                return classes[np.argmax(proba, axis=1)]
            return np.argmax(proba, axis=1)
        else:
            preds = [m.predict(X) for m in self.models]
            return sum(self.weights_[i] * preds[i] for i in range(len(self.models)))

    def __repr__(self) -> str:
        model_names = [m.__class__.__name__ for m in self.models]
        w_str = ", ".join(f"{w:.3f}" for w in self.weights_)
        return f"WeightedEnsemble(models={model_names}, weights=[{w_str}], task='{self.task}')"
