"""Naive Bayes wrapper for OmniTabular."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from .base import BaseModel
from .utils import detect_task


class NaiveBayesModel(BaseModel):
    """
    Naive Bayes classifier for OmniTabular.

    Supports three variants via the ``variant`` parameter:

    * ``"gaussian"`` → :class:`sklearn.naive_bayes.GaussianNB`
      (default, works with continuous features)
    * ``"multinomial"`` → :class:`sklearn.naive_bayes.MultinomialNB`
      (for count / TF-IDF features — requires non-negative X)
    * ``"bernoulli"`` → :class:`sklearn.naive_bayes.BernoulliNB`
      (for binary / boolean features)

    .. note::
       Naive Bayes is a **classification-only** model.  If ``task`` is set to
       ``"regression"`` an error is raised at fit time.

    Parameters
    ----------
    task : {"auto", "classification"}
    variant : {"gaussian", "multinomial", "bernoulli"}
    var_smoothing : float
        Portion of the largest variance added to variances for stability
        (GaussianNB only, default 1e-9).
    alpha : float
        Additive (Laplace) smoothing parameter (Multinomial / Bernoulli only,
        default 1.0).
    **kwargs
        Additional kwargs forwarded to the chosen sklearn estimator.

    Examples
    --------
    >>> from omnitabular import NaiveBayesModel
    >>> import numpy as np
    >>> X = np.random.rand(200, 4)
    >>> y = (X[:, 0] > 0.5).astype(int)
    >>> m = NaiveBayesModel(variant="gaussian")
    >>> m.fit(X, y)
    >>> m.predict_proba(X[:5])
    """

    VARIANTS = {"gaussian", "multinomial", "bernoulli"}

    def __init__(
        self,
        task: str = "auto",
        variant: str = "gaussian",
        var_smoothing: float = 1e-9,
        alpha: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(task=task)
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {sorted(self.VARIANTS)}."
            )
        self.variant = variant
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self._extra_kwargs = kwargs
        self._model = None

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "NaiveBayesModel":
        X = self._to_numpy(X)
        y = np.asarray(y)

        if self.task == "auto":
            self.task = detect_task(y)

        if self.task == "regression":
            raise ValueError(
                "NaiveBayesModel is a classification-only model. "
                "Set task='classification' or use a different model for regression."
            )

        if self.variant == "gaussian":
            self._model = GaussianNB(
                var_smoothing=self.var_smoothing,
                **self._extra_kwargs,
            )
        elif self.variant == "multinomial":
            self._model = MultinomialNB(
                alpha=self.alpha,
                **self._extra_kwargs,
            )
        else:  # bernoulli
            self._model = BernoulliNB(
                alpha=self.alpha,
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
    def class_prior_(self) -> np.ndarray:
        """Prior probabilities for each class."""
        self._check_fitted()
        return self._model.class_prior_

    def __repr__(self) -> str:
        return (
            f"NaiveBayesModel(task='{self.task}', variant='{self.variant}')"
        )
