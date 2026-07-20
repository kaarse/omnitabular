"""Hill Climbing ensemble weight optimizer for OmniTabular."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .utils import evaluate, get_default_metric, METRIC_REGISTRY


class HillClimbingEnsemble:
    """
    Stochastic Hill Climbing optimiser for ensemble blending weights.

    Given a collection of OOF predictions from multiple base models, this
    optimiser searches for a weight vector **w** (summing to 1) that maximises
    (or minimises, depending on the metric) the blended prediction score.

    Algorithm
    ---------
    1. Start from a uniform weight vector ``w = [1/K, ..., 1/K]``.
    2. At each step, sample a random perturbation ``Δw`` and compute
       ``w' = normalise(w + step_size * Δw)``.
    3. Accept ``w'`` if it improves the objective.
    4. Gradually shrink ``step_size`` (simulated annealing schedule).
    5. Repeat for ``n_iterations`` steps.

    Parameters
    ----------
    metric : str or None
        Metric name from ``METRIC_REGISTRY``. If None, defaults to
        ``"log_loss"`` (classification) or ``"rmse"`` (regression).
    n_iterations : int
        Number of hill climbing iterations.
    step_size : float
        Initial perturbation magnitude.
    step_decay : float
        Multiplicative decay applied to ``step_size`` every ``decay_every`` steps.
    decay_every : int
        How often (steps) to apply step decay.
    random_state : int or None
    verbose : bool
        Print progress if True.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_models,)
        Optimal blending weights found after ``fit``.
    best_score_ : float
        Best metric value achieved.
    score_history_ : list of float
        Score at each accepted improvement.

    Examples
    --------
    >>> import numpy as np
    >>> from omnitabular import HillClimbingEnsemble
    >>> rng = np.random.default_rng(0)
    >>> y_true = rng.integers(0, 2, 200)
    >>> # Simulate OOF probabilities from 3 models
    >>> oof_probas = [rng.dirichlet([1, 1], size=200) for _ in range(3)]
    >>> hc = HillClimbingEnsemble(metric="log_loss", n_iterations=500)
    >>> hc.fit(y_true, oof_preds=None, oof_probas=oof_probas, task="classification")
    >>> hc.weights_
    """

    def __init__(
        self,
        metric: Optional[Union[str, Callable]] = None,
        metric_higher_is_better: bool = False,
        metric_needs_proba: bool = False,
        n_iterations: int = 3000,
        step_size: float = 0.1,
        step_decay: float = 0.99,
        decay_every: int = 50,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ) -> None:
        self.metric = metric
        self.metric_higher_is_better = metric_higher_is_better
        self.metric_needs_proba = metric_needs_proba
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.step_decay = step_decay
        self.decay_every = decay_every
        self.random_state = random_state
        self.verbose = verbose

        self.weights_: Optional[np.ndarray] = None
        self.best_score_: Optional[float] = None
        self.score_history_: List[float] = []

    # ------------------------------------------------------------------

    def fit(
        self,
        y_true: np.ndarray,
        task: str,
        oof_preds: Optional[List[np.ndarray]] = None,
        oof_probas: Optional[List[np.ndarray]] = None,
    ) -> "HillClimbingEnsemble":
        """
        Find optimal blending weights.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
        task : {"classification", "regression"}
        oof_preds : list of np.ndarray of shape (n_samples,), optional
            Hard OOF predictions per model (used for regression or hard-vote blending).
        oof_probas : list of np.ndarray of shape (n_samples, n_classes), optional
            Soft OOF probabilities per model (classification only).
        """
        y_true = np.asarray(y_true)

        # Choose metric
        if callable(self.metric):
            metric_name = self.metric
            higher_is_better = self.metric_higher_is_better
            needs_proba = self.metric_needs_proba
        else:
            metric_name = self.metric or get_default_metric(task)
            if metric_name not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. Available: {list(METRIC_REGISTRY.keys())}"
                )
            _, higher_is_better, needs_proba = METRIC_REGISTRY[metric_name]

        # Validate inputs
        if needs_proba and (oof_probas is None or len(oof_probas) == 0):
            raise ValueError(
                f"Metric '{metric_name}' requires OOF probability predictions (oof_probas)."
            )
        if not needs_proba and (oof_preds is None or len(oof_preds) == 0):
            raise ValueError(
                f"Metric '{metric_name}' requires hard OOF predictions (oof_preds)."
            )

        predictions = oof_probas if needs_proba else oof_preds
        K = len(predictions)

        rng = np.random.default_rng(self.random_state)

        # Initialise uniform weights
        w = np.ones(K) / K
        step = self.step_size

        def _score(weights: np.ndarray) -> float:
            blended = sum(weights[i] * predictions[i] for i in range(K))
            if needs_proba:
                # Clip to [0, 1] and renormalise rows to avoid FP overflow
                blended = np.clip(blended, 0, 1)
                row_sums = blended.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1, row_sums)
                blended = blended / row_sums
                return evaluate(metric_name, y_true, None, proba=blended, higher_is_better=higher_is_better, needs_proba=needs_proba)
            return evaluate(metric_name, y_true, blended, higher_is_better=higher_is_better, needs_proba=needs_proba)

        best_score = _score(w)
        self.score_history_ = [best_score]

        if self.verbose:
            print(f"[HillClimbing] Start | metric={metric_name} | score={best_score:.6f}")

        for iteration in range(1, self.n_iterations + 1):
            # Decay step size
            if iteration % self.decay_every == 0:
                step *= self.step_decay

            # Perturb: add random delta and re-normalise to simplex
            delta = rng.standard_normal(K)
            w_new = w + step * delta
            w_new = np.clip(w_new, 0, None)          # non-negative
            w_sum = w_new.sum()
            if w_sum == 0:
                continue
            w_new /= w_sum                            # sum to 1

            candidate_score = _score(w_new)

            improved = (
                candidate_score > best_score if higher_is_better
                else candidate_score < best_score
            )
            if improved:
                w = w_new
                best_score = candidate_score
                self.score_history_.append(best_score)
                if self.verbose:
                    print(
                        f"[HillClimbing] iter={iteration:>5} | "
                        f"score={best_score:.6f} | step={step:.5f}"
                    )

        self.weights_ = w
        self.best_score_ = best_score

        if self.verbose:
            print(
                f"[HillClimbing] Done  | best_score={best_score:.6f} | "
                f"weights={np.round(w, 4)}"
            )

        return self

    # ------------------------------------------------------------------

    def blend_predictions(
        self,
        predictions: List[np.ndarray],
    ) -> np.ndarray:
        """
        Apply ``weights_`` to a list of prediction arrays.

        Parameters
        ----------
        predictions : list of np.ndarray
            Same order as during ``fit``.

        Returns
        -------
        blended : np.ndarray
        """
        if self.weights_ is None:
            raise RuntimeError("HillClimbingEnsemble is not fitted yet.")
        if len(predictions) != len(self.weights_):
            raise ValueError(
                f"Expected {len(self.weights_)} prediction arrays, got {len(predictions)}."
            )
        blended = sum(self.weights_[i] * predictions[i] for i in range(len(predictions)))
        # Clip and renormalise probability arrays to avoid FP overflow
        if blended.ndim == 2:
            blended = np.clip(blended, 0, 1)
            row_sums = blended.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            blended = blended / row_sums
        return blended

    def __repr__(self) -> str:
        return (
            f"HillClimbingEnsemble(metric='{self.metric}', "
            f"n_iterations={self.n_iterations}, step_size={self.step_size})"
        )
