"""Tests for HillClimbingEnsemble."""

import numpy as np
import pytest

from omnitabular import HillClimbingEnsemble


def _make_binary_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # Build a "good" model: high probability on the true class
    base_conf = 0.80
    noise_scale = 0.10
    p0 = np.where(y == 0, base_conf + rng.uniform(0, noise_scale, n),
                           (1 - base_conf) - rng.uniform(0, noise_scale, n))
    p0 = np.clip(p0, 1e-6, 1 - 1e-6)
    proba_good = np.column_stack([p0, 1 - p0])

    # A "bad" model: near-random
    proba_bad = rng.dirichlet([1, 1], size=n)
    proba_avg = 0.5 * proba_good + 0.5 * proba_bad
    proba_avg /= proba_avg.sum(axis=1, keepdims=True)
    return y, [proba_good, proba_bad, proba_avg]


def _make_regression_data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n) * 10
    pred_good = y + rng.standard_normal(n) * 0.5
    pred_bad  = rng.standard_normal(n) * 10
    return y, [pred_good, pred_bad]


class TestHillClimbingClassification:
    def test_weights_sum_to_one(self):
        y, probas = _make_binary_data()
        hc = HillClimbingEnsemble(metric="log_loss", n_iterations=200, random_state=0)
        hc.fit(y_true=y, task="classification", oof_probas=probas)
        np.testing.assert_allclose(hc.weights_.sum(), 1.0, atol=1e-8)

    def test_weights_are_nonneg(self):
        y, probas = _make_binary_data()
        hc = HillClimbingEnsemble(metric="log_loss", n_iterations=200, random_state=0)
        hc.fit(y_true=y, task="classification", oof_probas=probas)
        assert (hc.weights_ >= 0).all()

    def test_weights_shape(self):
        y, probas = _make_binary_data()
        hc = HillClimbingEnsemble(metric="log_loss", n_iterations=200, random_state=0)
        hc.fit(y_true=y, task="classification", oof_probas=probas)
        assert hc.weights_.shape == (3,)

    def test_favours_good_model(self):
        """
        The good model (index 0) should receive the highest weight.
        """
        y, probas = _make_binary_data()
        hc = HillClimbingEnsemble(metric="log_loss", n_iterations=1000, random_state=42)
        hc.fit(y_true=y, task="classification", oof_probas=probas)
        assert hc.weights_[0] > hc.weights_[1], (
            f"Expected w[0]={hc.weights_[0]:.4f} > w[1]={hc.weights_[1]:.4f}"
        )

    def test_blend_predictions_shape(self):
        y, probas = _make_binary_data()
        hc = HillClimbingEnsemble(n_iterations=100, random_state=0)
        hc.fit(y_true=y, task="classification", oof_probas=probas)
        blended = hc.blend_predictions(probas)
        assert blended.shape == (len(y), 2)

    def test_score_history_monotone(self):
        """Accepted scores must be improving (lower log_loss)."""
        y, probas = _make_binary_data()
        hc = HillClimbingEnsemble(metric="log_loss", n_iterations=500, random_state=0)
        hc.fit(y_true=y, task="classification", oof_probas=probas)
        history = hc.score_history_
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-10, (
                f"Score not monotone at step {i}: {history[i-1]:.6f} -> {history[i]:.6f}"
            )

    def test_not_fitted_raises(self):
        hc = HillClimbingEnsemble()
        y, probas = _make_binary_data()
        with pytest.raises(RuntimeError):
            hc.blend_predictions(probas)

    def test_wrong_metric_raises(self):
        y, probas = _make_binary_data()
        hc = HillClimbingEnsemble(metric="nonexistent_metric", n_iterations=10)
        with pytest.raises(ValueError):
            hc.fit(y_true=y, task="classification", oof_probas=probas)

    def test_missing_probas_raises(self):
        y, _ = _make_binary_data()
        hc = HillClimbingEnsemble(metric="log_loss", n_iterations=10)
        with pytest.raises(ValueError):
            hc.fit(y_true=y, task="classification", oof_preds=None, oof_probas=None)


class TestHillClimbingRegression:
    def test_weights_sum_to_one(self):
        y, preds = _make_regression_data()
        hc = HillClimbingEnsemble(metric="rmse", n_iterations=300, random_state=0)
        hc.fit(y_true=y, task="regression", oof_preds=preds)
        np.testing.assert_allclose(hc.weights_.sum(), 1.0, atol=1e-8)

    def test_favours_good_model(self):
        y, preds = _make_regression_data()
        hc = HillClimbingEnsemble(metric="rmse", n_iterations=1000, random_state=99)
        hc.fit(y_true=y, task="regression", oof_preds=preds)
        assert hc.weights_[0] > hc.weights_[1], (
            f"Expected w[0]={hc.weights_[0]:.4f} > w[1]={hc.weights_[1]:.4f}"
        )
