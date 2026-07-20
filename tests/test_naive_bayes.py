"""Tests for NaiveBayesModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from omnitabular import NaiveBayesModel


@pytest.fixture
def clf_data():
    return load_iris(return_X_y=True)


class TestNaiveBayesGaussian:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = NaiveBayesModel(task="classification", variant="gaussian")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = NaiveBayesModel(variant="gaussian")
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_proba_sums_to_one(self, clf_data):
        X, y = clf_data
        m = NaiveBayesModel(variant="gaussian")
        m.fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_accuracy_above_chance(self, clf_data):
        X, y = clf_data
        m = NaiveBayesModel(variant="gaussian")
        m.fit(X, y)
        acc = (m.predict(X) == y).mean()
        assert acc > 0.85, f"Accuracy too low: {acc}"

    def test_auto_task_detection(self):
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        m = NaiveBayesModel(task="auto")
        m.fit(X, y)
        assert m.task == "classification"

    def test_class_prior(self, clf_data):
        X, y = clf_data
        m = NaiveBayesModel(variant="gaussian")
        m.fit(X, y)
        assert m.class_prior_.shape == (3,)

    def test_unfitted_raises(self):
        m = NaiveBayesModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 4)))


class TestNaiveBayesBernoulli:
    def test_fit_predict(self, clf_data):
        X, y = clf_data
        m = NaiveBayesModel(variant="bernoulli")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)


class TestNaiveBayesMultinomial:
    def test_fit_predict(self):
        """Multinomial NB requires non-negative features."""
        rng = np.random.default_rng(0)
        X = rng.integers(0, 10, size=(200, 6)).astype(float)
        y = rng.integers(0, 2, 200)
        m = NaiveBayesModel(variant="multinomial")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (200,)


class TestNaiveBayesErrors:
    def test_regression_raises(self):
        X = np.random.rand(100, 4)
        y = np.random.rand(100) * 100
        m = NaiveBayesModel(task="regression")
        with pytest.raises(ValueError, match="classification-only"):
            m.fit(X, y)

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            NaiveBayesModel(variant="invalid")
