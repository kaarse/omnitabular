"""Tests for SVMModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import SVMModel


@pytest.fixture
def clf_data():
    return load_iris(return_X_y=True)


@pytest.fixture
def reg_data():
    X, y = load_diabetes(return_X_y=True)
    return X[:100], y[:100]  # smaller subset for SVM speed


# ── Classification ──

class TestSVMClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = SVMModel(task="classification", kernel="rbf")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = SVMModel(task="classification")
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_proba_sums_to_one(self, clf_data):
        X, y = clf_data
        m = SVMModel(task="classification")
        m.fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_accuracy_above_chance(self, clf_data):
        X, y = clf_data
        m = SVMModel(task="classification")
        m.fit(X, y)
        acc = (m.predict(X) == y).mean()
        assert acc > 0.90, f"Accuracy too low: {acc}"

    def test_linear_kernel(self, clf_data):
        X, y = clf_data
        m = SVMModel(task="classification", kernel="linear")
        m.fit(X, y)
        assert m.predict(X).shape == (len(y),)

    def test_auto_task_detection(self):
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        m = SVMModel(task="auto")
        m.fit(X, y)
        assert m.task == "classification"

    def test_unfitted_raises(self):
        m = SVMModel(task="classification")
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 4)))


# ── Regression ──

class TestSVMRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = SVMModel(task="regression")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_proba_raises(self, reg_data):
        X, y = reg_data
        m = SVMModel(task="regression")
        m.fit(X, y)
        with pytest.raises(NotImplementedError):
            m.predict_proba(X)
