"""Tests for LinearModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import LinearModel


@pytest.fixture
def clf_data():
    return load_iris(return_X_y=True)


@pytest.fixture
def reg_data():
    return load_diabetes(return_X_y=True)


# ── Classification ──

class TestLinearClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = LinearModel(task="classification")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = LinearModel(task="classification")
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_proba_sums_to_one(self, clf_data):
        X, y = clf_data
        m = LinearModel(task="classification")
        m.fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_accuracy_above_chance(self, clf_data):
        X, y = clf_data
        m = LinearModel(task="classification")
        m.fit(X, y)
        acc = (m.predict(X) == y).mean()
        assert acc > 0.80, f"Accuracy too low: {acc}"

    def test_auto_task_detection(self):
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        m = LinearModel(task="auto")
        m.fit(X, y)
        assert m.task == "classification"

    def test_coef_attribute(self, clf_data):
        X, y = clf_data
        m = LinearModel(task="classification")
        m.fit(X, y)
        assert m.coef_.shape[1] == X.shape[1]

    def test_unfitted_raises(self):
        m = LinearModel(task="classification")
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 4)))

    def test_no_scaling(self, clf_data):
        X, y = clf_data
        m = LinearModel(task="classification", scale=False)
        m.fit(X, y)
        assert m._scaler is None
        preds = m.predict(X)
        assert preds.shape == (len(y),)


# ── Regression ──

class TestLinearRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = LinearModel(task="regression")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_proba_raises(self, reg_data):
        X, y = reg_data
        m = LinearModel(task="regression")
        m.fit(X, y)
        with pytest.raises(NotImplementedError):
            m.predict_proba(X)

    def test_auto_task_detection(self):
        X = np.random.rand(200, 4)
        y = np.random.rand(200) * 100
        m = LinearModel(task="auto")
        m.fit(X, y)
        assert m.task == "regression"
