"""Tests for KNNModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import KNNModel


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture
def reg_data():
    X, y = load_diabetes(return_X_y=True)
    return X, y


# -------------------------------------------------------------------------
# Classification
# -------------------------------------------------------------------------

class TestKNNClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = KNNModel(task="classification")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = KNNModel(task="classification")
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_proba_sums_to_one(self, clf_data):
        X, y = clf_data
        m = KNNModel(task="classification")
        m.fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_auto_task_detection(self):
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        m = KNNModel(task="auto")
        m.fit(X, y)
        assert m.task == "classification"

    def test_accuracy_above_chance(self, clf_data):
        X, y = clf_data
        m = KNNModel(task="classification", n_neighbors=5)
        m.fit(X, y)
        acc = (m.predict(X) == y).mean()
        assert acc > 0.85, f"Accuracy too low: {acc}"

    def test_unfitted_raises(self):
        m = KNNModel(task="classification")
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 4)))

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            KNNModel(task="bad_task")


# -------------------------------------------------------------------------
# Regression
# -------------------------------------------------------------------------

class TestKNNRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = KNNModel(task="regression")
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_proba_raises_for_regression(self, reg_data):
        X, y = reg_data
        m = KNNModel(task="regression")
        m.fit(X, y)
        with pytest.raises(NotImplementedError):
            m.predict_proba(X)

    def test_auto_task_detection_regression(self):
        X = np.random.rand(200, 4)
        y = np.random.rand(200) * 100
        m = KNNModel(task="auto")
        m.fit(X, y)
        assert m.task == "regression"
