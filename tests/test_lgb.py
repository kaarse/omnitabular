"""Tests for LGBModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import LGBModel


@pytest.fixture
def clf_data():
    X, y = load_iris(return_X_y=True)
    return X, y


@pytest.fixture
def reg_data():
    X, y = load_diabetes(return_X_y=True)
    return X, y


class TestLGBClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = LGBModel(task="classification", n_estimators=50)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = LGBModel(task="classification", n_estimators=50)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_proba_sums_to_one(self, clf_data):
        X, y = clf_data
        m = LGBModel(task="classification", n_estimators=50)
        m.fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_accuracy_above_chance(self, clf_data):
        X, y = clf_data
        m = LGBModel(task="classification", n_estimators=100)
        m.fit(X, y)
        acc = (m.predict(X) == y).mean()
        assert acc > 0.90, f"Accuracy too low: {acc}"

    def test_auto_task_detection(self):
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        m = LGBModel(task="auto", n_estimators=20)
        m.fit(X, y)
        assert m.task == "classification"

    def test_feature_importances(self, clf_data):
        X, y = clf_data
        m = LGBModel(task="classification", n_estimators=50)
        m.fit(X, y)
        imp = m.feature_importances_
        assert imp.shape == (X.shape[1],)
        assert imp.sum() > 0


class TestLGBRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = LGBModel(task="regression", n_estimators=50)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_proba_raises_for_regression(self, reg_data):
        X, y = reg_data
        m = LGBModel(task="regression", n_estimators=20)
        m.fit(X, y)
        with pytest.raises(NotImplementedError):
            m.predict_proba(X)

    def test_unfitted_raises(self):
        m = LGBModel(task="regression")
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 4)))
