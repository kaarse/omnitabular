"""Tests for MLPModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import MLPModel

@pytest.fixture
def clf_data():
    return load_iris(return_X_y=True)

@pytest.fixture
def reg_data():
    return load_diabetes(return_X_y=True)

class TestMLPClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = MLPModel(task="classification", epochs=2, batch_size=16)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = MLPModel(task="classification", epochs=2, batch_size=16)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_proba_sums_to_one(self, clf_data):
        X, y = clf_data
        m = MLPModel(task="classification", epochs=2, batch_size=16)
        m.fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

class TestMLPRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = MLPModel(task="regression", epochs=2, batch_size=16)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_proba_raises(self, reg_data):
        X, y = reg_data
        m = MLPModel(task="regression", epochs=2, batch_size=16)
        m.fit(X, y)
        with pytest.raises(NotImplementedError):
            m.predict_proba(X)
