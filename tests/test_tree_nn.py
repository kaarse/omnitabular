"""Tests for TreeNNModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import TreeNNModel

@pytest.fixture
def clf_data():
    return load_iris(return_X_y=True)

@pytest.fixture
def reg_data():
    return load_diabetes(return_X_y=True)

class TestTreeNNClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = TreeNNModel(task="classification", epochs=2, batch_size=16, depth=2)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = TreeNNModel(task="classification", epochs=2, batch_size=16, depth=2)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

class TestTreeNNRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = TreeNNModel(task="regression", epochs=2, batch_size=16, depth=2)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
