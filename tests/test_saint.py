"""Tests for SAINTModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import SAINTModel

@pytest.fixture
def clf_data():
    return load_iris(return_X_y=True)

@pytest.fixture
def reg_data():
    # Only use a few rows for SAINT test to make it fast
    X, y = load_diabetes(return_X_y=True)
    return X[:100], y[:100]

class TestSAINTClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = SAINTModel(task="classification", epochs=2, batch_size=16, embed_dim=16, num_heads=4, num_layers=1)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = SAINTModel(task="classification", epochs=2, batch_size=16, embed_dim=16, num_heads=4, num_layers=1)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

class TestSAINTRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = SAINTModel(task="regression", epochs=2, batch_size=16, embed_dim=16, num_heads=4, num_layers=1)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
