"""Tests for TabFMModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import TabFMModel

@pytest.fixture
def clf_data():
    return load_iris(return_X_y=True)

@pytest.fixture
def reg_data():
    return load_diabetes(return_X_y=True)

class TestTabFMClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = TabFMModel(task="classification", epochs=2, batch_size=16, embed_dim=4, hidden_dims=[16])
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = TabFMModel(task="classification", epochs=2, batch_size=16, embed_dim=4, hidden_dims=[16])
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

class TestTabFMRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = TabFMModel(task="regression", epochs=2, batch_size=16, embed_dim=4, hidden_dims=[16])
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
