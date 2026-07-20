"""Tests for FTTransformerModel."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from omnitabular import FTTransformerModel

@pytest.fixture
def clf_data():
    return load_iris(return_X_y=True)

@pytest.fixture
def reg_data():
    return load_diabetes(return_X_y=True)

class TestFTTransformerClassification:
    def test_fit_predict_shape(self, clf_data):
        X, y = clf_data
        m = FTTransformerModel(task="classification", epochs=2, batch_size=16, embed_dim=16, num_heads=2, num_layers=1)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = FTTransformerModel(task="classification", epochs=2, batch_size=16, embed_dim=16, num_heads=2, num_layers=1)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

class TestFTTransformerRegression:
    def test_fit_predict_shape(self, reg_data):
        X, y = reg_data
        m = FTTransformerModel(task="regression", epochs=2, batch_size=16, embed_dim=16, num_heads=2, num_layers=1)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
