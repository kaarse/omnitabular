import pytest
import numpy as np
import pandas as pd
from omnitabular import OmniTabular

def test_auto_preprocessing():
    # Create dataset with NaNs and strings
    X = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat1": ["a", "b", "a", "c", np.nan],
        "num2": [10, 20, 30, 40, np.nan]
    })
    y = np.array([0, 1, 0, 1, 0])
    
    # Init OmniTabular with auto_preprocess=True (which is default)
    model = OmniTabular(
        task="classification",
        auto_preprocess=True,
        cv_folds=2,
        verbose=False
    )
    
    # Temporarily override registry to ONLY use rf and lgb for fast test
    model._build_model_defs_orig = model._build_model_defs
    def mock_build():
        defs = model._build_model_defs_orig()
        return {k: v for k, v in defs.items() if k in ["rf", "lgb"]}
    model._build_model_defs = mock_build
    
    model.fit(X, y)
    assert model._fitted
    
def test_global_explain():
    np.random.seed(42)
    X = np.random.rand(20, 3)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] 
    
    model = OmniTabular(
        task="regression",
        cv_folds=2,
        verbose=False
    )
    
    model._build_model_defs_orig = model._build_model_defs
    def mock_build():
        defs = model._build_model_defs_orig()
        return {k: v for k, v in defs.items() if k == "rf"}
    model._build_model_defs = mock_build
    
    model.fit(X, y)
    explanation = model.explain(X, y, n_repeats=2)
    assert "feature_index" in explanation.columns

def test_custom_callable_scorer():
    np.random.seed(42)
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
    
    # Define a completely custom business logic loss
    # E.g. "False Negatives cost $500, False Positives cost $10"
    def business_profit_scorer(y_true, y_pred):
        profit = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                profit += 100
            elif yt == 1 and yp == 0:
                profit -= 500
            elif yt == 0 and yp == 1:
                profit -= 10
            elif yt == 0 and yp == 0:
                profit += 0
        return float(profit)
        
    model = OmniTabular(
        task="classification",
        metric=business_profit_scorer,
        metric_higher_is_better=True,
        metric_needs_proba=False,
        cv_folds=2,
        verbose=False
    )
    
    model._build_model_defs_orig = model._build_model_defs
    def mock_build():
        defs = model._build_model_defs_orig()
        return {k: v for k, v in defs.items() if k == "rf"}
    model._build_model_defs = mock_build

    model.fit(X, y)
    assert model._fitted
    assert model.ensemble_oof_score_ is not None
