import pytest
import numpy as np
import pandas as pd
from omnitabular import OmniTabular

def test_parallel_time_series():
    """Test that time_series CV runs cleanly and parallel backend doesn't crash."""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    
    # We will only use LGB and XGB for a fast integration test
    model = OmniTabular(
        task="classification",
        cv_strategy="time_series",
        cv_folds=3,
        n_jobs=2,  # test parallel workers > 1
        verbose=True
    )
    
    # Mock to speed up
    model._build_model_defs_orig = model._build_model_defs
    def mock_build():
        defs = model._build_model_defs_orig()
        return {k: v for k, v in defs.items() if k in ["xgb", "lgb"]}
    model._build_model_defs = mock_build
    
    # Run fit
    # Because it is time_series, groups isn't needed.
    res = model.fit(X, y)
    
    assert res._fitted
    # Assure weights reflect the 2 generated models
    assert len(res.weights_) == 2
    assert res.ensemble_oof_score_ is not None

def test_group_kfold():
    """Test GroupKFold forces requirement of 'groups' array."""
    np.random.seed(42)
    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, 20)
    # 5 unique groups
    groups = np.repeat([0, 1, 2, 3, 4], 4)
    
    model = OmniTabular(
        task="classification",
        cv_strategy="group_kfold",
        cv_folds=2,
        n_jobs=-1,
        verbose=False
    )
    
    # Mock to speed up
    model._build_model_defs_orig = model._build_model_defs
    def mock_build():
        defs = model._build_model_defs_orig()
        return {k: v for k, v in defs.items() if k == "lgb"}
    model._build_model_defs = mock_build
    
    # If we forget groups, it should raise ValueError
    with pytest.raises(ValueError):
        model.fit(X, y)
        
    # With groups, it should succeed
    res = model.fit(X, y, groups=groups)
    assert res._fitted
