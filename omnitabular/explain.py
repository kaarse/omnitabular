"""
Global Explainability module using Permutation Importance.
Evaluates the feature importances of the entire OmniTabular ensemble organically.
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .omnitabular import OmniTabular

def ensemble_permutation_importance(
    model: 'OmniTabular',
    X,
    y,
    scorer,
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes agnostic Permutation Importance for the fully fitted instance.
    
    Parameters
    ----------
    model : OmniTabular
        The fitted OmniTabular ensemble.
    X : array-like or pd.DataFrame
        Dataset features to evaluate.
    y : array-like
        True labels.
    scorer : str or callable
        Scorer string defined in sklearn.metrics (e.g. 'neg_log_loss' or 'neg_root_mean_squared_error')
    n_repeats : int
        Number of times to permute a feature.
    random_state : int
        Seeding for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Sorted dataframe with ['feature_index', 'importance_mean', 'importance_std']
    """
    
    from sklearn.base import BaseEstimator
    class EnsembleWrapper(BaseEstimator):
        def __init__(self, ensemble):
            self.ensemble = ensemble
            # Fake classes_ attribute if classification
            if self.ensemble.task_ == "classification":
                self.classes_ = np.unique(y)
                self._estimator_type = "classifier"
            else:
                self._estimator_type = "regressor"
                
        def fit(self, X, y):
            pass # Already fitted

        def predict(self, X_infer):
            return self.ensemble.predict(X_infer)
            
        def predict_proba(self, X_infer):
            return self.ensemble.predict_proba(X_infer)

    wrapper = EnsembleWrapper(model)
    
    result = permutation_importance(
        wrapper, X, y, scoring=scorer, n_repeats=n_repeats,
        random_state=random_state, n_jobs=1
    )
    
    importances_mean = result.importances_mean
    importances_std = result.importances_std
    
    df = pd.DataFrame({
        'feature_index': range(len(importances_mean)),
        'importance_mean': importances_mean,
        'importance_std': importances_std
    })
    
    return df.sort_values(by='importance_mean', ascending=False).reset_index(drop=True)
