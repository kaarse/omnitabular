"""
AutoML Preprocessing Pipeline for OmniTabular.
Protects and scales models automatically by imputing NaNs, encoding categorical features,
and ensuring Deep/Evolutionary models receive StandardScaled float arrays.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class TabularPreprocessor:
    """
    Automated preprocessing pipeline.
    Identifies numeric and categorical variables,
    applies optimal imputation, scaling, and one-hot encoding.
    """
    def __init__(self):
        self.pipeline = None
        self.fitted = False

    def fit(self, X):
        # Convert X to DataFrame if it isn't one so we can inspect dtypes easily
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
            # If they passed an array with strings, it might be object dtype
            X_df = X_df.infer_objects()
        else:
            X_df = X

        numeric_features = X_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_features = X_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough' # For types we didn't catch, just pass them, but there shouldn't be
        )

        self.pipeline.fit(X_df)
        self.fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        # Pipeline outputs numpy array since sparse_output=False
        transformed_X = self.pipeline.transform(X_df)
        return np.asarray(transformed_X, dtype=np.float32)

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
