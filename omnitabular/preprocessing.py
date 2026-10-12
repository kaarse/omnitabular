"""Fold-local imputation, scaling and categorical encoding for tabular data."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class TabularPreprocessor:
    def __init__(self):
        self.pipeline = None
        self.fitted = False

    def _frame(self, X, fitting=False):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            if not frame.columns.is_unique:
                raise ValueError("Feature column names must be unique.")
            if not fitting:
                if set(frame.columns) != set(self.columns_):
                    raise ValueError("Prediction columns must match the training columns.")
                frame = frame.loc[:, self.columns_]
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be a two-dimensional table.")
            if not fitting and array.shape[1] != len(self.columns_):
                raise ValueError("Prediction feature count must match training data.")
            frame = pd.DataFrame(array, columns=None if fitting else self.columns_).infer_objects()
        if fitting:
            self.columns_ = frame.columns.tolist()
            self.numeric_ = frame.select_dtypes(include="number").columns.tolist()
            self.categorical_ = [c for c in self.columns_ if c not in self.numeric_]
        # Use positional indices internally, including for non-string column names.
        frame.columns = range(len(frame.columns))
        for col in self.numeric_:
            idx = self.columns_.index(col)
            frame[idx] = pd.to_numeric(frame[idx], errors="raise").to_numpy(dtype=float, na_value=np.nan)
        for col in self.categorical_:
            idx = self.columns_.index(col)
            series = frame[idx]
            frame[idx] = series.map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)
        return frame

    def fit(self, X):
        frame = self._frame(X, fitting=True)
        numeric = [self.columns_.index(c) for c in self.numeric_]
        categorical = [self.columns_.index(c) for c in self.categorical_]
        transformers = []
        if numeric:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]), numeric))
        if categorical:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical))
        if not transformers:
            raise ValueError("X must contain at least one feature.")
        self.pipeline = ColumnTransformer(transformers=transformers)
        self.pipeline.fit(frame)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        return np.asarray(self.pipeline.transform(self._frame(X)), dtype=np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
