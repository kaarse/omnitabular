"""Integration tests for OmniTabular."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes

from omnitabular import OmniTabular


@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


@pytest.fixture
def breast_cancer():
    return load_breast_cancer(return_X_y=True)


@pytest.fixture
def diabetes():
    return load_diabetes(return_X_y=True)

FAST_PARAMS = {
    "lgb_params": {"n_estimators": 2},
    "mlp_params": {"epochs": 1},
    "ft_transformer_params": {"epochs": 1},
    "tab_transformer_params": {"epochs": 1},
    "saint_params": {"epochs": 1},
    "tree_nn_params": {"epochs": 1},
    "tabnet_params": {"epochs": 1},
    "node_params": {"epochs": 1},
    "tabfm_params": {"epochs": 1},
    "gru_params": {"epochs": 1},
    "bigru_params": {"epochs": 1},
    "lstm_params": {"epochs": 1},
    "bilstm_params": {"epochs": 1},
    "cnn1d_params": {"epochs": 1},
    "cnn2d_params": {"epochs": 1},
    "gnn_params": {"epochs": 1},
    "temporal_transformer_params": {"epochs": 1},
    "sa_params": {"max_iter": 1},
    "ga_params": {"generations": 1, "pop_size": 2},
    "symbolic_regression_params": {"generations": 1, "pop_size": 2},
    "lcs_params": {"epochs": 1, "pop_size": 2},
    "rulefit_params": {"max_iter": 1, "n_rules": 2},
    "ripper_params": {"max_rules": 1, "max_conditions": 1},
    "pre_params": {"max_rules": 1},
}


# ── Number of models per task ──
CLF_MODEL_COUNT = 32 
REG_MODEL_COUNT = 31 


# ── Classification — Iris ────────────────────────────────────────────

class TestOmniTabularClassification:
    def test_fit_and_predict(self, iris):
        X, y = iris
        model = OmniTabular(
            task="classification",
            verbose=False,
            **FAST_PARAMS
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba(self, iris):
        X, y = iris
        model = OmniTabular(task="classification", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_leaderboard_structure(self, iris):
        X, y = iris
        model = OmniTabular(task="classification", verbose=False,
                            lgb_params={"n_estimators": 2},
                            mlp_params={"epochs": 1},
                            ft_transformer_params={"epochs": 1},
                            tab_transformer_params={"epochs": 1},
                            saint_params={"epochs": 1},
                            tree_nn_params={"epochs": 1},
                            tabnet_params={"epochs": 1},
                            node_params={"epochs": 1},
                            tabfm_params={"epochs": 1})
        model.fit(X, y)
        lb = model.leaderboard()
        assert isinstance(lb, pd.DataFrame)
        assert "model" in lb.columns
        assert "ENSEMBLE" in lb["model"].values
        assert len(lb) == CLF_MODEL_COUNT + 1  # models + ENSEMBLE

    def test_weights_sum_to_one(self, iris):
        X, y = iris
        model = OmniTabular(task="classification", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        np.testing.assert_allclose(model.weights_.sum(), 1.0, atol=1e-8)
        assert model.weights_.shape == (CLF_MODEL_COUNT,)

    def test_model_names(self, iris):
        X, y = iris
        model = OmniTabular(task="classification", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        assert set(model.model_names_) == {
            "knn", "lgb", "linear", "svm", "naive_bayes", "rf", "et", "xgb", "catboost",
            "mlp", "ft_transformer", "tab_transformer", "saint", "tree_nn", "tabnet",
            "node", "tabfm", "gru", "bigru", "lstm", "bilstm", "cnn1d", "cnn2d", "gnn",
            "temporal_transformer", "sa", "ga", "symbolic_regression", "lcs", "rulefit", "ripper", "pre"
        }

    def test_accuracy_above_threshold(self, iris):
        X, y = iris
        model = OmniTabular(task="classification", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        acc = (model.predict(X) == y).mean()
        assert acc > 0.90, f"Accuracy too low: {acc}"

    def test_auto_task_detection(self, iris):
        X, y = iris
        model = OmniTabular(task="auto", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        assert model.task_ == "classification"

    def test_dataframe_input(self, iris):
        X, y = iris
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        model = OmniTabular(task="classification", verbose=False, **FAST_PARAMS)
        model.fit(X_df, y)
        preds = model.predict(X_df)
        assert preds.shape == (len(y),)


# ── Binary Classification — Breast Cancer ────────────────────────────

class TestOmniTabularBinaryClassification:
    def test_binary_fit_and_predict(self, breast_cancer):
        X, y = breast_cancer
        model = OmniTabular(task="classification", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_binary_predict_proba_shape(self, breast_cancer):
        X, y = breast_cancer
        model = OmniTabular(task="classification", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_all_models_present(self, breast_cancer):
        X, y = breast_cancer
        model = OmniTabular(task="classification", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        assert len(model.models_) == CLF_MODEL_COUNT


# ── Regression — Diabetes ────────────────────────────────────────────

class TestOmniTabularRegression:
    def test_fit_and_predict(self, diabetes):
        X, y = diabetes
        model = OmniTabular(task="regression", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_raises(self, diabetes):
        X, y = diabetes
        model = OmniTabular(task="regression", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_leaderboard_metric_column(self, diabetes):
        X, y = diabetes
        model = OmniTabular(task="regression", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        lb = model.leaderboard()
        assert "oof_rmse" in lb.columns

    def test_naive_bayes_skipped(self, diabetes):
        X, y = diabetes
        model = OmniTabular(task="regression", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        assert "naive_bayes" not in model.model_names_
        assert len(model.model_names_) == REG_MODEL_COUNT

    def test_weights_shape_regression(self, diabetes):
        X, y = diabetes
        model = OmniTabular(task="regression", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        assert model.weights_.shape == (REG_MODEL_COUNT,)
        np.testing.assert_allclose(model.weights_.sum(), 1.0, atol=1e-8)

    def test_model_names_regression(self, diabetes):
        X, y = diabetes
        model = OmniTabular(task="regression", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        assert set(model.model_names_) == {
            "knn", "lgb", "linear", "svm", "rf", "et", "xgb", "catboost",
            "mlp", "ft_transformer", "tab_transformer", "saint", "tree_nn", "tabnet",
            "node", "tabfm", "gru", "bigru", "lstm", "bilstm", "cnn1d", "cnn2d", "gnn",
            "temporal_transformer", "sa", "ga", "symbolic_regression", "lcs", "rulefit", "ripper", "pre"
        }

    def test_feature_importances(self, diabetes):
        X, y = diabetes
        model = OmniTabular(task="regression", verbose=False, **FAST_PARAMS)
        model.fit(X, y)
        fi = model.feature_importances()
        assert isinstance(fi, pd.DataFrame)
        assert len(fi) == X.shape[1]


# ── Error handling ───────────────────────────────────────────────────

class TestOmniTabularErrors:
    def test_predict_before_fit_raises(self):
        model = OmniTabular(task="classification")
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 4)))

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            OmniTabular(task="unsupported_task")
