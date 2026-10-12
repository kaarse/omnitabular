"""Regression coverage for training, validation and inference failures."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from omnitabular import OmniTabular, MLPModel, WeightedEnsemble
from omnitabular.preprocessing import TabularPreprocessor


def quick_model(**kwargs):
    return OmniTabular(models=["linear", "rf"], cv_folds=2, verbose=False,
                       rf_params={"n_estimators": 5}, hc_params={"n_iterations": 15}, **kwargs)


@pytest.mark.parametrize("labels", [np.array(["a", "b", "c"]), np.array([10, 20, 30])])
@pytest.mark.parametrize("metric", ["accuracy", "log_loss"])
def test_original_labels_and_metrics(labels, metric):
    X, y = load_iris(return_X_y=True)
    model = quick_model(metric=metric).fit(X, labels[y])
    assert set(model.predict(X)) <= set(labels)
    assert model.predict_proba(X).shape == (len(y), 3)
    assert np.isfinite(model.ensemble_oof_score_)
    assert model.leaderboard().shape[0] == 3
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_custom_metric_leaderboard_and_explanation():
    X, y = load_iris(return_X_y=True)
    model = quick_model(metric=accuracy_score, metric_higher_is_better=True).fit(X, y)
    assert "oof_accuracy_score" in model.leaderboard()
    assert len(model.explain(X, y, n_repeats=1)) == X.shape[1]


def test_time_series_only_scores_validated_rows():
    X = np.arange(120, dtype=float).reshape(60, 2)
    y = np.arange(60, dtype=float) + 100
    model = OmniTabular(models=["linear"], task="regression", cv_strategy="time_series",
                       cv_folds=3, metric="mae", verbose=False,
                       hc_params={"n_iterations": 0}).fit(X, y)
    assert not model.oof_mask_[:15].any()
    assert model.oof_mask_[15:].all()
    assert model.ensemble_oof_score_ < 1e-5


def test_preprocessing_is_fitted_inside_folds(monkeypatch):
    observed = []
    original = TabularPreprocessor.fit

    def record(self, X):
        observed.append(len(X))
        return original(self, X)

    monkeypatch.setattr(TabularPreprocessor, "fit", record)
    X = np.arange(60, dtype=float).reshape(30, 2)
    OmniTabular(models=["linear"], task="regression", cv_folds=3, verbose=False,
                hc_params={"n_iterations": 0}).fit(X, X[:, 0])
    assert observed == [20, 20, 20, 30]


def test_preprocessing_schema_missing_values_and_unknown_categories():
    X = pd.DataFrame({"number": pd.Series([1, 2, None], dtype="Int64"),
                      "category": pd.Series(["a", None, "b"], dtype="string"),
                      "empty": [np.nan] * 3})
    processor = TabularPreprocessor().fit(X)
    expected = processor.transform(X)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(processor.transform(X[list(reversed(X.columns))]), expected)
    np.testing.assert_allclose(processor.transform(X.to_numpy()), expected)
    unseen = X.copy()
    unseen.loc[0, "category"] = "unseen"
    assert np.isfinite(processor.transform(unseen)).all()
    with pytest.raises(ValueError, match="columns"):
        processor.transform(X.drop(columns="empty"))


def test_neural_auto_labels_and_ensemble():
    X = np.random.default_rng(0).normal(size=(12, 3))
    y = np.array(["yes", "no"] * 6)
    model = MLPModel(epochs=1, batch_size=4).fit(X, y)
    assert model.task == "classification"
    assert set(model.predict(X)) <= set(y)
    ensemble = WeightedEnsemble([model], [1.0], "classification")
    np.testing.assert_array_equal(ensemble.predict(X), model.predict(X))


@pytest.mark.parametrize("kwargs", [{"models": []}, {"models": ["typo"]},
                                   {"cv_strategy": "typo"}, {"cv_folds": 1}, {"n_jobs": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OmniTabular(**kwargs)


def test_refit_clears_old_models_and_failed_fit_invalidates_state():
    X, y = load_iris(return_X_y=True)
    model = quick_model().fit(X, y)
    model.models = ["linear"]
    model.fit(X, y)
    assert list(model.models_) == ["linear"]
    with pytest.raises(ValueError):
        model.fit(X, y[:-1])
    with pytest.raises(RuntimeError):
        model.predict(X)


@pytest.mark.parametrize("features", [1, 2, 4, 5])
@pytest.mark.parametrize("kind", ["cnn1d", "cnn2d"])
def test_cnn_handles_narrow_tables(features, kind):
    from omnitabular.cnn import CNN1DModel, CNN2DModel
    model_cls = CNN1DModel if kind == "cnn1d" else CNN2DModel
    X = np.random.default_rng(0).normal(size=(12, features))
    model = model_cls(task="regression", epochs=1, batch_size=4).fit(X, np.arange(12.))
    assert model.predict(X).shape == (12,)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_simulated_annealing_scalar_bias(task):
    from omnitabular.evolutionary import SAModel
    X = np.random.default_rng(0).normal(size=(12, 3))
    y = np.arange(12) % 2 if task == "classification" else np.arange(12.)
    model = SAModel(task=task, max_iter=2).fit(X, y)
    assert np.isfinite(model.predict(X)).all()
