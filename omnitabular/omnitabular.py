"""
OmniTabular — the unified high-level API.

Ensembles KNN + LightGBM + Linear/Logistic Regression + SVM + Naive Bayes
with Hill Climbing weight optimisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd

from .knn import KNNModel
from .lgb import LGBModel
from .linear import LinearModel
from .svm import SVMModel
from .naive_bayes import NaiveBayesModel
from .rf import RFModel
from .et import ETModel
from .xgb import XGBModel
from .cat import CatBoostModel
from .mlp import MLPModel
from .ft_transformer import FTTransformerModel
from .tab_transformer import TabTransformerModel
from .saint import SAINTModel
from .tree_nn import TreeNNModel
from .tabnet import TabNetModel
from .node import NODEModel
from .tabfm import TabFMModel
from .hill_climbing import HillClimbingEnsemble
from .ensemble import WeightedEnsemble
from .rnn import GRUModel, BiGRUModel, LSTMModel, BiLSTMModel
from .cnn import CNN1DModel, CNN2DModel
from .gnn import GNNModel
from .temporal_transformer import TemporalTransformerModel
from .evolutionary import SAModel, GAModel, SymbolicRegressionModel, LCSModel
from .rules import RuleFitModel, RipperModel, PREModel
from .utils import detect_task, evaluate, get_default_metric, METRIC_REGISTRY
import joblib
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from .preprocessing import TabularPreprocessor
from .explain import ensemble_permutation_importance


# ── Model registry ───────────────────────────────────────────────────

# (class, default_kwargs, supported_tasks)
_MODEL_REGISTRY = {
    "knn":         (KNNModel,        {},                      {"classification", "regression"}),
    "lgb":         (LGBModel,        {},                      {"classification", "regression"}),
    "linear":      (LinearModel,     {},                      {"classification", "regression"}),
    "svm":         (SVMModel,        {},                      {"classification", "regression"}),
    "naive_bayes": (NaiveBayesModel, {"variant": "gaussian"}, {"classification"}),
    "rf":          (RFModel,         {},                      {"classification", "regression"}),
    "et":          (ETModel,         {},                      {"classification", "regression"}),
    "xgb":         (XGBModel,        {},                      {"classification", "regression"}),
    "catboost":    (CatBoostModel,   {},                      {"classification", "regression"}),
    "mlp":         (MLPModel,        {},                      {"classification", "regression"}),
    "ft_transformer": (FTTransformerModel, {},               {"classification", "regression"}),
    "tab_transformer": (TabTransformerModel, {},             {"classification", "regression"}),
    "saint":       (SAINTModel,      {},                      {"classification", "regression"}),
    "tree_nn":     (TreeNNModel,     {},                      {"classification", "regression"}),
    "tabnet":      (TabNetModel,     {},                      {"classification", "regression"}),
    "node":        (NODEModel,       {},                      {"classification", "regression"}),
    "tabfm":       (TabFMModel,      {},                      {"classification", "regression"}),
    "gru":         (GRUModel,        {},                      {"classification", "regression"}),
    "bigru":       (BiGRUModel,      {},                      {"classification", "regression"}),
    "lstm":        (LSTMModel,       {},                      {"classification", "regression"}),
    "bilstm":      (BiLSTMModel,     {},                      {"classification", "regression"}),
    "cnn1d":       (CNN1DModel,      {},                      {"classification", "regression"}),
    "cnn2d":       (CNN2DModel,      {},                      {"classification", "regression"}),
    "gnn":         (GNNModel,        {},                      {"classification", "regression"}),
    "temporal_transformer": (TemporalTransformerModel, {},    {"classification", "regression"}),
    "sa":          (SAModel,         {},                      {"classification", "regression"}),
    "ga":          (GAModel,         {},                      {"classification", "regression"}),
    "symbolic_regression": (SymbolicRegressionModel, {},      {"classification", "regression"}),
    "lcs":         (LCSModel,        {},                      {"classification", "regression"}),
    "rulefit":     (RuleFitModel,    {},                      {"classification", "regression"}),
    "ripper":      (RipperModel,     {},                      {"classification", "regression"}),
    "pre":         (PREModel,        {},                      {"classification", "regression"}),
}


class OmniTabular:
    """
    High-level tabular ML class combining **KNN**, **LightGBM**,
    **Linear / Logistic Regression**, **SVM**, **Naive Bayes**,
    **Random Forest**, **Extra Trees**, **XGBoost**, and **CatBoost**
    with **Hill Climbing ensemble weight optimisation**.

    Workflow
    --------
    1. ``fit(X, y)``
       a. Detect task type (classification / regression).
       b. Select applicable models (Naive Bayes is skipped for regression).
       c. Generate cross-validated OOF predictions per model.
       d. Run Hill Climbing to find optimal blending weights.
       e. Re-train each model on the full dataset.
    2. ``predict(X)`` / ``predict_proba(X)``
       Returns blended predictions using optimal weights.
    3. ``leaderboard()``
       Returns a DataFrame comparing individual OOF scores vs. ensemble.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}
    knn_params : dict, optional
    lgb_params : dict, optional
    linear_params : dict, optional
    svm_params : dict, optional
    naive_bayes_params : dict, optional
    rf_params : dict, optional
    et_params : dict, optional
    xgb_params : dict, optional
    catboost_params : dict, optional
    hc_params : dict, optional
        Extra parameters for :class:`HillClimbingEnsemble`.
    metric : str or None
        Evaluation metric.  Defaults to ``"log_loss"`` (classification) or
        ``"rmse"`` (regression).
    cv_folds : int
        Number of cross-validation folds for OOF generation.
    random_state : int
    verbose : bool

    Attributes
    ----------
    task_ : str
        Detected or specified task type.
    model_names_ : list[str]
        Names of the models that participated in the ensemble (task-compatible).
    models_ : dict[str, BaseModel]
        Fitted base models keyed by name.
    weights_ : np.ndarray
        Optimal blending weights (same order as ``model_names_``).
    ensemble_ : WeightedEnsemble
        The final blending ensemble.
    oof_scores_ : dict[str, float]
        Per-model OOF metric scores.
    ensemble_oof_score_ : float
        OOF metric score of the blended ensemble.

    Examples
    --------
    >>> from omnitabular import OmniTabular
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = OmniTabular(task="classification", verbose=True)
    >>> model.fit(X, y)
    >>> print(model.leaderboard())
    >>> preds = model.predict(X[:10])
    """

    def __init__(
        self,
        task: str = "auto",
        auto_preprocess: bool = True,
        knn_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        linear_params: Optional[Dict] = None,
        svm_params: Optional[Dict] = None,
        naive_bayes_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
        et_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        catboost_params: Optional[Dict] = None,
        mlp_params: Optional[Dict] = None,
        ft_transformer_params: Optional[Dict] = None,
        tab_transformer_params: Optional[Dict] = None,
        saint_params: Optional[Dict] = None,
        tree_nn_params: Optional[Dict] = None,
        tabnet_params: Optional[Dict] = None,
        node_params: Optional[Dict] = None,
        tabfm_params: Optional[Dict] = None,
        gru_params: Optional[Dict] = None,
        bigru_params: Optional[Dict] = None,
        lstm_params: Optional[Dict] = None,
        bilstm_params: Optional[Dict] = None,
        cnn1d_params: Optional[Dict] = None,
        cnn2d_params: Optional[Dict] = None,
        gnn_params: Optional[Dict] = None,
        temporal_transformer_params: Optional[Dict] = None,
        sa_params: Optional[Dict] = None,
        ga_params: Optional[Dict] = None,
        symbolic_regression_params: Optional[Dict] = None,
        lcs_params: Optional[Dict] = None,
        rulefit_params: Optional[Dict] = None,
        ripper_params: Optional[Dict] = None,
        pre_params: Optional[Dict] = None,
        hc_params: Optional[Dict] = None,
        metric: Optional[Union[str, Callable]] = None,
        metric_higher_is_better: bool = False,
        metric_needs_proba: bool = False,
        cv_folds: int = 5,
        cv_strategy: str = "kfold",
        window_size: Optional[int] = None,
        random_state: int = 42,
        verbose: bool = True,
        n_jobs: int = -1,
    ) -> None:
        if task not in ("auto", "classification", "regression"):
            raise ValueError(f"Unknown task '{task}'.")

        self.task = task
        self._user_params = {
            "knn":         knn_params or {},
            "lgb":         lgb_params or {},
            "linear":      linear_params or {},
            "svm":         svm_params or {},
            "naive_bayes": naive_bayes_params or {},
            "rf":          rf_params or {},
            "et":          et_params or {},
            "xgb":         xgb_params or {},
            "catboost":    catboost_params or {},
            "mlp":         mlp_params or {},
            "ft_transformer": ft_transformer_params or {},
            "tab_transformer": tab_transformer_params or {},
            "saint":       saint_params or {},
            "tree_nn":     tree_nn_params or {},
            "tabnet":      tabnet_params or {},
            "node":        node_params or {},
            "tabfm":       tabfm_params or {},
            "gru":         gru_params or {},
            "bigru":       bigru_params or {},
            "lstm":        lstm_params or {},
            "bilstm":      bilstm_params or {},
            "cnn1d":       cnn1d_params or {},
            "cnn2d":       cnn2d_params or {},
            "gnn":         gnn_params or {},
            "temporal_transformer": temporal_transformer_params or {},
            "sa":          sa_params or {},
            "ga":          ga_params or {},
            "symbolic_regression": symbolic_regression_params or {},
            "lcs":         lcs_params or {},
            "rulefit":     rulefit_params or {},
            "ripper":      ripper_params or {},
            "pre":         pre_params or {},
        }
        self.hc_params = hc_params or {}
        self.metric = metric
        self.metric_higher_is_better = metric_higher_is_better
        self.metric_needs_proba = metric_needs_proba
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.window_size = window_size
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.auto_preprocess = auto_preprocess
        self.preprocessor_: Optional[TabularPreprocessor] = None

        # ── backward-compat convenience accessors ──
        self.knn_params = self._user_params["knn"]
        self.lgb_params = self._user_params["lgb"]
        self.linear_params = self._user_params["linear"]
        self.svm_params = self._user_params["svm"]
        self.naive_bayes_params = self._user_params["naive_bayes"]
        self.rf_params = self._user_params["rf"]
        self.et_params = self._user_params["et"]
        self.xgb_params = self._user_params["xgb"]
        self.catboost_params = self._user_params["catboost"]
        self.mlp_params = self._user_params["mlp"]
        self.ft_transformer_params = self._user_params["ft_transformer"]
        self.tab_transformer_params = self._user_params["tab_transformer"]
        self.saint_params = self._user_params["saint"]
        self.tree_nn_params = self._user_params["tree_nn"]
        self.tabnet_params = self._user_params["tabnet"]
        self.node_params = self._user_params["node"]
        self.tabfm_params = self._user_params["tabfm"]
        self.gru_params = self._user_params["gru"]
        self.bigru_params = self._user_params["bigru"]
        self.lstm_params = self._user_params["lstm"]
        self.bilstm_params = self._user_params["bilstm"]
        self.cnn1d_params = self._user_params["cnn1d"]
        self.cnn2d_params = self._user_params["cnn2d"]
        self.gnn_params = self._user_params["gnn"]
        self.temporal_transformer_params = self._user_params["temporal_transformer"]
        self.sa_params = self._user_params["sa"]
        self.ga_params = self._user_params["ga"]
        self.symbolic_regression_params = self._user_params["symbolic_regression"]
        self.lcs_params = self._user_params["lcs"]
        self.rulefit_params = self._user_params["rulefit"]
        self.ripper_params = self._user_params["ripper"]
        self.pre_params = self._user_params["pre"]

        # Will be set during fit
        self.task_: Optional[str] = None
        self.model_names_: List[str] = []
        self.models_: Dict[str, object] = {}
        self.weights_: Optional[np.ndarray] = None
        self.ensemble_: Optional[WeightedEnsemble] = None
        self.oof_scores_: Dict[str, float] = {}
        self.ensemble_oof_score_: Optional[float] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[OmniTabular] {msg}")

    def _generate_splits(self, X, y, groups=None):
        if self.cv_strategy == "group_kfold":
            if groups is None:
                raise ValueError("cv_strategy='group_kfold' requires 'groups' in fit().")
            return GroupKFold(n_splits=self.cv_folds).split(X, y, groups)
        elif self.cv_strategy == "time_series":
            return TimeSeriesSplit(n_splits=self.cv_folds).split(X)
        elif self.cv_strategy == "sliding_window":
            return TimeSeriesSplit(n_splits=self.cv_folds, max_train_size=self.window_size).split(X)
        else: # "kfold" default
            if self.task_ == "classification":
                return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state).split(X, y)
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state).split(X, y)

    def _build_model_defs(self) -> Dict[str, Tuple]:
        """
        Return {name: (cls, kwargs)} for models compatible with the current task.
        """
        defs = {}
        for name, (cls, defaults, supported_tasks) in _MODEL_REGISTRY.items():
            if self.task_ not in supported_tasks:
                self._log(f"Skipping {name.upper()} (not supported for {self.task_})")
                continue
            merged = {"task": self.task_, **defaults, **self._user_params.get(name, {})}
            defs[name] = (cls, merged)
        return defs

    def _oof_for_model(
        self,
        name: str,
        model_cls,
        model_kwargs: dict,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Cross-validated OOF generation for a single model class."""
        n = len(y)
        oof_pred = np.zeros(n, dtype=float)

        if self.task_ == "classification":
            n_classes = len(np.unique(y))
            oof_proba = np.zeros((n, n_classes), dtype=float)
        else:
            oof_proba = None

        splits = list(self._generate_splits(X, y, groups))
        for fold, (tr_idx, val_idx) in enumerate(splits, start=1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr = y[tr_idx]

            m = model_cls(**model_kwargs)
            m.fit(X_tr, y_tr)
            oof_pred[val_idx] = m.predict(X_val)

            if self.task_ == "classification":
                oof_proba[val_idx] = m.predict_proba(X_val)

        return oof_pred, oof_proba

    def _score_oof(
        self,
        y: np.ndarray,
        oof_pred: np.ndarray,
        oof_proba: Optional[np.ndarray],
        metric_name: Union[str, Callable],
    ) -> float:
        if callable(metric_name):
            needs_proba = self.metric_needs_proba
        else:
            _, _, needs_proba = METRIC_REGISTRY[metric_name]
            
        if needs_proba:
            return evaluate(metric_name, y, oof_pred, proba=oof_proba, needs_proba=needs_proba)
        return evaluate(metric_name, y, oof_pred, needs_proba=needs_proba)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, groups=None) -> "OmniTabular":
        """
        Fit OmniTabular on (X, y).

        Steps
        -----
        1. Detect task + choose metric.
        2. OOF prediction for every compatible base model.
        3. Hill Climbing on OOF predictions to find weights.
        4. Re-train on full dataset.
        """
        # --- Step 0: Auto Preprocessing ---
        if self.auto_preprocess:
            self.preprocessor_ = TabularPreprocessor()
            X = self.preprocessor_.fit_transform(X)
        else:
            X = np.asarray(X) if not isinstance(X, np.ndarray) else X
            
        y = np.asarray(y)

        # --- Step 1: task & metric ---
        if self.task == "auto":
            self.task_ = detect_task(y)
        else:
            self.task_ = self.task
        self._log(f"Task detected: {self.task_}")

        if callable(self.metric):
            metric_name = self.metric
            self._log(f"Optimising custom callable metric")
            needs_proba = self.metric_needs_proba
        else:
            metric_name = self.metric or get_default_metric(self.task_)
            self._log(f"Optimising metric: {metric_name}")
            _, _, needs_proba = METRIC_REGISTRY[metric_name]

        # --- Step 2: OOF predictions ---
        model_defs = self._build_model_defs()
        self.model_names_ = list(model_defs.keys())
        n_models = len(self.model_names_)
        self._log(f"Base models ({n_models}): {', '.join(n.upper() for n in self.model_names_)}")

        oof_preds_list = []
        oof_probas_list = []
        oof_scores = {}

        self._log(f"Generating OOF predictions... (Parallel n_jobs={self.n_jobs})")
        
        def _run_oof(name, cls, kwargs):
            o_pred, o_proba = self._oof_for_model(name, cls, kwargs, X, y, groups)
            s = self._score_oof(y, o_pred, o_proba, metric_name)
            return name, o_pred, o_proba, s
            
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_run_oof)(name, cls, kwargs) for name, (cls, kwargs) in model_defs.items()
        )

        for name, oof_pred, oof_proba, score in results:
            oof_scores[name] = score
            oof_preds_list.append(oof_pred)
            oof_probas_list.append(oof_proba)
            self._log(f"  {name.upper()} OOF {metric_name}: {score:.6f}")

        self.oof_scores_ = oof_scores

        # --- Step 3: Hill Climbing weight search ---
        self._log("Running Hill Climbing to optimise ensemble weights ...")

        hc_defaults = dict(
            metric=metric_name,
            metric_higher_is_better=self.metric_higher_is_better,
            metric_needs_proba=self.metric_needs_proba,
            n_iterations=3000,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        hc_defaults.update(self.hc_params)

        hc = HillClimbingEnsemble(**hc_defaults)
        hc.fit(
            y_true=y,
            task=self.task_,
            oof_preds=oof_preds_list if not needs_proba else None,
            oof_probas=oof_probas_list if needs_proba else None,
        )

        self.weights_ = hc.weights_
        self._hc = hc

        # Ensemble OOF score
        if needs_proba:
            blended_proba = hc.blend_predictions(oof_probas_list)
            blended_pred = np.argmax(blended_proba, axis=1)
            self.ensemble_oof_score_ = evaluate(
                metric_name, y, blended_pred, proba=blended_proba, needs_proba=needs_proba
            )
        else:
            blended_pred = hc.blend_predictions(oof_preds_list)
            self.ensemble_oof_score_ = evaluate(metric_name, y, blended_pred, needs_proba=needs_proba)

        self._log(f"Ensemble OOF {metric_name}: {self.ensemble_oof_score_:.6f}")
        weight_str = ", ".join(
            f"{n.upper()}={w:.4f}" for n, w in zip(self.model_names_, self.weights_)
        )
        self._log(f"Optimal weights → {weight_str}")

        # --- Step 4: Full-data re-training ---
        self._log(f"Re-training models on full dataset... (Parallel n_jobs={self.n_jobs})")
        
        def _train_full(name, cls, kwargs):
            m = cls(**kwargs)
            m.fit(X, y) # Groups ignored at final train map
            return name, m
            
        fitted_results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_train_full)(name, cls, kwargs) for name, (cls, kwargs) in model_defs.items()
        )
        
        fitted_models = []
        for name, m in fitted_results:
            self.models_[name] = m
            fitted_models.append(m)

        self.ensemble_ = WeightedEnsemble(
            models=fitted_models,
            weights=self.weights_,
            task=self.task_,
        )
        self._fitted = True
        self._log("Fit complete ✓")
        return self

    def predict(self, X) -> np.ndarray:
        """Return final blended predictions."""
        self._check_fitted()
        if self.auto_preprocess and self.preprocessor_ is not None:
            X = self.preprocessor_.transform(X)
        else:
            X = np.asarray(X)
        return self.ensemble_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return final blended probability predictions (classification only)."""
        self._check_fitted()
        if self.task_ != "classification":
            raise NotImplementedError("predict_proba is only available for classification.")
        if self.auto_preprocess and self.preprocessor_ is not None:
            X = self.preprocessor_.transform(X)
        else:
            X = np.asarray(X)
        return self.ensemble_.predict_proba(X)

    def explain(self, X, y, n_repeats=5) -> pd.DataFrame:
        """
        Compute Permutation Feature Importance for the entire ensemble.
        """
        self._check_fitted()
        
        if callable(self.metric):
            from sklearn.metrics import make_scorer
            scorer = make_scorer(
                self.metric, 
                greater_is_better=self.metric_higher_is_better, 
                needs_proba=self.metric_needs_proba
            )
        else:
            metric_name = self.metric or get_default_metric(self.task_)
            # Scikit-learn permutation importance utilizes scoring strings
            if metric_name == "log_loss":
                scorer = "neg_log_loss"
            elif metric_name == "roc_auc":
                scorer = "roc_auc"
            elif metric_name == "rmse":
                scorer = "neg_root_mean_squared_error"
            else:
                scorer = "accuracy" if self.task_ == "classification" else "r2"

        return ensemble_permutation_importance(
            self, X, y, scorer=scorer, n_repeats=n_repeats, random_state=self.random_state
        )

    def leaderboard(self) -> pd.DataFrame:
        """
        Return a comparison DataFrame of individual model OOF scores and the ensemble.

        Returns
        -------
        pd.DataFrame with columns: [model, oof_<metric>, weight]
        """
        self._check_fitted()
        metric_name = self.metric or get_default_metric(self.task_)
        _, higher_is_better, _ = METRIC_REGISTRY[metric_name]

        rows = []
        for i, (name, score) in enumerate(self.oof_scores_.items()):
            rows.append({
                "model": name.upper(),
                f"oof_{metric_name}": round(score, 6),
                "weight": round(float(self.weights_[i]), 4),
            })
        rows.append({
            "model": "ENSEMBLE",
            f"oof_{metric_name}": round(self.ensemble_oof_score_, 6),
            "weight": 1.0,
        })

        df = pd.DataFrame(rows)
        df = df.sort_values(
            f"oof_{metric_name}", ascending=not higher_is_better
        ).reset_index(drop=True)
        return df

    def feature_importances(self) -> pd.DataFrame:
        """
        Return LGB feature importances (gain-based).

        Returns
        -------
        pd.DataFrame with columns: [feature_index, importance]
        """
        self._check_fitted()
        if "lgb" not in self.models_:
            raise RuntimeError("LGB model is not in the ensemble.")
        lgb_model = self.models_["lgb"]
        imp = lgb_model.feature_importances_
        return (
            pd.DataFrame({"feature_index": range(len(imp)), "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("OmniTabular is not fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"OmniTabular(task='{self.task}', cv_folds={self.cv_folds}, "
            f"metric='{self.metric}', status={status})"
        )
