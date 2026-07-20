"""
OmniTabular end-to-end demonstration.

Models:  KNN · LightGBM · Linear/Logistic · SVM · Naive Bayes
Ensemble: Hill Climbing weight optimisation

Run with:  python examples/demo.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from omnitabular import OmniTabular


SEPARATOR = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ─────────────────────────────────────────────────────────────────────
# 1.  Multiclass Classification — Iris
# ─────────────────────────────────────────────────────────────────────

def demo_iris():
    section("Demo 1 · Multiclass Classification · Iris")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = OmniTabular(
        task="classification",
        lgb_params={"n_estimators": 200},
        hc_params={"n_iterations": 2000},
        verbose=True,
    )
    model.fit(X_train, y_train)

    print("\n--- Leaderboard ---")
    print(model.leaderboard().to_string(index=False))

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy    : {acc:.4f}")
    print(f"Ensemble weights : {np.round(model.weights_, 4)}")
    print(f"Active models    : {model.model_names_}")


# ─────────────────────────────────────────────────────────────────────
# 2.  Binary Classification — Breast Cancer
# ─────────────────────────────────────────────────────────────────────

def demo_breast_cancer():
    section("Demo 2 · Binary Classification · Breast Cancer")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = OmniTabular(
        task="classification",
        metric="auc",
        lgb_params={"n_estimators": 200},
        hc_params={"n_iterations": 1500},
        verbose=True,
    )
    model.fit(X_train, y_train)

    print("\n--- Leaderboard ---")
    print(model.leaderboard().to_string(index=False))

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nTest AUC         : {auc:.4f}")
    print(f"Ensemble weights : {np.round(model.weights_, 4)}")


# ─────────────────────────────────────────────────────────────────────
# 3.  Regression — Diabetes
# ─────────────────────────────────────────────────────────────────────

def demo_diabetes():
    section("Demo 3 · Regression · Diabetes")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = OmniTabular(
        task="regression",
        lgb_params={"n_estimators": 300, "num_leaves": 31},
        hc_params={"n_iterations": 1500},
        verbose=True,
    )
    model.fit(X_train, y_train)

    print("\n--- Leaderboard ---")
    print(model.leaderboard().to_string(index=False))

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    print(f"\nTest RMSE        : {rmse:.4f}")
    print(f"Ensemble weights : {np.round(model.weights_, 4)}")
    print(f"Active models    : {model.model_names_}")

    print("\n--- Top Feature Importances (LGB) ---")
    fi = model.feature_importances()
    print(fi.head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────
# 4.  Individual model usage examples
# ─────────────────────────────────────────────────────────────────────

def demo_individual_models():
    section("Demo 4 · Individual Model Examples")
    from omnitabular import KNNModel, LGBModel, LinearModel, SVMModel, NaiveBayesModel

    X, y = load_iris(return_X_y=True)

    models = {
        "KNN":         KNNModel(task="classification", n_neighbors=5),
        "LightGBM":    LGBModel(task="classification", n_estimators=100),
        "Logistic":    LinearModel(task="classification", C=1.0),
        "SVM (RBF)":   SVMModel(task="classification", kernel="rbf"),
        "Naive Bayes": NaiveBayesModel(task="classification", variant="gaussian"),
    }

    print(f"\n{'Model':<15} {'Train Accuracy':>15}")
    print("-" * 32)
    for name, m in models.items():
        m.fit(X, y)
        acc = (m.predict(X) == y).mean()
        print(f"{name:<15} {acc:>15.4f}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_iris()
    demo_breast_cancer()
    demo_diabetes()
    demo_individual_models()

    print(f"\n{SEPARATOR}")
    print("  All demos completed successfully ✓")
    print(SEPARATOR)
