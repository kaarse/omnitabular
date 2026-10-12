# OmniTabular

> **The Ultimate Tabular ML Framework.** Seamlessly ensemble and optimise over 30 state-of-the-art architectures—from classic decision trees to deep neural networks, rule-extractors, and evolutionary algorithms—all united under a single API with Hill Climbing weight optimization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)

---

## 🚀 Features

| Feature | Detail |
|---|---|
| **32 Architectures** | Native support for Gradient Boosting, Transformers, NNs, LSTMs, CNNs, GNNs, and Evolutionary models. |
| **Hill Climbing Optimization** | Stochastic weight search on OOF predictions with thermodynamic annealing to find the perfect ensemble blend. |
| **Zero-Configuration Ensembling** | Call `.fit(X, y)` and automatically evaluate, rank, and blend all compatible models mathematically. |
| **Leaderboard Profiling** | Per-model and ensemble OOF (Out-Of-Fold) scores ranked side-by-side. |
| **Task Auto-Detection** | Infers binary classification, multi-class classification, or regression directly from the target tensor. |

---

## 🧠 Supported Architectures

The framework is divided into five core algorithmic families, all accessible out-of-the-box.

### Tree Ensembles & Classic ML
*   **LightGBM (`lgb`)**, **XGBoost (`xgb`)**, **CatBoost (`catboost`)**
*   **Random Forest (`rf`)**, **Extra Trees (`et`)**
*   **Linear / Logistic Regression (`linear`)**, **SVM (`svm`)**
*   **KNN (`knn`)**, **Naive Bayes (`naive_bayes`)**

### Deep Tabular Neural Networks
*   **MLP (`mlp`)**
*   **TabNet (`tabnet`)**: Sparse attention structure.
*   **NODE (`node`)**: Neural Oblivious Decision Ensembles.
*   **TreeNN (`tree_nn`)**: Soft, differentiable decision trees.
*   **TabFM (`tabfm`)**: PyTorch Factorization Machines.

### Sequence, Graph & Transformer Models
*   **FT-Transformer (`ft_transformer`)**, **TabTransformer (`tab_transformer`)**, **SAINT (`saint`)**, **Temporal Transformer (`temporal_transformer`)**
*   **RNNs**: **LSTM (`lstm`)**, **BiLSTM (`bilstm`)**, **GRU (`gru`)**, **BiGRU (`bigru`)**
*   **CNNs**: **Tabular 1D CNN (`cnn1d`)**, **Tabular 2D CNN (`cnn2d`)**
*   **GNNs**: **Graph Neural Networks (`gnn`)**

### Rule-Based & Propositional Extractors
*   **RuleFit (`rulefit`)**: Random stumps evaluated via Coordinate Descent L1 sparsity.
*   **RIPPER (`ripper`)**: Incremental prune and grow rule induction.
*   **Propositional Rule Extractor (`pre`)**: Greedy condition boundary extraction.

### Evolutionary & Metaheuristics
*   **Simulated Annealing Model (`sa`)**: Thermodynamic linear weight convergence.
*   **Genetic Algorithm Model (`ga`)**: Genetic crossover and uniform mutation searches.
*   **Learning Classifier Systems (`lcs`)**: Rule-based reward matching algorithms.
*   **Symbolic Regression (`symbolic_regression`)**: AST/Genetic Programming evaluating raw mathematical syntax (+, -, *, /) to fit labels.

---

## 📦 Installation

To install the latest version directly from GitHub, run:

```bash
pip install git+https://github.com/kaarse/omnitabular.git
```

---

## ⚡ Quick Start

```python
from omnitabular import OmniTabular
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# Start with three fast models; omit models to train all compatible architectures
model = OmniTabular(
    task="classification",
    models=["lgb", "rf", "linear"],
    verbose=True,
    # You can pass customized arguments to ANY model via [name]_params
    lgb_params={"n_estimators": 200},
    mlp_params={"epochs": 50},
    sa_params={"max_iter": 500}
)
model.fit(X, y)

print(model.leaderboard())
print("Ensemble Predictions:", model.predict(X[:5]))
```

**Sample Output:**

```
[OmniTabular] Task detected: classification
[OmniTabular] Optimising metric: log_loss
[OmniTabular] Generating OOF predictions for LGB ...
...
[OmniTabular] Generating OOF predictions for SYMBOLIC_REGRESSION ...
[OmniTabular] Running Hill Climbing to optimise ensemble weights ...
[OmniTabular] Ensemble OOF log_loss: 0.047913
[OmniTabular] Optimal weights → LGB=0.5512, TABNET=0.2104, KNN=0.1001, ...

      model  oof_log_loss  weight
   ENSEMBLE      0.047913    1.0000
        LGB      0.054208    0.5512
     TABNET      0.061102    0.2104
...
```

---

## 📖 API Reference

### `OmniTabular` Params

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | str | `"auto"` | `"classification"`, `"regression"`, or `"auto"` |
| `metric` | str\|None | None | OOF evaluation cross-validation metric (e.g. `"log_loss"`, `"rmse"`) |
| `cv_folds` | int | 5 | Number of stratified OOF CV folds used to generate ensemble weightings |
| `random_state` | int | 42 | Reproducibility seed for all sub-components |
| `verbose` | bool | True | Print active training progress to STDOUT |
| `hc_params` | dict | `{}` | Arguments passed to the backend HillClimbingEnsemble optimizer |
| `[model]_params` | dict | `{}` | Replace `[model]` with any internal architecture name (e.g. `xgb_params`, `mlp_params`, `rulefit_params`, `lcs_params`) to directly feed initialization arguments into the specific `BaseModel` variant |

**Methods:**
- `.fit(X, y)`
- `.predict(X)`
- `.predict_proba(X)`
- `.leaderboard() -> pd.DataFrame`
- `.feature_importances() -> pd.DataFrame` (Maps to LightGBM core branch)

---

## 🧪 Running Tests

A comprehensive integration test suite verifies dimensions, weight sums, multi-class handling, and auto-detection schemas using `pytest`.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```


## Running locally

From the repository root (Python 3.10+ recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,api]"
python -m pytest tests/ -q
```

For a smaller CPU-only installation, install PyTorch first with
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`,
then install the project as above.

Start the API in one terminal:

```bash
source .venv/bin/activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8080
```

Interactive API documentation is at `http://localhost:8080/docs`.
The API trains LightGBM, random forest and linear models by default. JSON
`POST /train` accepts a `models` list and `cv_folds`; `POST /train_upload`
accepts a CSV file, `target_col`, and optional `task` as multipart form data.
`POST /predict` accepts `{"data": [[1, "category"]]}` with columns in training
order, and optional `return_proba: true` for classification.

Start the website in another terminal (Node.js 20.9+):

```bash
cd website
npm ci
npm run dev
```

Open `http://localhost:3000/dashboard` to upload a CSV and train an ensemble.
Set `NEXT_PUBLIC_API_URL` before starting or building the website to use a
remote API. Set `OMNITABULAR_CORS_ORIGINS` on the API to the website origin
(comma-separated for multiple origins). The model is saved to `model.pkl`;
set `OMNITABULAR_MODEL_PATH` to change its location. Run one API worker because
model state is held in memory; restart the service to reload the saved model.

To build the API container, run from the repository root:

```bash
docker build -f api/Dockerfile -t omnitabular .
docker run --rm -p 8080:8080 omnitabular
```

## Training controls

- `models=["lgb", "rf", "linear"]` selects a manageable starting ensemble.
  `OmniTabular.available_models()` lists all 32 options. Training every model
  across five folds can take substantially longer.
- `n_jobs=1` is the default to avoid creating a process for every CPU core.
  Increase it deliberately if memory permits; individual tree models use one
  thread unless overridden through their parameter dictionaries.
- Classification preserves string and nonconsecutive numeric target labels.
  Use explicit `task="regression"` for numeric targets that the automatic
  heuristic could interpret as classes.
- Preprocessing is fitted separately inside each validation fold. Temporal
  validation scores only rows included in validation folds. Every classification
  training fold must contain all target classes.
- Ensemble weights are selected using cross-validation scores. Evaluate on a
  separate test set to estimate performance on new data.
