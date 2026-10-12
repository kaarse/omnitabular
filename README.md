# OmniTabular

Train, compare, and combine machine learning models for tabular data through one Python API. OmniTabular generates out-of-fold predictions, searches for nonnegative ensemble weights with stochastic hill climbing, then retrains the selected models on the training data.

It includes 32 model options for classification and 31 for regression, a local CSV training dashboard, and a FastAPI service. Start with a small model selection; training all architectures across five folds can take substantially longer.

## Install

The package is not yet published on PyPI. Until the first release is uploaded, `pip install omnitabular` reports “No matching distribution found”. Install through pip from GitHub using the command below, or run `python -m pip install .` from a local checkout.

Python 3.9+ is supported; Python 3.10+ is recommended. Create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install git+https://github.com/kaarse/omnitabular.git
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. On Linux, LightGBM may require a system OpenMP runtime, such as `libgomp1` on Debian / Ubuntu.

For CPU-only use, install PyTorch before the project to avoid downloading its GPU libraries:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick start

Reserve test data before fitting:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from omnitabular import OmniTabular

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = OmniTabular(
    task="classification", models=["lgb", "rf", "linear"]
)
model.fit(X_train, y_train)
print(model.leaderboard())
print(model.predict(X_test[:5]))
```

The leaderboard contains `model`, `oof_log_loss` (or the selected metric), and `weight` columns. Scores and weights vary with data and settings. Cross-validation scores are used to select the blend; evaluate on the held-out test set for an independent assessment.

## Models

Use `OmniTabular.available_models()` for registry names. Pass a subset through `models=[...]`; omitting `models` trains every task-compatible option.

| Family | Registry names |
| --- | --- |
| Trees and classic ML | `lgb`, `xgb`, `catboost`, `rf`, `et`, `linear`, `svm`, `knn`, `naive_bayes` |
| Deep tabular models | `mlp`, `tabnet`, `node`, `tree_nn`, `tabfm` |
| Transformers | `ft_transformer`, `tab_transformer`, `saint`, `temporal_transformer` |
| Recurrent, convolutional, graph | `lstm`, `bilstm`, `gru`, `bigru`, `cnn1d`, `cnn2d`, `gnn` |
| Rule-based | `rulefit`, `ripper`, `pre` |
| Evolutionary | `sa`, `ga`, `lcs`, `symbolic_regression` |

Naive Bayes supports classification only. Sequence and graph options operate on tabular representations; they do not automatically construct a forecasting dataset.

## Configuration

| Parameter | Default | Behavior |
| --- | --- | --- |
| `task` | `"auto"` | `"classification"`, `"regression"`, or automatic detection. Specify regression for numeric targets that could be interpreted as classes. |
| `models` | `None` | All compatible models; start with `["lgb", "rf", "linear"]`. |
| `auto_preprocess` | `True` | Impute missing values, scale numeric features, encode categories inside each validation fold. |
| `metric` | `None` | Defaults to `log_loss` for classification and `rmse` for regression. Also supports `accuracy`, `auc`, `mae`, or a callable. |
| `metric_higher_is_better` | `False` | Direction for a custom metric. Built-in directions are predefined. |
| `metric_needs_proba` | `False` | Whether a custom metric expects probabilities. |
| `cv_folds` | `5` | Validation folds. Stratified classification requires at least this many rows per class. |
| `cv_strategy` | `"kfold"` | `kfold`, `group_kfold`, `time_series`, or `sliding_window`. |
| `window_size` | `None` | Maximum training window for sliding-window validation. |
| `n_jobs` | `1` | Parallel model workers; `-1` requests available CPUs. Increase deliberately if memory permits. |
| `random_state` | `42` | Seed for compatible models, splitting, and optimization. |
| `verbose` | `True` | Training progress. |
| `hc_params` | `None` | Hill-climbing overrides, e.g. `{"n_iterations": 500}`. |
| `[model]_params` | `None` | Options for that model, e.g. `lgb_params={"n_estimators": 200}`. Only selected models are trained. |

Every classification training fold must contain all classes. Group validation requires `fit(X, y, groups=groups)`. Temporal validation preserves row order and scores only validated rows.

### Methods

- `fit(X, y, groups=None)`: train the ensemble.
- `predict(X)`: predicted labels or numeric values.
- `predict_proba(X)`: classification probabilities, with columns in `model.classes_` order.
- `leaderboard()`: individual and ensemble out-of-fold scores and weights.
- `explain(X, y, n_repeats=5)`: permutation importance over original input features.
- `feature_importances()`: LightGBM importances; requires `lgb` in the ensemble. With preprocessing enabled, indices refer to transformed features.

DataFrames retain column names; prediction columns must match training. Arrays must use training feature order. Classification labels may be strings or nonconsecutive numbers.

## Run the dashboard and API locally

From a checkout, install the development and API extras:

```bash
git clone https://github.com/kaarse/omnitabular.git
cd omnitabular
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,api]"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8080
```

In another terminal, from the checkout, start the website with Node.js 20.9+:

```bash
cd website
npm ci
npm run dev
```

Open **http://localhost:3000/dashboard**. There is no login or admin account. Upload a comma-separated CSV with unique header names, choose its target column and task, and train. Uploads use LightGBM, random forest, and linear / logistic regression with 3-fold validation; classification needs at least 3 rows per class.

The API provides interactive documentation at **http://localhost:8080/docs**. See [REST API examples](docs/api.md) for JSON training, CSV uploads, predictions, and responses.

## Hosting

`npm run build` followed by `npm start` runs the website server. `NEXT_OUTPUT_EXPORT=1 npm run build` exports a static site to `website/out/`. Set `NEXT_PUBLIC_BASE_PATH=/omnitabular` before building for that subpath; the Pages workflow sets it automatically.

GitHub Pages serves only the website. Host the Python API separately and set `NEXT_PUBLIC_API_URL` to its reachable address **before the website build**. An HTTPS website needs an HTTPS API. The API defaults to allowing `http://localhost:3000` and `http://127.0.0.1:3000`; configure `OMNITABULAR_CORS_ORIGINS` for other website origins.

The service has no built-in authentication. Use your hosting access controls when restricted access is required. CORS is a browser policy, not authentication. Use a single API worker: the active model is held in process memory, saved to `model.pkl` (or `OMNITABULAR_MODEL_PATH`), and reloaded at startup. Training replaces the active model.

Build Docker from the repository root. Use a volume to preserve the trained model:

```bash
docker build -f api/Dockerfile -t omnitabular .
docker run --rm -p 8080:8080 \
  -e OMNITABULAR_MODEL_PATH=/data/model.pkl \
  -v omnitabular-data:/data omnitabular
```

## Development and verification

```bash
python -m pip install -e ".[dev,api]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/ -q
python examples/demo.py
python scripts/build_docs.py --check
cd website
npm ci
npm test
npm run lint
npm run build
```

The integration suite trains all model families and can take several minutes. API tests require the `api` extra. To update generated website API defaults and method names after changing Python signatures, run `python scripts/build_docs.py` from the repository root. The generator uses Python's standard library and preserves the hand-written website guides.

See [website setup](website/README.md), [troubleshooting](docs/troubleshooting.md), and [publishing releases to PyPI](docs/publishing.md).
