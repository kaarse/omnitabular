# Troubleshooting

## The dashboard cannot reach the API

Start the service from the repository root with `python -m uvicorn api.main:app --port 8080`. Check `http://localhost:8080/ping`. The website defaults to that local API; set `NEXT_PUBLIC_API_URL` before starting or building to change it.

For a remote website, localhost refers to the visitor's computer. Use a reachable HTTPS API URL for an HTTPS website and include the website origin in `OMNITABULAR_CORS_ORIGINS`. GitHub Pages does not run the Python API.

## The page asks for an admin login

OmniTabular has no built-in login page, administrator account, or default password. A login prompt belongs to the hosting layer or another service. Check the URL and that service's access configuration.

## Too few samples per class

The dashboard uses three-fold validation, requiring at least three samples in each target class. For a continuous target such as price, select regression explicitly. For smaller classification datasets, configure `cv_folds` through the JSON API or Python library. Every classification training fold must still contain all classes.

## Missing OpenMP library

LightGBM can require a system OpenMP library. On Debian / Ubuntu this is commonly `libgomp1`. The provided Docker image installs it.

## Training uses too much time or memory

Select a small ensemble such as `models=["lgb", "rf", "linear"]`. Omitting `models` in Python trains all compatible architectures. Keep `n_jobs=1` initially and reduce neural epochs or evolutionary iteration counts when experimenting. Changing only `mlp_params` does not select MLP; include `mlp` in `models`.

## Prediction column errors

For DataFrames, use the same named columns as training. For arrays and JSON rows, preserve the feature order used for training and exclude the target column. Categories unseen during training are ignored by the encoder; missing values are imputed.

## The trained model disappears after restart

Set `OMNITABULAR_MODEL_PATH` to a persistent filesystem path. For Docker, mount a volume and point the model path into it. Use one API worker so training and prediction share the same active model. A new training job replaces the active model.

## Python reference data is stale

Run `python scripts/build_docs.py`, then include `website/src/data/api-reference.json` with the Python API changes. Check with `python scripts/build_docs.py --check`. The generator does not overwrite the website page or guides.
