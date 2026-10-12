# REST API

Start from the repository root after installing `.[api]`:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8080
```

Base URL: `http://localhost:8080`. Interactive schemas: `/docs`. The service has one active model and no built-in authentication.

## Status

```bash
curl http://localhost:8080/ping
```

Returns `{"status": "ok", "model_loaded": false}` before a model is loaded or trained.

## Train with JSON

`POST /train` accepts feature rows `X`, targets `y`, optional `task` (`auto`, `classification`, `regression`), `cv_folds` (default 3), and `models` (default `lgb`, `rf`, `linear`). This complete small example overrides the model selection and folds:

```bash
curl -X POST http://localhost:8080/train \
  -H 'Content-Type: application/json' \
  -d '{"X": [[0], [1], [2], [3], [4], [5]], "y": [0, 1, 0, 1, 0, 1], "task": "classification", "cv_folds": 2, "models": ["linear"]}'
```

## Train from CSV

Prepare `data.csv`, for example:

```csv
number,category,target
0,a,0
1,b,1
2,a,0
3,b,1
4,a,0
5,b,1
```

```bash
curl -X POST http://localhost:8080/train_upload \
  -F 'file=@data.csv' \
  -F 'target_col=target' \
  -F 'task=classification'
```

`task` defaults to `auto`. This endpoint uses 3 folds and the default three-model ensemble. Use the JSON endpoint or Python library to select other models and folds.

Training returns `status`, `task`, `ensemble_oof_score`, and `weights`. For example (illustrative score and weights):

```json
{
  "status": "success",
  "task": "classification",
  "ensemble_oof_score": 0.42,
  "weights": {"lgb": 0.5, "rf": 0.3, "linear": 0.2}
}
```

The default score is log loss for classification and RMSE for regression; lower is better. These are the same validation scores used to choose ensemble weights.

## Predict

After the CSV training example above:

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"data": [[0, "a"], [1, "b"]], "return_proba": false}'
```

The `data` rows contain features only, in the original training order. The response is `{"predictions": [...]}`. For classification, `return_proba: true` returns a probability matrix instead of labels. Probability columns use the trained model's `classes_` order; this API response does not currently include class names. Regression does not support probability predictions.

## Errors and persistence

| Status | Meaning |
| --- | --- |
| 400 | Invalid dataset, missing CSV target, no loaded model, invalid prediction shape, or regression probability request. |
| 409 | Another training request is already running in this API process. |
| 422 | Request fields fail schema validation, e.g. an unsupported task. |
| 500 | Unexpected service or filesystem failure. |

Training saves the new model atomically before replacing the active model. `OMNITABULAR_MODEL_PATH` defaults to `model.pkl`. The model is loaded at service startup; use one API worker because active models and the training lock are process-local.

`OMNITABULAR_CORS_ORIGINS` is a comma-separated list of website origins. The default allows the local website on port 3000. For remote hosting, expose the API at a reachable HTTPS address and set the website's `NEXT_PUBLIC_API_URL` before building. See the [hosting guide](../README.md#hosting).
