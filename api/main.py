"""REST service for training and serving one OmniTabular ensemble."""

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Literal
import io
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from omnitabular import OmniTabular

MODEL_CACHE_PATH = os.getenv("OMNITABULAR_MODEL_PATH", "model.pkl")
_global_model = None
_training_lock = Lock()
Task = Literal["auto", "classification", "regression"]
DEFAULT_MODELS = ["lgb", "rf", "linear"]


@asynccontextmanager
async def lifespan(app):
    global _global_model
    path = Path(MODEL_CACHE_PATH)
    if path.exists():
        with path.open("rb") as stream:
            _global_model = pickle.load(stream)
    yield


app = FastAPI(title="OmniTabular Live API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("OMNITABULAR_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    data: list[list[Any]]
    return_proba: bool = False


class TrainRequest(BaseModel):
    X: list[list[Any]]
    y: list[Any]
    task: Task = "auto"
    cv_folds: int = Field(default=3, ge=2)
    models: list[str] = Field(default_factory=lambda: DEFAULT_MODELS.copy())


def _train(X, y, task, cv_folds=3, models=None):
    global _global_model
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training is already in progress.")
    temporary_path = None
    try:
        model = OmniTabular(task=task, cv_folds=cv_folds, models=DEFAULT_MODELS if models is None else models,
                           verbose=False, n_jobs=1)
        model.fit(X, y)
        path = Path(MODEL_CACHE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(model, stream)
        os.replace(temporary_path, path)
        _global_model = model
        return {
            "status": "success",
            "task": model.task_,
            "ensemble_oof_score": float(model.ensemble_oof_score_),
            "weights": {name: float(weight) for name, weight in zip(model.model_names_, model.weights_)},
        }
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        _training_lock.release()


@app.get("/ping")
def ping():
    return {"status": "ok", "model_loaded": _global_model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    model = _global_model
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not trained or loaded yet.")
    try:
        X = np.asarray(req.data, dtype=object)
        predictions = model.predict_proba(X) if req.return_proba else model.predict(X)
        return {"predictions": predictions.tolist()}
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train")
def train(req: TrainRequest):
    # DataFrame preserves per-column types when numbers and strings are mixed.
    return _train(pd.DataFrame(req.X), np.asarray(req.y), req.task, req.cv_folds, req.models)


@app.post("/train_upload")
def train_upload(file: UploadFile = File(...), target_col: str = Form(...), task: Task = Form("auto")):
    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in CSV.")
    return _train(df.drop(columns=[target_col]), df[target_col].to_numpy(), task)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
