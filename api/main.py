from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os
import io
import uvicorn
from typing import List, Dict, Any, Union

from omnitabular import OmniTabular

app = FastAPI(
    title="OmniTabular Live API",
    description="High-speed REST API for running OmniTabular ML predictions.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory model
MODEL_CACHE_PATH = os.getenv("OMNITABULAR_MODEL_PATH", "model.pkl")
_global_model = None

@app.on_event("startup")
def load_model():
    global _global_model
    if os.path.exists(MODEL_CACHE_PATH):
        with open(MODEL_CACHE_PATH, "rb") as f:
            _global_model = pickle.load(f)
        print(f"Loaded OmniTabular model from {MODEL_CACHE_PATH}")
    else:
        print("Warning: No pre-trained model found. Call /train or mount a model.pkl")

class PredictRequest(BaseModel):
    data: List[List[float]]
    return_proba: bool = False

class TrainRequest(BaseModel):
    X: List[List[float]]
    y: List[Union[int, float]]
    task: str = "auto"
    cv_folds: int = 3

@app.get("/ping")
def ping():
    return {"status": "ok", "model_loaded": _global_model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    if _global_model is None:
        raise HTTPException(status_code=400, detail="Model is not trained or loaded yet.")
    
    try:
        X_arr = np.array(req.data)
        if req.return_proba and _global_model.task_ == "classification":
            preds = _global_model.predict_proba(X_arr)
        else:
            preds = _global_model.predict(X_arr)
            
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train(req: TrainRequest):
    global _global_model
    try:
        X_arr = np.array(req.X)
        y_arr = np.array(req.y)
        
        model = OmniTabular(
            task=req.task, 
            cv_folds=req.cv_folds, 
            auto_preprocess=True,
            verbose=False,
            n_jobs=-1
        )
        model.fit(X_arr, y_arr)
        
        _global_model = model
        with open(MODEL_CACHE_PATH, "wb") as f:
            pickle.dump(model, f)
            
        return {
            "status": "success", 
            "task": model.task_,
            "ensemble_oof_score": model.ensemble_oof_score_
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_upload")
async def train_upload(
    file: UploadFile = File(...), 
    target_col: str = Form(...),
    task: str = Form("auto")
):
    global _global_model
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in CSV.")
            
        y_arr = df[target_col].values
        X_df = df.drop(columns=[target_col])
        X_arr = X_df.values
        
        model = OmniTabular(
            task=task,
            auto_preprocess=True,
            verbose=False,
            n_jobs=-1
        )
        model.fit(X_arr, y_arr)
        
        _global_model = model
        with open(MODEL_CACHE_PATH, "wb") as f:
            pickle.dump(model, f)
            
        best_weights = model.ensemble.weights_ if hasattr(model, 'ensemble') and hasattr(model.ensemble, 'weights_') else []
        model_names = list(model.models_.keys())
        
        weights_dict = {}
        if len(best_weights) == len(model_names):
            for n, w in zip(model_names, best_weights):
                if w > 0.001:
                    weights_dict[n] = float(w)
                    
        return {
            "status": "success",
            "task": model.task_,
            "ensemble_oof_score": model.ensemble_oof_score_,
            "weights": weights_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
