"""Tabular Factorization Machine (TabFM) BaseModel implementation."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model, NumericalTokenizer

class TabularFM(nn.Module):
    def __init__(self, num_features: int, output_dim: int, embed_dim: int = 16, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        
        # 1. First-order interactions (Linear)
        self.first_order = nn.Linear(num_features, output_dim)
        
        # 2. Second-order interactions (FM based on feature embeddings)
        self.tokenizer = NumericalTokenizer(num_features, embed_dim)
        self.fm_head = nn.Linear(embed_dim, output_dim)
        
        # 3. Deep component (MLP)
        layers = []
        in_d = num_features * embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # 1. Linear part
        # [B, output_dim]
        out_linear = self.first_order(x)
        
        # 2. FM part
        # embeddings: [B, N, E]
        embeddings = self.tokenizer(x)
        
        # sum_square: [B, E]
        sum_square = torch.pow(torch.sum(embeddings, dim=1), 2)
        # square_sum: [B, E]
        square_sum = torch.sum(torch.pow(embeddings, 2), dim=1)
        
        # fm_out: [B, E]
        fm_out = 0.5 * (sum_square - square_sum)
        # [B, output_dim]
        out_fm = self.fm_head(fm_out)
        
        # 3. Deep part
        # [B, N * E]
        deep_in = embeddings.flatten(1)
        # [B, output_dim]
        out_deep = self.mlp(deep_in)
        
        return out_linear + out_fm + out_deep

class TabFMModel(BaseModel):
    """
    TabFM (PyTorch DeepFM adaptation) wrapper.
    """
    def __init__(
        self, 
        task: str = "auto",
        embed_dim: int = 16,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
        random_state: int = 42,
        verbose: bool = False
    ):
        super().__init__(task=task)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "TabFMModel":
        X = self._to_numpy(X)
        y = self._prepare_target(y)
        torch.manual_seed(self.random_state)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = TabularFM(
            num_features=input_dim, 
            output_dim=output_dim, 
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        )
        
        self.model = train_pytorch_model(
            self.model, X, y, self.task,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self._fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        predictions = predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=False)
        return self.classes_[predictions.astype(int)] if self.task == "classification" else predictions
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification.")
        X = self._to_numpy(X)
        return predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=True)
