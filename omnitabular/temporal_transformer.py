"""Temporal Transformer models implementation for Tabular Data."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict
import math

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model, NumericalTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, E]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, num_features: int, output_dim: int, embed_dim: int = 32, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = NumericalTokenizer(num_features, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=num_features)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )
        
    def forward(self, x):
        # x: [B, N]
        x = self.tokenizer(x) # [B, N, E]
        x = self.pos_encoder(x) # Add sequential info
        
        # We process it as sequence of features (a bit abstract for tabular but serves as sequence representation)
        # In temporal transformer, we assume causal masking or no masking depending on context.
        # Since it's tabular features, no mask is needed as order doesn't define causal timeline unless specified.
        out = self.transformer_encoder(x)
        
        # Mean pooling across features
        pooled = torch.mean(out, dim=1) # [B, E]
        return self.fc(pooled)

class TemporalTransformerModel(BaseModel):
    def __init__(
        self, 
        task: str = "auto",
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
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
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "TemporalTransformerModel":
        X = self._to_numpy(X)
        y = np.asarray(y)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = TemporalTransformer(
            num_features=input_dim, 
            output_dim=output_dim, 
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
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
        return predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=False)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification.")
        X = self._to_numpy(X)
        return predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=True)
