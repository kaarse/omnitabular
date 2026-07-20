"""SAINT BaseModel implementation."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model, NumericalTokenizer

class SAINTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_features: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Feature attention (seq_len = num_features)
        self.feature_att = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        
        # Intersample attention (seq_len = batch_size)
        row_dim = embed_dim * num_features
        self.row_att = nn.TransformerEncoderLayer(
            d_model=row_dim,
            # Adjust heads if row_dim is not divisible by num_heads
            nhead=num_heads if row_dim % num_heads == 0 else 1,
            dim_feedforward=row_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
    
    def forward(self, x):
        # x: [B, N, E]
        B, N, E = x.shape
        
        # Feature attention
        out = self.feature_att(x) # [B, N, E]
        
        # Intersample attention
        out = out.reshape(1, B, N * E) # [1, B, N*E]
        out = self.row_att(out) # [1, B, N*E]
        
        out = out.reshape(B, N, E) # [B, N, E]
        return out

class SAINT(nn.Module):
    def __init__(
        self, 
        num_features: int, 
        output_dim: int, 
        embed_dim: int = 16, 
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tokenizer = NumericalTokenizer(num_features, embed_dim)
        
        # CLS token explicitly for SAINT (treated as an extra feature)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            SAINTBlock(embed_dim, num_features + 1, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
        
    def forward(self, x):
        B = x.size(0)
        tokens = self.tokenizer(x) # [B, N, E]
        
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, E]
        x = torch.cat((cls_tokens, tokens), dim=1) # [B, N+1, E]
        
        for block in self.blocks:
            x = block(x)
            
        cls_out = x[:, 0, :]
        cls_out = self.ln(cls_out)
        out = self.head(cls_out)
        return out

class SAINTModel(BaseModel):
    """
    SAINT model wrapper.
    """
    def __init__(
        self, 
        task: str = "auto",
        embed_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-4,
        patience: int = 10,
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

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "SAINTModel":
        X = self._to_numpy(X)
        y = np.asarray(y)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = SAINT(
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
