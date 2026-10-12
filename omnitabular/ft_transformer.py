"""FT-Transformer BaseModel implementation."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model, NumericalTokenizer

class FTTransformer(nn.Module):
    def __init__(
        self, 
        num_features: int, 
        output_dim: int, 
        embed_dim: int = 192, 
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tokenizer = NumericalTokenizer(num_features, embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
        
    def forward(self, x):
        # x: [B, N]
        B = x.size(0)
        tokens = self.tokenizer(x) # [B, N, E]
        
        # prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, E]
        
        x = torch.cat((cls_tokens, tokens), dim=1) # [B, N+1, E]
        
        # transformer
        x = self.transformer(x) # [B, N+1, E]
        
        # extract cls representation
        cls_out = x[:, 0, :]
        
        cls_out = self.ln(cls_out)
        out = self.head(cls_out)
        return out

class FTTransformerModel(BaseModel):
    """
    FT-Transformer wrapper. Continues features only.
    """
    def __init__(
        self, 
        task: str = "auto",
        embed_dim: int = 192,
        num_heads: int = 8,
        num_layers: int = 3,
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

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "FTTransformerModel":
        X = self._to_numpy(X)
        y = self._prepare_target(y)
        torch.manual_seed(self.random_state)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = FTTransformer(
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
        predictions = predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=False)
        return self.classes_[predictions.astype(int)] if self.task == "classification" else predictions
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification.")
        X = self._to_numpy(X)
        return predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=True)
