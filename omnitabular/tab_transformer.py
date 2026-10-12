"""TabTransformer BaseModel implementation."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model, NumericalTokenizer

class TabTransformer(nn.Module):
    def __init__(
        self, 
        num_features: int, 
        output_dim: int, 
        embed_dim: int = 32, 
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()
        # In this continuous-only adaptation of TabTransformer, we treat all features 
        # as having a 'categorical' style embeddings (via tokenizer) passed through 
        # the transformer, and also pass the raw continuous features through LayerNorm
        self.tokenizer = NumericalTokenizer(num_features, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.continuous_ln = nn.LayerNorm(num_features)
        
        # Output of transformer is B x N x E. Flattened is B x (N*E)
        # Plus the raw continuous features is B x N
        mlp_input_dim = (num_features * embed_dim) + num_features
        
        layers = []
        in_d = mlp_input_dim
        for h in mlp_hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: [B, N]
        tokens = self.tokenizer(x) # [B, N, E]
        
        trans_out = self.transformer(tokens) # [B, N, E]
        trans_out = trans_out.flatten(1) # [B, N*E]
        
        cont_out = self.continuous_ln(x) # [B, N]
        
        # Concatenate transformer output and continuous output
        combined = torch.cat((trans_out, cont_out), dim=1) # [B, N*E + N]
        
        return self.mlp(combined)

class TabTransformerModel(BaseModel):
    """
    TabTransformer wrapper.
    """
    def __init__(
        self, 
        task: str = "auto",
        embed_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden_dims: List[int] = [128, 64],
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 10,
        random_state: int = 42,
        verbose: bool = False
    ):
        super().__init__(task=task)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mlp_hidden_dims = mlp_hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "TabTransformerModel":
        X = self._to_numpy(X)
        y = self._prepare_target(y)
        torch.manual_seed(self.random_state)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = TabTransformer(
            num_features=input_dim, 
            output_dim=output_dim, 
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            mlp_hidden_dims=self.mlp_hidden_dims
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
