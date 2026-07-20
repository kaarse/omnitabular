"""Graph Neural Networks (GNN) model implementation for Tabular Data."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import math

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model, NumericalTokenizer

class TabularGCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [B, N, in_features]
        # adj: [N, N]
        
        # Apply dropout to features
        x = self.dropout(x)
        
        # Linear transformation
        # [B, N, in_features] @ [in_features, out_features] -> [B, N, out_features]
        support = torch.matmul(x, self.weight)
        
        # Graph convolution
        # adj: [N, N], support: [B, N, out_features]
        # output: [B, N, out_features] -> (adj @ support)
        # We need to broadcast adj over batch
        output = torch.einsum('nm,bmd->bnd', adj, support)
        return output + self.bias

class TabularGNN(nn.Module):
    def __init__(self, num_features: int, output_dim: int, embed_dim: int = 16, hidden_dims: List[int] = [32, 16], dropout: float = 0.1):
        super().__init__()
        self.tokenizer = NumericalTokenizer(num_features, embed_dim)
        
        # Learnable Adjacency Matrix representing feature correlations
        self.adj = nn.Parameter(torch.Tensor(num_features, num_features))
        nn.init.xavier_uniform_(self.adj)
        
        self.layers = nn.ModuleList()
        in_dim = embed_dim
        for h in hidden_dims:
            self.layers.append(TabularGCNLayer(in_dim, h, dropout))
            in_dim = h
            
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim * num_features, output_dim)
        )
        
    def forward(self, x):
        # x: [B, N]
        x = self.tokenizer(x) # [B, N, E]
        
        # Symmetrize adjacency matrix and apply softmax for normalized weights
        # adj is [N, N]
        adj = (self.adj + self.adj.t()) / 2
        # Normalize rows to sum to 1
        adj = F.softmax(adj, dim=-1)
        
        for layer in self.layers:
            x = layer(x, adj)
            x = F.relu(x)
            
        # x: [B, N, hidden_dim]
        # Flatten all feature node embeddings
        x = x.flatten(1) # [B, N * hidden_dim]
        return self.fc(x)

class GNNModel(BaseModel):
    def __init__(
        self, 
        task: str = "auto",
        embed_dim: int = 16,
        hidden_dims: List[int] = [32, 16],
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

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "GNNModel":
        X = self._to_numpy(X)
        y = np.asarray(y)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = TabularGNN(
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
        return predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=False)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification.")
        X = self._to_numpy(X)
        return predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=True)
