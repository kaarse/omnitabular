"""Tree-mimicking NN (Soft Decision Tree) BaseModel implementation."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model

class SoftDecisionTree(nn.Module):
    def __init__(self, num_features: int, output_dim: int, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_inner = self.num_leaves - 1
        
        # Routing params for inner nodes
        self.W = nn.Parameter(torch.randn(self.num_inner, num_features))
        self.b = nn.Parameter(torch.randn(self.num_inner))
        
        # Outputs for leaf nodes
        self.leaves = nn.Parameter(torch.randn(self.num_leaves, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_normal_(self.W)
        nn.init.zeros_(self.b)
        nn.init.xavier_normal_(self.leaves)
        
    def forward(self, x):
        # x is [B, N]
        B = x.size(0)
        
        # Compute gating probabilities for all inner nodes
        # [B, num_inner]
        gates = torch.sigmoid(torch.matmul(self.dropout(x), self.W.t()) + self.b)
        
        # Compute path probabilities
        path_probs = []
        for i in range(self.num_leaves):
            # Binary representation tells us left(0) or right(1) at each depth
            node_idx = 0
            prob_i = 1.0
            for d in range(self.depth):
                # Is the d-th bit set? (from MSB)
                is_right = (i >> (self.depth - 1 - d)) & 1
                
                if is_right:
                    prob = gates[:, node_idx]
                    node_idx = node_idx * 2 + 2
                else:
                    prob = 1.0 - gates[:, node_idx]
                    node_idx = node_idx * 2 + 1
                    
                prob_i = prob_i * prob
            path_probs.append(prob_i.unsqueeze(1))
                
        # path_probs is [B, num_leaves]
        path_probs = torch.cat(path_probs, dim=1)
        # leaves is [num_leaves, output_dim]
        # output is weighted sum of leaves
        out = torch.matmul(path_probs, self.leaves) # [B, output_dim]
        return out

class TreeNNModel(BaseModel):
    """
    Soft Decision Tree wrapper.
    """
    def __init__(
        self, 
        task: str = "auto",
        depth: int = 4,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
        random_state: int = 42,
        verbose: bool = False
    ):
        super().__init__(task=task)
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "TreeNNModel":
        X = self._to_numpy(X)
        y = self._prepare_target(y)
        torch.manual_seed(self.random_state)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = SoftDecisionTree(
            num_features=input_dim, 
            output_dim=output_dim, 
            depth=self.depth,
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
