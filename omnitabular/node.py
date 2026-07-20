"""NODE (Neural Oblivious Decision Ensembles) BaseModel implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model

class ObliviousDecisionTreeEnsemble(nn.Module):
    def __init__(self, num_features: int, output_dim: int, num_trees: int = 16, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_trees = num_trees
        self.depth = depth
        self.num_leaves = 2 ** depth
        
        # Soft feature selection (num_trees, depth, num_features)
        self.feature_weights = nn.Parameter(torch.randn(num_trees, depth, num_features))
        
        # Thresholds (num_trees, depth)
        self.thresholds = nn.Parameter(torch.randn(num_trees, depth))
        
        # Scales for steepness of the sigmoid (num_trees, depth)
        self.scales = nn.Parameter(torch.ones(num_trees, depth))
        
        # Leaf values (num_trees, num_leaves, output_dim)
        self.leaves = nn.Parameter(torch.randn(num_trees, self.num_leaves, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, N]
        B, N = x.shape
        x = self.dropout(x)
        
        # Feature selection using softmax
        # [T, D, N]
        sel_weights = F.softmax(self.feature_weights, dim=-1)
        
        # Extract features for each tree and depth
        # x: [B, 1, 1, N]  * sel_weights: [1, T, D, N] -> sum over N -> [B, T, D]
        x_selected = torch.einsum('bn,tdn->btd', x, sel_weights)
        
        # Compute gates
        # thresholds: [1, T, D], scales: [1, T, D]
        gates = torch.sigmoid((x_selected - self.thresholds.unsqueeze(0)) * self.scales.unsqueeze(0))
        # gates: [B, T, D]
        
        # Compute path probabilities for all leaves
        path_probs = []
        for i in range(self.num_leaves):
            prob_i = 1.0
            for d in range(self.depth):
                is_right = (i >> (self.depth - 1 - d)) & 1
                gate_d = gates[:, :, d]
                
                if is_right:
                    prob = gate_d
                else:
                    prob = 1.0 - gate_d
                    
                prob_i = prob_i * prob
            path_probs.append(prob_i.unsqueeze(2))
        
        # path_probs is [B, T, num_leaves]
        path_probs = torch.cat(path_probs, dim=2)
                
        # Aggregate leaves
        # path_probs: [B, T, L], leaves: [T, L, O]
        # output: [B, T, O] -> sum over T -> [B, O]
        out = torch.einsum('btl,tlo->bo', path_probs, self.leaves)
        return out

class NODEModel(BaseModel):
    """
    NODE wrapper.
    """
    def __init__(
        self, 
        task: str = "auto",
        num_trees: int = 16,
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
        self.num_trees = num_trees
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "NODEModel":
        X = self._to_numpy(X)
        y = np.asarray(y)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = ObliviousDecisionTreeEnsemble(
            num_features=input_dim, 
            output_dim=output_dim, 
            num_trees=self.num_trees,
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
        return predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=False)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification.")
        X = self._to_numpy(X)
        return predict_pytorch_model(self.model, X, self.task, batch_size=self.batch_size, proba=True)
