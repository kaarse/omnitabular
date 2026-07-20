"""TabNet BaseModel implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model

class GLUBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.bn = nn.BatchNorm1d(output_dim * 2)
        
    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        # GLU activation
        out = out[:, :out.shape[1]//2] * torch.sigmoid(out[:, out.shape[1]//2:])
        return out

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.shared = GLUBlock(input_dim, output_dim)
        self.independent = GLUBlock(output_dim, output_dim)
        
    def forward(self, x):
        x = self.shared(x)
        x = self.independent(x)
        return x

class TabNet(nn.Module):
    def __init__(self, num_features: int, output_dim: int, n_d: int = 16, n_a: int = 16, n_steps: int = 3, gamma: float = 1.3):
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        self.initial_bn = nn.BatchNorm1d(num_features)
        
        self.feature_transformers = nn.ModuleList([
            FeatureTransformer(num_features, n_d + n_a) for _ in range(n_steps)
        ])
        
        self.attentive_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_a, num_features),
                nn.BatchNorm1d(num_features)
            ) for _ in range(n_steps - 1)
        ])
        
        self.final_mapping = nn.Linear(n_d, output_dim)
        
    def forward(self, x):
        x = self.initial_bn(x)
        B, N = x.shape
        
        prior = torch.ones(B, N, device=x.device)
        out_agg = 0
        
        # Initial step
        feat_out = self.feature_transformers[0](x)
        d = feat_out[:, :self.n_d]
        a = feat_out[:, self.n_d:]
        
        for step in range(1, self.n_steps):
            # Attentive transformer
            mask_values = self.attentive_transformers[step-1](a)
            # Softmax with prior
            mask = F.softmax(mask_values, dim=-1) * prior
            # Update prior
            prior = prior * (self.gamma - mask)
            
            # Mask features
            masked_x = mask * x
            
            # Feature transformer
            feat_out = self.feature_transformers[step](masked_x)
            d = feat_out[:, :self.n_d]
            a = feat_out[:, self.n_d:]
            
            out_agg += F.relu(d)
            
        return self.final_mapping(out_agg)

class TabNetModel(BaseModel):
    """
    TabNet (Sparse sequential attention) wrapper.
    """
    def __init__(
        self, 
        task: str = "auto",
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        gamma: float = 1.3,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 2e-2,  # TabNet often uses larger LR
        patience: int = 15,
        random_state: int = 42,
        verbose: bool = False
    ):
        super().__init__(task=task)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "TabNetModel":
        X = self._to_numpy(X)
        y = np.asarray(y)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = TabNet(
            num_features=input_dim, 
            output_dim=output_dim, 
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma
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
