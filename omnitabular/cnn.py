"""Convolutional Neural Networks (CNN) models implementation for Tabular Data."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict
import math

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model

class TabularCNN1D(nn.Module):
    def __init__(self, num_features: int, output_dim: int, channels: List[int] = [16, 32], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_c = 1 # We treat the 1D tabular vector as having 1 channel
        
        current_len = num_features
        for out_c in channels:
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_c = out_c
            current_len = current_len // 2
            
        self.conv_net = nn.Sequential(*layers)
        
        flattened_dim = in_c * current_len
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, output_dim)
        )
        
    def forward(self, x):
        # x is [B, N] -> reshape to [B, 1, N]
        x = x.unsqueeze(1)
        x = self.conv_net(x)
        return self.fc(x)

class TabularCNN2D(nn.Module):
    def __init__(self, num_features: int, output_dim: int, channels: List[int] = [16, 32], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        # Calculate square image dimensions by padding
        self.grid_size = math.ceil(math.sqrt(num_features))
        self.pad_size = (self.grid_size ** 2) - num_features
        
        layers = []
        in_c = 1
        current_grid = self.grid_size
        
        for out_c in channels:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_c = out_c
            current_grid = current_grid // 2
            
        self.conv_net = nn.Sequential(*layers)
        
        flattened_dim = in_c * (current_grid ** 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flattened_dim, output_dim)
        )
        
    def forward(self, x):
        B = x.size(0)
        # Pad sequence so it can be reshaped into a square grid
        if self.pad_size > 0:
            pad = torch.zeros(B, self.pad_size, device=x.device)
            x = torch.cat([x, pad], dim=1)
            
        # Reshape to [B, 1, H, W]
        x = x.view(B, 1, self.grid_size, self.grid_size)
        x = self.conv_net(x)
        return self.fc(x)

class CNN1DModel(BaseModel):
    def __init__(
        self, 
        task: str = "auto",
        channels: List[int] = [16, 32],
        kernel_size: int = 3,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
        random_state: int = 42,
        verbose: bool = False
    ):
        super().__init__(task=task)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "CNN1DModel":
        X = self._to_numpy(X)
        y = np.asarray(y)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = TabularCNN1D(
            num_features=input_dim, 
            output_dim=output_dim, 
            channels=self.channels,
            kernel_size=self.kernel_size,
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

class CNN2DModel(BaseModel):
    def __init__(
        self, 
        task: str = "auto",
        channels: List[int] = [16, 32],
        kernel_size: int = 3,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
        random_state: int = 42,
        verbose: bool = False
    ):
        super().__init__(task=task)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "CNN2DModel":
        X = self._to_numpy(X)
        y = np.asarray(y)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = TabularCNN2D(
            num_features=input_dim, 
            output_dim=output_dim, 
            channels=self.channels,
            kernel_size=self.kernel_size,
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
