"""Recurrent Neural Networks (RNN) models implementation."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .base import BaseModel
from .dl_utils import train_pytorch_model, predict_pytorch_model, NumericalTokenizer

class TabularRNN(nn.Module):
    def __init__(self, num_features: int, output_dim: int, rnn_type: str = "lstm", bidirectional: bool = False, embed_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = NumericalTokenizer(num_features, embed_dim)
        
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=embed_dim, 
                hidden_size=hidden_dim, 
                num_layers=num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim, 
                hidden_size=hidden_dim, 
                num_layers=num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError("rnn_type must be either 'lstm' or 'gru'")
            
        direction_factor = 2 if bidirectional else 1
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * direction_factor, output_dim)
        )
        
    def forward(self, x):
        # x: [B, N]
        embeddings = self.tokenizer(x) # [B, N, E]
        
        # rnn_out: [B, N, D*H]
        rnn_out, _ = self.rnn(embeddings)
        
        # Take the final output across the sequence (the last feature)
        # Note: For tabular data, treating features as sequence might not have natural ordering.
        # Mean pooling over the sequence might be better than taking the last hidden state.
        pooled = torch.mean(rnn_out, dim=1) # [B, D*H]
        
        out = self.fc(pooled)
        return out


class RNNBaseModel(BaseModel):
    """
    Base wrapper for Tabular RNN models.
    """
    def __init__(
        self, 
        task: str = "auto",
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        embed_dim: int = 16,
        hidden_dim: int = 64,
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
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        X = self._to_numpy(X)
        y = np.asarray(y)
        
        input_dim = X.shape[1]
        
        if self.task == "classification":
            output_dim = len(np.unique(y))
        else:
            output_dim = 1
            
        self.model = TabularRNN(
            num_features=input_dim, 
            output_dim=output_dim, 
            rnn_type=self.rnn_type,
            bidirectional=self.bidirectional,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
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


class LSTMModel(RNNBaseModel):
    def __init__(self, task: str = "auto", **kwargs):
        super().__init__(task=task, rnn_type="lstm", bidirectional=False, **kwargs)

class BiLSTMModel(RNNBaseModel):
    def __init__(self, task: str = "auto", **kwargs):
        super().__init__(task=task, rnn_type="lstm", bidirectional=True, **kwargs)
        
class GRUModel(RNNBaseModel):
    def __init__(self, task: str = "auto", **kwargs):
        super().__init__(task=task, rnn_type="gru", bidirectional=False, **kwargs)

class BiGRUModel(RNNBaseModel):
    def __init__(self, task: str = "auto", **kwargs):
        super().__init__(task=task, rnn_type="gru", bidirectional=True, **kwargs)
