"""Deep Learning Utilities for OmniTabular."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Optional, List, Dict, Any, Tuple
import math

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, task: str = "auto"):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None
        self.task = task
        if y is not None:
            if self.task == "classification":
                self.y = torch.tensor(y, dtype=torch.long)
            else:
                self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")

class EarlyStopper:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "min" and score > self.best_score) or \
             (self.mode == "max" and score < self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_pytorch_model(
    model: nn.Module, 
    X: np.ndarray, 
    y: np.ndarray, 
    task: str, 
    epochs: int = 100, 
    batch_size: int = 256, 
    lr: float = 1e-3, 
    patience: int = 10,
    val_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False
):
    device = get_device()
    model = model.to(device)
    
    if len(y) < 2:
        raise ValueError("Neural models require at least two training samples.")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for batch normalization.")
    if len(y) < 4:
        # Tiny datasets cannot support both a holdout and batch normalization.
        X_train, X_val, y_train, y_val = X, X, y, y
    else:
        test_count = max(1, int(np.ceil(len(y) * val_size)))
        stratify = None
        if task == "classification":
            _, counts = np.unique(y, return_counts=True)
            test_count = max(test_count, len(counts))
            if counts.min() >= 2 and len(y) - test_count >= len(counts):
                stratify = y
        test_count = min(test_count, len(y) - 2)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_count, random_state=random_state, stratify=stratify
        )

    train_dataset = TabularDataset(X_train, y_train, task)
    val_dataset = TabularDataset(X_val, y_val, task)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(len(train_dataset) % batch_size == 1))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    stopper = EarlyStopper(patience=patience, mode="min")
    best_state_dict = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            if task == "classification":
                loss = criterion(outputs, batch_y)
            else:
                loss = criterion(outputs.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_dataset)
        
        # Eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                if task == "classification":
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(-1), batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_dataset)
        
        if stopper.best_score is None or val_loss < stopper.best_score:
            import copy
            best_state_dict = copy.deepcopy(model.state_dict())
            
        stopper(val_loss)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
            
        if stopper.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
            
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    return model

def predict_pytorch_model(model: nn.Module, X: np.ndarray, task: str, batch_size: int = 512, proba: bool = False):
    device = get_device()
    model = model.to(device)
    model.eval()
    
    dataset = TabularDataset(X, task=task)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for batch_X in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            if task == "classification":
                if proba:
                    probs = torch.softmax(outputs, dim=-1)
                    preds.append(probs.cpu().numpy())
                else:
                    preds.append(outputs.argmax(dim=-1).cpu().numpy())
            else:
                preds.append(outputs.squeeze(-1).cpu().numpy())
                
    return np.concatenate(preds, axis=0)

class NumericalTokenizer(nn.Module):
    """Tokenizes numerical features by multiplying them with a weight vector + bias."""
    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.Tensor(num_features, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(num_features, embed_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N]
        assert x.dim() == 2
        # outputs: [B, N, E]
        x_expanded = x.unsqueeze(-1)  # [B, N, 1]
        out = x_expanded * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return out
