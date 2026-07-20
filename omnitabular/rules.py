"""Rule-based Models (RuleFit, RIPPER, PRE)."""

import numpy as np
from typing import Optional, List, Dict
from collections import defaultdict
import copy

from .base import BaseModel

# ── helper for Lasso ──
def _coordinate_descent_lasso(X, y, alpha=0.1, max_iter=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    for _ in range(max_iter):
        for j in range(n_features):
            # calculate residual without feature j
            r_j = y - X @ w + w[j] * X[:, j]
            rho_j = X[:, j] @ r_j
            z_j = np.sum(X[:, j]**2)
            if z_j == 0:
                w[j] = 0
                continue
            
            # soft thresholding
            if rho_j < -alpha * n_samples:
                w[j] = (rho_j + alpha * n_samples) / z_j
            elif rho_j > alpha * n_samples:
                w[j] = (rho_j - alpha * n_samples) / z_j
            else:
                w[j] = 0
    return w

# ── 1. RuleFit ───────────────────────────────────────────────────────

class RuleFitModel(BaseModel):
    """
    RuleFit generates simple splits (rules) and fits a linear model over them.
    Implemented from scratch using basic random stumps and coordinate descent.
    """
    def __init__(self, task="auto", n_rules=50, alpha=0.1, max_iter=100, random_state=42, **kwargs):
        super().__init__(task)
        self.n_rules = n_rules
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.rules = [] # list of (feature_idx, threshold, sign)
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = self._to_numpy(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y) if self.task == "classification" else None
        is_multi = self.classes_ is not None and len(self.classes_) > 2
        
        # 1. Generate rules (random stumps)
        self.rules = []
        for _ in range(self.n_rules):
            f_idx = rng.randint(n_features)
            # Pick a random threshold between min and max
            f_min, f_max = np.min(X[:, f_idx]), np.max(X[:, f_idx])
            if f_min == f_max:
                continue
            thresh = rng.uniform(f_min, f_max)
            sign = rng.choice([1, -1])
            self.rules.append((f_idx, thresh, sign))
            
        # 2. Extract rule features
        # X_rules shape: (n_samples, n_rules + n_features)
        X_rules = np.zeros((n_samples, len(self.rules) + n_features))
        X_rules[:, :n_features] = X
        for i, (f_idx, thresh, sign) in enumerate(self.rules):
            if sign == 1:
                X_rules[:, n_features + i] = (X[:, f_idx] >= thresh).astype(float)
            else:
                X_rules[:, n_features + i] = (X[:, f_idx] < thresh).astype(float)

        # Standardize features for lasso
        self.means = np.mean(X_rules, axis=0)
        self.stds = np.std(X_rules, axis=0)
        self.stds[self.stds == 0] = 1.0
        X_rules_norm = (X_rules - self.means) / self.stds

        # 3. Fit linear model
        if not is_multi:
            # Binary or Regression
            y_target = y if self.task == "regression" else (y * 2 - 1)
            self.w = _coordinate_descent_lasso(X_rules_norm, y_target, alpha=self.alpha, max_iter=self.max_iter)
            self.b = np.mean(y_target - X_rules_norm @ self.w)
        else:
            # Multi-class One-vs-Rest
            self.w = np.zeros((len(self.classes_), X_rules_norm.shape[1]))
            self.b = np.zeros(len(self.classes_))
            for i, c in enumerate(self.classes_):
                y_c = (y == c).astype(float) * 2 - 1
                self.w[i] = _coordinate_descent_lasso(X_rules_norm, y_c, alpha=self.alpha, max_iter=self.max_iter)
                self.b[i] = np.mean(y_c - X_rules_norm @ self.w[i])
                
        self._fitted = True
        return self

    def _transform(self, X):
        n_samples, n_features = X.shape
        X_rules = np.zeros((n_samples, len(self.rules) + n_features))
        X_rules[:, :n_features] = X
        for i, (f_idx, thresh, sign) in enumerate(self.rules):
            if sign == 1:
                X_rules[:, n_features + i] = (X[:, f_idx] >= thresh).astype(float)
            else:
                X_rules[:, n_features + i] = (X[:, f_idx] < thresh).astype(float)
        return (X_rules - self.means) / self.stds

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        X_feat = self._transform(X)
        if self.task == "regression":
            return X_feat @ self.w + self.b
        else:
            if len(self.classes_) > 2:
                pred = X_feat @ self.w.T + self.b
                return self.classes_[np.argmax(pred, axis=1)]
            else:
                pred = X_feat @ self.w + self.b
                return (pred >= 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError
        X = self._to_numpy(X)
        X_feat = self._transform(X)
        if len(self.classes_) > 2:
            pred = X_feat @ self.w.T + self.b
            exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
            return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        else:
            pred = X_feat @ self.w + self.b
            p = 1 / (1 + np.exp(-pred))
            return np.column_stack([1 - p, p])


# ── 2. RIPPER ────────────────────────────────────────────────────────

class RipperRule:
    def __init__(self):
        self.conditions = [] # list of (f_idx, threshold, sign)
        self.label = None

    def covers(self, X):
        mask = np.ones(X.shape[0], dtype=bool)
        for f_idx, thresh, sign in self.conditions:
            if sign == 1:
                mask &= (X[:, f_idx] >= thresh)
            else:
                mask &= (X[:, f_idx] < thresh)
        return mask

class RipperModel(BaseModel):
    """
    Simplified RIPPER (Repeated Incremental Pruning to Produce Error Reduction).
    """
    def __init__(self, task="auto", max_rules=10, max_conditions=3, random_state=42, **kwargs):
        super().__init__(task)
        self.max_rules = max_rules
        self.max_conditions = max_conditions
        self.random_state = random_state
        self.rules = []
        self.default_label = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = self._to_numpy(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y) if self.task == "classification" else None
        
        # Determine default label (majority class or mean)
        if self.task == "classification":
            vals, counts = np.unique(y, return_counts=True)
            self.default_label = vals[np.argmax(counts)]
        else:
            self.default_label = np.mean(y)

        self.rules = []
        remaining_indices = np.arange(n_samples)

        for _ in range(self.max_rules):
            if len(remaining_indices) == 0:
                break
                
            curr_X = X[remaining_indices]
            curr_y = y[remaining_indices]

            # Find majority target in current subset
            if self.task == "classification":
                vals, counts = np.unique(curr_y, return_counts=True)
                target_val = vals[np.argmax(counts)]
                if target_val == self.default_label and len(vals) > 1:
                    target_val = vals[np.argsort(counts)[-2]]
            else:
                target_val = np.mean(curr_y)
                
            rule = RipperRule()
            rule.label = target_val
            
            # Grow
            for _ in range(self.max_conditions):
                best_info_gain = -1
                best_cond = None
                
                # Sample a few random splits to test
                for _ in range(5):
                    f_idx = rng.randint(n_features)
                    thresh = curr_X[rng.randint(len(curr_X)), f_idx]
                    for sign in [1, -1]:
                        if sign == 1:
                            mask = curr_X[:, f_idx] >= thresh
                        else:
                            mask = curr_X[:, f_idx] < thresh
                            
                        # Evaluate split
                        if not np.any(mask):
                            continue
                        
                        if self.task == "classification":
                            acc = np.mean(curr_y[mask] == target_val)
                        else:
                            acc = -np.mean((curr_y[mask] - target_val)**2)
                            
                        if acc > best_info_gain:
                            best_info_gain = acc
                            best_cond = (f_idx, thresh, sign)
                            
                if best_cond is not None:
                    rule.conditions.append(best_cond)
                    
            if len(rule.conditions) > 0:
                self.rules.append(rule)
                # Remove covered
                mask = rule.covers(X)
                remaining_indices = np.where(~mask)[0]
                
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        preds = np.full(X.shape[0], self.default_label)
        
        # Apply rules in order
        for rule in self.rules:
            mask = rule.covers(X)
            # update only those heavily not covered yet or just update all 
            # (standard order-list approach: first rule that matches fires)
            # We'll do reverse order so first rule has precedence
            pass
            
        unassigned = np.ones(X.shape[0], dtype=bool)
        for rule in self.rules:
            mask = rule.covers(X) & unassigned
            preds[mask] = rule.label
            unassigned &= ~mask
            
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError
        preds = self.predict(X)
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, val in enumerate(self.classes_):
            probs[:, i] = (preds == val).astype(float)
        # Smoothing
        probs = (probs + 0.1) / (probs.sum(axis=1, keepdims=True) + 0.1 * len(self.classes_))
        return probs


# ── 3. Propositional Rule Extractor ──────────────────────────────────

class PREModel(BaseModel):
    """
    Propositional Rule Extractor (PRE).
    Greedy set covering algorithm searching propositions.
    """
    def __init__(self, task="auto", max_rules=15, random_state=42, **kwargs):
        super().__init__(task)
        self.max_rules = max_rules
        self.random_state = random_state
        self.rules = []
        self.default_label = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = self._to_numpy(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y) if self.task == "classification" else None
        
        if self.task == "classification":
            vals, counts = np.unique(y, return_counts=True)
            self.default_label = vals[np.argmax(counts)]
        else:
            self.default_label = np.mean(y)

        # Generate proposition bank
        propositions = []
        for f in range(n_features):
            percentiles = np.percentile(X[:, f], [25, 50, 75])
            for p in percentiles:
                propositions.append((f, p, 1))
                propositions.append((f, p, -1))
                
        self.rules = []
        remaining = np.ones(n_samples, dtype=bool)
        
        for _ in range(self.max_rules):
            if not np.any(remaining):
                break
                
            best_prop = None
            best_score = -np.inf
            best_label = None
            
            for f_idx, thresh, sign in propositions:
                if sign == 1:
                    mask = X[:, f_idx] >= thresh
                else:
                    mask = X[:, f_idx] < thresh
                    
                mask = mask & remaining
                if np.sum(mask) == 0:
                    continue
                    
                subset_y = y[mask]
                if self.task == "classification":
                    vals, counts = np.unique(subset_y, return_counts=True)
                    target = vals[np.argmax(counts)]
                    score = np.max(counts) / np.sum(mask) # purity
                else:
                    target = np.mean(subset_y)
                    score = -np.mean((subset_y - target)**2)
                    
                # weight score by coverage slightly
                score = score * np.sqrt(np.sum(mask))
                
                if score > best_score:
                    best_score = score
                    best_prop = [(f_idx, thresh, sign)]
                    best_label = target
                    
            if best_prop is not None:
                r = RipperRule()
                r.conditions = best_prop
                r.label = best_label
                self.rules.append(r)
                remaining &= ~r.covers(X)
            else:
                break
                
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._to_numpy(X)
        preds = np.full(X.shape[0], self.default_label)
        unassigned = np.ones(X.shape[0], dtype=bool)
        for rule in self.rules:
            mask = rule.covers(X) & unassigned
            preds[mask] = rule.label
            unassigned &= ~mask
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task != "classification":
            raise NotImplementedError
        preds = self.predict(X)
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, val in enumerate(self.classes_):
            probs[:, i] = (preds == val).astype(float)
        # Smoothing
        probs = (probs + 0.1) / (probs.sum(axis=1, keepdims=True) + 0.1 * len(self.classes_))
        return probs
